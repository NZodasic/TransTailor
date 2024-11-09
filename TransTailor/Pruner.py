import torch
import torchvision
import torchvision.transforms as transforms
import itertools
import pickle
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Pruner:
    def __init__(self, model, train_loader, device, amount=0.2):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.amount = amount
        self.scaling_factors = {}
        self.importance_scores = {}
        self.pruned_filters = set()

    def InitScalingFactors(self):
        logger.info("Init alpha from scratch!")
        num_layers = len(self.model.features)
        self.scaling_factors = {}

        for i in range(num_layers):
            layer = self.model.features[i]
            if isinstance(layer, torch.nn.Conv2d):
                print(layer, layer.out_channels)
                self.scaling_factors[i] = torch.rand((1, layer.out_channels, 1, 1), requires_grad=True)

    def TrainScalingFactors(self, root, num_epochs, learning_rate, momentum):
        for param in self.model.parameters():
            param.requires_grad = False

        criterion = torch.nn.CrossEntropyLoss()
        num_layers = len(self.model.features)

        logger.info("\n===Train the factors alpha by optimizing the loss function===")

        params_to_optimize = itertools.chain(self.scaling_factors[sf] for sf in self.scaling_factors.keys())
        optimizer_alpha = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum)

        for epoch in range(num_epochs):
            logger.info("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            iter_count = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.shape[0]
                optimizer_alpha.zero_grad()
                outputs = inputs
                outputs.requires_grad = False

                for i in range(num_layers):
                    if isinstance(self.model.features[i], torch.nn.Conv2d):
                        outputs = self.model.features[i](outputs)
                        outputs = outputs * self.scaling_factors[i].cuda()
                    else:
                        outputs = self.model.features[i](outputs)

                outputs = torch.flatten(outputs, 1)
                classification_output = self.model.classifier(outputs)
                loss = criterion(classification_output, labels)
                loss.backward()
                optimizer_alpha.step()

    def GenerateImportanceScores(self):
        print("===Generate importance score===")
        self.importance_scores = {}
        num_layers = len(self.model.features)
        criterion = torch.nn.CrossEntropyLoss()

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = inputs
            for i in range(num_layers):
                if isinstance(self.model.features[i], torch.nn.Conv2d):
                    outputs = self.model.features[i](outputs)
                    outputs = outputs * self.scaling_factors[i].cuda()
                else:
                    outputs = self.model.features[i](outputs)

            outputs = torch.flatten(outputs, 1)
            classification_output = self.model.classifier(outputs)
            loss = criterion(classification_output, labels)

        for i, scaling_factor in self.scaling_factors.items():
            first_order_derivative = torch.autograd.grad(loss, scaling_factor, retain_graph=True)[0]
            self.importance_scores[i] = torch.abs(first_order_derivative * scaling_factor).detach()

    # Apply exponential moving average smoothing
        alpha = 0.9
        if hasattr(self, 'prev_importance_scores'):
            for layer_idx in self.importance_scores:
                self.importance_scores[layer_idx] = \
                    alpha * self.prev_importance_scores[layer_idx] + \
                    (1 - alpha) * self.importance_scores[layer_idx]
        
        self.prev_importance_scores = {k: v.clone() for k, v in self.importance_scores.items()}

    def FindFiltersToPrune(self):
        """
        Identify all filters with the minimum importance score (β) to prune them at once.
        """
        filters_to_prune = {}  # List to store filters to prune

        all_scores = []
        for layer_idx, scores in self.importance_scores.items():
            for filter_idx, score in enumerate(scores[0]):
                all_scores.append((layer_idx, filter_idx, score.item()))
        
        # Sort by score ascending
        all_scores.sort(key=lambda x: x[2])
        
        # Take bottom X% of filters
        prune_count = int(len(all_scores) * 0.05)  # Prune 5% each time
        for layer_idx, filter_idx, _ in all_scores[:prune_count]:
            if layer_idx not in filters_to_prune:
                filters_to_prune[layer_idx] = []
            filters_to_prune[layer_idx].append(filter_idx)


        return filters_to_prune  # Return the list of all filters to prune

    def Prune(self, layer_to_prune, filter_to_prune):
        pruned_layer = self.model.features[layer_to_prune]

        with torch.no_grad():
            pruned_layer.weight.data[filter_to_prune] = 0
            pruned_layer.bias.data[filter_to_prune] = 0

        # After pruning, you can update the pruned_filters set
        self.pruned_filters.add((layer_to_prune, filter_to_prune))

    def PruneSingleFilter(self, layer_index, filter_index):
        conv_layer = self.model.features[layer_index]
        new_conv = torch.nn.Conv2d(in_channels=conv_layer.in_channels,
                                    out_channels=conv_layer.out_channels - 1,
                                    kernel_size=conv_layer.kernel_size,
                                    stride=conv_layer.stride,
                                    padding=conv_layer.padding,
                                    dilation=conv_layer.dilation,
                                    groups=conv_layer.groups,
                                    bias=conv_layer.bias is not None)
        new_filters = torch.cat((conv_layer.weight.data[:filter_index], conv_layer.weight.data[filter_index + 1:]))
        new_conv.weight.data = new_filters

        if conv_layer.bias is not None:
                new_biases = torch.cat((conv_layer.bias.data[:filter_index], conv_layer.bias.data[filter_index + 1:]))
                new_conv.bias.data = new_biases

        return new_conv

    def RestructureNextLayer(self, layer_index, filter_index):
        next_conv_layer = None
        for layer in self.model.features[layer_index + 1:]:
            if isinstance(layer, torch.nn.Conv2d):
                next_conv_layer = layer
                break

        next_new_conv = torch.nn.Conv2d(in_channels=next_conv_layer.in_channels - 1,
                                            out_channels=next_conv_layer.out_channels,
                                            kernel_size=next_conv_layer.kernel_size,
                                            stride=next_conv_layer.stride,
                                            padding=next_conv_layer.padding,
                                            dilation=next_conv_layer.dilation,
                                            groups=next_conv_layer.groups,
                                            bias=next_conv_layer.bias is not None)

        next_new_conv.weight.data = next_conv_layer.weight.data[:, :filter_index, :, :].clone()
        next_new_conv.weight.data = torch.cat([next_new_conv.weight.data, next_conv_layer.weight.data[:, filter_index + 1:, :, :]], dim=1)

        if next_conv_layer.bias is not None:
                next_new_conv.bias.data = next_conv_layer.bias.data.clone()

        return next_new_conv

    def PruneAndRestructure(self, filters_to_prune):
        print("===Prune and Restructre===\n")
        for layer_to_prune in filters_to_prune:
            next_layer_index = layer_to_prune + 1
            for i, layer in enumerate(self.model.features[layer_to_prune + 1:]):
                if isinstance(layer, torch.nn.Conv2d):
                    next_layer_index = layer_to_prune + i + 1
                    break
            for filter_to_prune in filters_to_prune[layer_to_prune][::-1]:
                if isinstance(self.model.features[layer_to_prune], torch.nn.Conv2d):
                    self.model.features[layer_to_prune] = self.PruneSingleFilter(layer_to_prune, filter_to_prune)
                if isinstance(self.model.features[next_layer_index], torch.nn.Conv2d):
                    self.model.features[next_layer_index] = self.RestructureNextLayer(layer_to_prune, filter_to_prune)

    def ModifyClassifier(self):
        print("===Modify classifier===\n")
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model.features(inputs)
            print(outputs.shape)
            break

        # Update the first linear layer in the classifier
        old_fc_layer = self.model.classifier[0]
        new_input_features = outputs.shape[1] * outputs.shape[2] * outputs.shape[3]

        # Create a new fully connected layer with the updated input features
        new_fc_layer = torch.nn.Linear(in_features=new_input_features, out_features=old_fc_layer.out_features)
        new_fc_layer = new_fc_layer.to(self.device)  # Move the new fully connected layer to device

        # Replace the old FC layer with the new one in the classifier
        self.model.classifier[0] = new_fc_layer
        # self.model.to(self.device)  # Ensure the entire model is on the correct device

    def PruneScalingFactors(self, filters_to_prune):
        print("===Prune Scaling Factors===\n")
        for layer_index in filters_to_prune:
            filter_indexes = filters_to_prune[layer_index]
            current_factors = self.scaling_factors[layer_index][0]
            
            # Mask for selected filters
            mask = torch.ones(current_factors.size(0), dtype = torch.bool)
            mask[filter_indexes] = False
            
            # Select & reshape remaining filters
            selected_filters = current_factors[mask]
            new_scaling_factor = selected_filters.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
            # for i, f in enumerate(self.scaling_factors[layer_index][0]) if i not in filter_indexes]

            # if selected_filters:
            #     new_scaling_factor = torch.cat(selected_filters)
            #     new_scaling_factor = new_scaling_factor.view(1, new_scaling_factor.shape[0], 1, 1)
            #     # Set requires_grad=True for the new scaling factor
            #     # new_scaling_factor.requires_grad_(True)
            #     new_scaling_factor = new_scaling_factor.detach().requires_grad_(True)
            #     self.scaling_factors[layer_index] = new_scaling_factor
            if new_scaling_factor.requires_grad:
                new_scaling_factor = new_scaling_factor.detach().requires_grad_(True)
                
            self.scaling_factors[layer_index] = new_scaling_factor

    def PruneImportanceScore(self, filters_to_prune):
        print("===Prune Importance Score===\n")
        for layer_index in filters_to_prune:
            filter_indexes = filters_to_prune[layer_index]
            selected_filters = [f.unsqueeze(0) for i, f in enumerate(self.importance_scores[layer_index][0]) if i not in filter_indexes]

            if selected_filters:
                new_importance_score = torch.cat(selected_filters)
                new_importance_score = new_importance_score.view(1, new_importance_score.shape[0], 1, 1)
                self.importance_scores[layer_index] = new_importance_score

    # def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
    #     best_accuracy = 0.0
    #     patience_counter = 0

    #     # Enable gradients for all model parameters for fine-tuning
    #     for param in self.model.parameters():
    #         param.requires_grad = True
            
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
    #     criterion = torch.nn.CrossEntropyLoss()

    #     class ImportanceMultiplier(torch.autograd.Function):
    #         @staticmethod
    #         def forward(ctx, input, importance_score):
    #             ctx.save_for_backward(importance_score)
    #             return input * importance_score

    #         @staticmethod
    #         def backward(ctx, grad_output):
    #             importance_score, = ctx.saved_tensors
    #             # Multiply the incoming gradient with importance score
    #             grad_input = grad_output * importance_score
    #             # Return None for importance_score gradient since it's fixed
    #             return grad_input, None

    #     multiply_importance = ImportanceMultiplier.apply

    #     for epoch in range(num_epochs):
    #         self.model.train()
    #         total_loss = 0
    #         correct = 0
    #         total = 0

    #         for batch_idx, (inputs, labels) in enumerate(self.train_loader):
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             optimizer.zero_grad()
                
    #             # Forward pass through features with importance scores
    #             x = inputs
    #             for idx, layer in enumerate(self.model.features):
    #                 x = layer(x)
    #                 if isinstance(layer, torch.nn.Conv2d) and idx in self.importance_scores:
    #                     importance_score = self.importance_scores[idx].to(self.device)
    #                     x = multiply_importance(x, importance_score)
                
    #             # Flatten and pass through classifier
    #             x = x.view(x.size(0), -1)
    #             outputs = self.model.classifier(x)
                
    #             loss = criterion(outputs, labels)
    #             total_loss += loss.item()
                
    #             # Backward pass
    #             loss.backward()
    #             optimizer.step()

    #             # Calculate accuracy
    #             _, predicted = outputs.max(1)
    #             total += labels.size(0)
    #             correct += predicted.eq(labels).sum().item()

    #             if (batch_idx + 1) % 100 == 0:
    #                 print(f'Epoch [{epoch + 1}/{num_epochs}], '
    #                     f'Step [{batch_idx + 1}/{len(self.train_loader)}], '
    #                     f'Loss: {loss.item():.4f}, '
    #                     f'Acc: {100. * correct / total:.2f}%')

    #         epoch_loss = total_loss / len(self.train_loader)
    #         epoch_acc = 100. * correct / total
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], '
    #             f'Loss: {epoch_loss:.4f}, '
    #             f'Acc: {epoch_acc:.2f}%')

    #     return self.model
    
    # def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
    #     best_accuracy = 0.0
    #     patience_counter = 0

    #     # Enable gradients for all model parameters for fine-tuning
    #     for param in self.model.parameters():
    #         param.requires_grad = True
                
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
    #     criterion = torch.nn.CrossEntropyLoss()

    #     class ImportanceMultiplier(torch.autograd.Function):
    #         @staticmethod
    #         def forward(ctx, input, importance_score):
    #             ctx.save_for_backward(input, importance_score)
    #             return input * importance_score

    #         @staticmethod
    #         def backward(ctx, grad_output):
    #             input, importance_score = ctx.saved_tensors
    #             # In backward pass:
    #             # 1. Multiply gradient with importance score (beta)
    #             # 2. Divide by the input value to maintain proper scaling
    #             grad_input = grad_output * importance_score
    #             # We don't need gradient for importance_score since it's fixed
    #             return grad_input, None

    #     multiply_importance = ImportanceMultiplier.apply

    #     for epoch in range(num_epochs):
    #         self.model.train()
    #         total_loss = 0
    #         correct = 0
    #         total = 0

    #         for batch_idx, (inputs, labels) in enumerate(self.train_loader):
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             optimizer.zero_grad()
                    
    #             # Forward pass through features with importance scores
    #             x = inputs
    #             layer_outputs = {}  # Store intermediate outputs for gradient computation
                
    #             for idx, layer in enumerate(self.model.features):
    #                 x = layer(x)
    #                 if isinstance(layer, torch.nn.Conv2d) and idx in self.importance_scores:
    #                     # Store the output before importance multiplication
    #                     layer_outputs[idx] = x.clone()
    #                     importance_score = self.importance_scores[idx].to(self.device)
    #                     x = multiply_importance(x, importance_score)
                    
    #             # Flatten and pass through classifier
    #             x = x.view(x.size(0), -1)
    #             outputs = self.model.classifier(x)
                    
    #             loss = criterion(outputs, labels)
    #             total_loss += loss.item()
                    
    #             # Backward pass
    #             loss.backward()
                
    #             # Scale gradients for layers with importance scores
    #             for idx, layer in enumerate(self.model.features):
    #                 if isinstance(layer, torch.nn.Conv2d) and idx in self.importance_scores:
    #                     if layer.weight.grad is not None:
    #                         # Apply importance score scaling to gradients
    #                         importance_score = self.importance_scores[idx].to(self.device)
    #                         layer.weight.grad = layer.weight.grad * importance_score.squeeze()
                
    #             optimizer.step()

    #             # Calculate accuracy
    #             _, predicted = outputs.max(1)
    #             total += labels.size(0)
    #             correct += predicted.eq(labels).sum().item()

    #             if (batch_idx + 1) % 100 == 0:
    #                 print(f'Epoch [{epoch + 1}/{num_epochs}], '
    #                     f'Step [{batch_idx + 1}/{len(self.train_loader)}], '
    #                     f'Loss: {loss.item():.4f}, '
    #                     f'Acc: {100. * correct / total:.2f}%')

    #         epoch_loss = total_loss / len(self.train_loader)
    #         epoch_acc = 100. * correct / total
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], '
    #             f'Loss: {epoch_loss:.4f}, '
    #             f'Acc: {epoch_acc:.2f}%')

    #     return self.model
    
    # def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
    #     best_accuracy = 0.0
    #     patience_counter = 0

    #     # Enable gradients for all model parameters for fine-tuning
    #     for param in self.model.parameters():
    #         param.requires_grad = True
                
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
    #     criterion = torch.nn.CrossEntropyLoss()

    #     class ImportanceMultiplier(torch.autograd.Function):
    #         @staticmethod
    #         def forward(ctx, input, importance_score):
    #             # Save both input and importance score for backward pass
    #             ctx.save_for_backward(input, importance_score)
    #             # Forward: multiply by beta
    #             return input * importance_score

    #         @staticmethod
    #         def backward(ctx, grad_output):
    #             input, importance_score = ctx.saved_tensors
    #             # In backward pass:
    #             # 1. Multiply gradient with importance score (beta)
    #             # 2. Then divide by beta to remove its effect from final gradients
    #             # This ensures beta only affects relative importance during training
    #             grad_input = grad_output * importance_score / (importance_score + 1e-8)  # Add small epsilon to prevent division by zero
    #             return grad_input, None

    #     multiply_importance = ImportanceMultiplier.apply

    #     for epoch in range(num_epochs):
    #         self.model.train()
    #         total_loss = 0
    #         correct = 0
    #         total = 0

    #         for batch_idx, (inputs, labels) in enumerate(self.train_loader):
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             optimizer.zero_grad()
                    
    #             # Forward pass through features with importance scores
    #             x = inputs
    #             layer_outputs = {}  # Store intermediate outputs
                
    #             for idx, layer in enumerate(self.model.features):
    #                 x = layer(x)
    #                 if isinstance(layer, torch.nn.Conv2d) and idx in self.importance_scores:
    #                     importance_score = self.importance_scores[idx].to(self.device)
    #                     # Apply importance score in forward pass
    #                     x = multiply_importance(x, importance_score)
                    
    #             # Flatten and pass through classifier
    #             x = x.view(x.size(0), -1)
    #             outputs = self.model.classifier(x)
                    
    #             loss = criterion(outputs, labels)
    #             total_loss += loss.item()
                    
    #             # Backward pass
    #             loss.backward()
                
    #             # Scale gradients for layers with importance scores
    #             for idx, layer in enumerate(self.model.features):
    #                 if isinstance(layer, torch.nn.Conv2d) and idx in self.importance_scores:
    #                     if layer.weight.grad is not None:
    #                         importance_score = self.importance_scores[idx].to(self.device)
    #                         # Apply importance score and then divide it out
    #                         layer.weight.grad = layer.weight.grad * importance_score.squeeze() / (importance_score.squeeze() + 1e-8)
                
    #             optimizer.step()

    #             # Calculate accuracy
    #             _, predicted = outputs.max(1)
    #             total += labels.size(0)
    #             correct += predicted.eq(labels).sum().item()

    #             if (batch_idx + 1) % 100 == 0:
    #                 print(f'Epoch [{epoch + 1}/{num_epochs}], '
    #                     f'Step [{batch_idx + 1}/{len(self.train_loader)}], '
    #                     f'Loss: {loss.item():.4f}, '
    #                     f'Acc: {100. * correct / total:.2f}%')

    #         epoch_loss = total_loss / len(self.train_loader)
    #         epoch_acc = 100. * correct / total
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], '
    #             f'Loss: {epoch_loss:.4f}, '
    #             f'Acc: {epoch_acc:.2f}%')

    #     return self.model
    
    def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
        best_accuracy = 0.0
        patience_counter = 0

        # Enable gradients for all model parameters for fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True
                
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()

        class ImportanceMultiplier(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, importance_score):
                # Save both input and importance score for backward pass
                ctx.save_for_backward(input, importance_score)
                # Forward: multiply by importance score with proper broadcasting
                # Reshape importance score to [1, channels, 1, 1] for proper broadcasting
                importance_score = importance_score.view(1, -1, 1, 1)
                return input * importance_score

            @staticmethod
            def backward(ctx, grad_output):
                input, importance_score = ctx.saved_tensors
                # Reshape importance score for broadcasting
                importance_score = importance_score.view(1, -1, 1, 1)
                # Apply importance score with proper broadcasting
                grad_input = grad_output * importance_score / (importance_score + 1e-8)
                return grad_input, None

        multiply_importance = ImportanceMultiplier.apply

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                    
                # Forward pass through features with importance scores
                x = inputs
                
                for idx, layer in enumerate(self.model.features):
                    x = layer(x)
                    if isinstance(layer, torch.nn.Conv2d) and idx in self.importance_scores:
                        importance_score = self.importance_scores[idx].to(self.device)
                        # Ensure importance score has the right number of channels
                        if importance_score.size(0) != x.size(1):
                            raise ValueError(f"Importance score size ({importance_score.size(0)}) must match "
                                        f"number of channels in feature map ({x.size(1)}) at layer {idx}")
                        # Apply importance score with proper broadcasting
                        x = multiply_importance(x, importance_score)
                    
                # Flatten and pass through classifier
                x = x.view(x.size(0), -1)
                outputs = self.model.classifier(x)
                    
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                    
                # Backward pass
                loss.backward()
                
                # Scale gradients for layers with importance scores
                for idx, layer in enumerate(self.model.features):
                    if isinstance(layer, torch.nn.Conv2d) and idx in self.importance_scores:
                        if layer.weight.grad is not None:
                            importance_score = self.importance_scores[idx].to(self.device)
                            # Reshape importance score for proper broadcasting with gradients
                            importance_score = importance_score.view(-1, 1, 1, 1)
                            layer.weight.grad = layer.weight.grad * importance_score / (importance_score + 1e-8)
                
                optimizer.step()

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], '
                        f'Step [{batch_idx + 1}/{len(self.train_loader)}], '
                        f'Loss: {loss.item():.4f}, '
                        f'Acc: {100. * correct / total:.2f}%')

            epoch_loss = total_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                f'Loss: {epoch_loss:.4f}, '
                f'Acc: {epoch_acc:.2f}%')

        return self.model
    
    
    def SaveState(self, path):
        """
        Save the pruner's state to a file.

        Args:
            path (str): The path to save the state to.
        """
        state = {
            'model': self.model,
            'scaling_factors': self.scaling_factors,
            'importance_scores': self.importance_scores,
            'pruned_filters': self.pruned_filters
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def LoadState(self, path):
        """
        Load the pruner's state from a file.

        Args:
            path (str): The path to load the state from.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.scaling_factors = state['scaling_factors']
        self.importance_scores = state['importance_scores']
        self.pruned_filters = state['pruned_filters']