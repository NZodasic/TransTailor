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
                self.scaling_factors[i] = torch.rand((1, layer.out_channels, 1, 1), requires_grad=True) * 0.1

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

    def GenerateImportanceScores(self, num_passes=3):
        print("===Generate importance score===")
        importance_accumulator = {}
        num_layers = len(self.model.features)
        criterion = torch.nn.CrossEntropyLoss()

        for pass_num in range(num_passes):
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
                    importance = torch.abs(first_order_derivative * scaling_factor).detach()
                    if i not in importance_accumulator:
                        importance_accumulator[i] = importance
                    else:
                        importance_accumulator[i] += importance

        # Average the accumulated scores
        self.importance_scores = {k: v/num_passes for k, v in importance_accumulator.items()}

    def FindFiltersToPrune(self, model_complexity_factor=None):
        # Calculate adaptive pruning percentage based on model complexity
        if model_complexity_factor is None:
            model_complexity_factor = len(list(self.model.features)) / 10
        prune_percentage = max(1, min(5, 3 * model_complexity_factor))

        all_scores = []
        filter_mapping = []  # To keep track of (layer_idx, filter_idx) for each score
        
        for layer_idx, scores_tensor in self.importance_scores.items():
            for filter_idx, score in enumerate(scores_tensor[0]):
                all_scores.append(score.item())
                filter_mapping.append((layer_idx, filter_idx))
        
        # Calculate the threshold for bottom k%
        num_filters = len(all_scores)
        num_to_prune = int(num_filters * (prune_percentage / 100))
        
        if num_to_prune == 0:
            print(f"Warning: {prune_percentage}% of {num_filters} filters is less than 1. Defaulting to 1 filter.")
            num_to_prune = 1
        
        # Get indices of bottom k% scores
        sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])
        bottom_k_indices = sorted_indices[:num_to_prune]
        
        # Group filters by layer
        filters_to_prune = {}
        for idx in bottom_k_indices:
            layer_idx, filter_idx = filter_mapping[idx]
            if layer_idx not in filters_to_prune:
                filters_to_prune[layer_idx] = []
            filters_to_prune[layer_idx].append(filter_idx)
        
        # Sort filter indices within each layer for consistent pruning
        for layer_idx in filters_to_prune:
            filters_to_prune[layer_idx].sort()
        
        print(f"Total filters: {num_filters}")
        print(f"Pruning {num_to_prune} filters ({prune_percentage}%)")
        for layer_idx in filters_to_prune:
            print(f"Layer {layer_idx}: pruning {len(filters_to_prune[layer_idx])} filters")
            
        return filters_to_prune

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
                
        if next_conv_layer is None:
            return None

        # Validate input channel compatibility 
        if next_conv_layer.in_channels <= 1:
            raise ValueError("Cannot further reduce channels")

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
            i = 1

            while next_layer_index < len(self.model.features) and not isinstance(self.model.features[next_layer_index], torch.nn.Conv2d):
                next_layer_index += 1
                i += 1

            for filter_to_prune in filters_to_prune[layer_to_prune][::-1]:
                if isinstance(self.model.features[layer_to_prune], torch.nn.Conv2d):
                    self.model.features[layer_to_prune] = self.PruneSingleFilter(layer_to_prune, filter_to_prune)
                if next_layer_index < len(self.model.features) and isinstance(self.model.features[next_layer_index], torch.nn.Conv2d):
                    self.model.features[next_layer_index] = self.RestructureNextLayer(layer_to_prune, filter_to_prune)   
                     
            # for i, layer in enumerate(self.model.features[layer_to_prune + 1:]):
            #     if isinstance(layer, torch.nn.Conv2d):
            #         next_layer_index = layer_to_prune + i + 1
            #         break
            # for filter_to_prune in filters_to_prune[layer_to_prune][::-1]:
            #     if isinstance(self.model.features[layer_to_prune], torch.nn.Conv2d):
            #         self.model.features[layer_to_prune] = self.PruneSingleFilter(layer_to_prune, filter_to_prune)
            #     if isinstance(self.model.features[next_layer_index], torch.nn.Conv2d):
            #         self.model.features[next_layer_index] = self.RestructureNextLayer(layer_to_prune, filter_to_prune)

                    

    def ModifyClassifier(self):
        print("===Modify classifier===\n")
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model.features(inputs)
            print(outputs.shape)
            break

        for i, module in enumerate(self.model.classifier):
            if isinstance(module, torch.nn.Linear):
                old_fc_layer = module
                new_input_features = outputs.shape[1] * outputs.shape[2] * outputs.shape[3]
                new_fc_layer = torch.nn.Linear(in_features=new_input_features, out_features=old_fc_layer.out_features)
                new_fc_layer = new_fc_layer.to(self.device)
                self.model.classifier[i] = new_fc_layer
                break
            
        # # Update the first linear layer in the classifier
        # old_fc_layer = self.model.classifier[0]
        # new_input_features = outputs.shape[1] * outputs.shape[2] * outputs.shape[3]

        # # Create a new fully connected layer with the updated input features
        # new_fc_layer = torch.nn.Linear(in_features=new_input_features, out_features=old_fc_layer.out_features)
        # new_fc_layer = new_fc_layer.to(self.device)  # Move the new fully connected layer to device

        # # Replace the old FC layer with the new one in the classifier
        # self.model.classifier[0] = new_fc_layer
        # # self.model.to(self.device)  # Ensure the entire model is on the correct device

    def PruneScalingFactors(self, filters_to_prune):
        print("===Prune Scaling Factors===\n")
        for layer_index in filters_to_prune:
            filter_indexes = filters_to_prune[layer_index]
            selected_filters = [f.unsqueeze(0) for i, f in enumerate(self.scaling_factors[layer_index][0]) if i not in filter_indexes]

            if selected_filters:
                new_scaling_factor = torch.cat(selected_filters)
                new_scaling_factor = new_scaling_factor.view(1, new_scaling_factor.shape[0], 1, 1)
                # Set requires_grad=True for the new scaling factor
                # new_scaling_factor.requires_grad_(True)
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

    def Finetune(self, num_epochs, initial_lr, momentum, checkpoint_epoch):
        logger.info("\n===Fine-tune the model to achieve W_s*===")
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=initial_lr,
                                    momentum=momentum,
                                    weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= num_epochs)
        criterion = torch.nn.CrossEntropyLoss()
        best_accuracy = 0

        for epoch in range(checkpoint_epoch, num_epochs):
            logger.info("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            self.model.train()
            total_loss = 0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            logger.info(f"Epoch {epoch+1} Validation Accuracy: {accuracy}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')
                
        return best_accuracy


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