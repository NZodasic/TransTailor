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
    def __init__(self, model, train_loader, device, amount=10):
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
        
        for i in range(num_layers):
            layer = self.model.features[i]
            
            if isinstance(layer, torch.nn.Conv2d):
                self.scaling_factors[i] = torch.rand((1, layer.out_channels, 1, 1), requires_grad=True)

    def TrainScalingFactors(self, num_epochs, learning_rate, momentum):
            for param in self.model.parameters():
                if param.requires_grad != False:
                    param.requires_grad = False

            criterion = torch.nn.CrossEntropyLoss()
            num_layers = len(self.model.features)

            logger.info("\n===Train the factors alpha by optimizing the loss function===")

            params_to_optimize = []
            for sf in self.scaling_factors.keys():
                if isinstance(self.scaling_factors[sf], list):
                    for param in self.scaling_factors[sf]:
                        if param.requires_grad:
                            params_to_optimize.append(param)
                        else:
                            params_to_optimize.append(param.clone().detach().requires_grad_(True))
                else:
                    params_to_optimize.append(self.scaling_factors[sf].clone().detach().requires_grad_(True))
            optimizer_alpha = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum)

            for epoch in range(num_epochs):
                logger.info("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
                print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
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
            logger.info("Generating importance scores")
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
                first_order_derivative = torch.autograd.grad(loss, scaling_factor, retain_graph=True, allow_unused=True)[0]
                self.importance_scores[i] = torch.abs(first_order_derivative * scaling_factor).detach()

    def ModifyClassifier(self):
        for inputs, labels in self.train_loader:
            self.model.to(self.device)
            inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move inputs and labels to device
            outputs = self.model.features(inputs)
            print(outputs.shape)
            break

        old_fc_layer = self.model.classifier[0]
        new_input_features = outputs.shape[1] * outputs.shape[2] * outputs.shape[3]

        new_fc_layer = torch.nn.Linear(in_features=new_input_features, out_features=old_fc_layer.out_features)
        new_fc_layer = new_fc_layer.to(self.device)

        self.model.classifier[0] = new_fc_layer
        self.model.to(self.device)

    def PruneImportanceScore(self, filters_to_prune):
            for layer_index in filters_to_prune:
                filter_indexes = filters_to_prune[layer_index]
                selected_filters = [f.unsqueeze(0) for i, f in enumerate(self.importance_scores[layer_index][0]) if i not in filter_indexes]

                if selected_filters:
                    new_importance_score = torch.cat(selected_filters)
                    new_importance_score = new_importance_score.view(1, new_importance_score.shape[0], 1, 1)
                    self.importance_scores[layer_index] = new_importance_score

            return self.importance_scores
        
    def PruneScalingFactors(self, filters_to_prune):
        for layer_index in filters_to_prune:
            filter_indexes = filters_to_prune[layer_index]
            selected_filters = [f.unsqueeze(0) for i, f in enumerate(self.scaling_factors[layer_index][0]) if i not in filter_indexes]

            if selected_filters:
                new_scaling_factor = torch.cat(selected_filters)
                new_scaling_factor = new_scaling_factor.view(1, new_scaling_factor.shape[0], 1, 1)
                # Set requires_grad=True for the new scaling factor
                new_scaling_factor.requires_grad_(True)
                self.scaling_factors[layer_index] = new_scaling_factor

        return self.scaling_factors

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
        for layer_to_prune in filters_to_prune:
            next_layer_index = layer_to_prune + 1
            for i, layer in enumerate(self.model.features[layer_to_prune + 1:]):
                if isinstance(layer, torch.nn.Conv2d):
                    next_layer_index = layer_to_prune + i + 1
                    break
            for filter_to_prune in filters_to_prune[layer_to_prune][::-1]:
                if isinstance(self.model.features[layer_to_prune], torch.nn.Conv2d):
                    self.model.features[layer_to_prune] = self.PruneSingleFilter(self.model, layer_to_prune, filter_to_prune)
                if isinstance(self.model.features[next_layer_index], torch.nn.Conv2d):
                    self.model.features[next_layer_index] = self.RestructureNextLayer(self.model, layer_to_prune, filter_to_prune)
        return self.model
    
    def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
        logger.info("\n===Fine-tune the model to achieve W_s*===")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)
        
        epoch = checkpoint_epoch

        for epoch in range(epoch, num_epochs):
            logger.info("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
    def ImportanceAwareFineTuning(self, num_epochs, learning_rate, momentum):

        logger.info("\n===Importance-aware fine-tuning===")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                output = inputs
                for idx, layer in enumerate(self.model.features):
                    output = layer(output)
                    
                    if isinstance(layer, torch.nn.Conv2d):
                        with torch.no_grad():
                            importance_score = self.importance_scores[idx].to(self.device)
                            output = output * importance_score
                            
                output = output.view(output.size(0), -1)
                
                output = self.model.classifier(output)

                loss = criterion(output, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}')

        return self.model

    def SaveState(self, path):
            state = {
                'model': self.model,
                'scaling_factors': self.scaling_factors,
                'importance_scores': self.importance_scores,
                'pruned_filters': self.pruned_filters
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)

    def LoadState(path):
            with open(path, 'rb') as f:
                state = pickle.load(f)

            model = state['model']
            scaling_factors = state['scaling_factors']
            importance_scores = state['importance_scores']
            pruned_filters = state['pruned_filters']

            return model, scaling_factors, importance_scores, pruned_filters