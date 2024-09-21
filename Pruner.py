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
        self.scaling_factors = {}

        for i, layer in enumerate(self.model.modules()):
            if isinstance(layer, torch.nn.Conv2d):
                print(f"Layer {i}: {layer}, out_channels: {layer.out_channels}")
                self.scaling_factors[i] = torch.rand((1, layer.out_channels, 1, 1), requires_grad=True, device=self.device)

    def TrainScalingFactors(self, root, num_epochs, learning_rate, momentum):
        checkpoint_path = os.path.join(root, 'checkpoint/Importance_aware/ia_epoch_{epoch}.pt')

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
        self.importance_scores = {}
        criterion = torch.nn.CrossEntropyLoss()
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss = self._forward_with_scaling_factors(inputs, labels, criterion)
            
            for i, scaling_factor in self.scaling_factors.items():
                grad = torch.autograd.grad(loss, scaling_factor, retain_graph=True)[0]
                self.importance_scores[i] = (grad * scaling_factor).detach()

    def _forward_with_scaling_factors(self, inputs, labels, criterion):
        outputs = inputs
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, torch.nn.Conv2d):
                outputs = layer(outputs)
                if i in self.scaling_factors:
                    outputs = outputs * self.scaling_factors[i]
            else:
                outputs = layer(outputs)
        
        # Flatten the output before passing to classifier
        outputs = torch.flatten(outputs, 1)
        outputs = self.model.classifier(outputs)
        
        return criterion(outputs, labels)
                    
                    
        #     for i in range(num_layers):
        #         if isinstance(self.model.features[i], torch.nn.Conv2d):
        #             outputs = self.model.features[i](outputs)
        #             outputs = outputs * self.scaling_factors[i].cuda()
        #         else:
        #             outputs = self.model.features[i](outputs)

        #     outputs = torch.flatten(outputs, 1)
        #     classification_output = self.model.classifier(outputs)
        #     loss = criterion(classification_output, labels)

        # for i, scaling_factor in self.scaling_factors.items():
        #     first_order_derivative = torch.autograd.grad(loss, scaling_factor, retain_graph=True)[0]
        #     self.importance_scores[i] = torch.abs(first_order_derivative * scaling_factor).detach()

    def FindFilterToPrune(self, threshold):
        for layer_index, scores_tensor in self.importance_scores.items():
            for filter_index, score in enumerate(scores_tensor[0]):
                if (layer_index, filter_index) not in self.pruned_filters and score < threshold:
                    return layer_index, filter_index
        return None, None
        # min_value = float('inf')
        # min_filter = None
        # min_layer = None

        # for layer_index, scores_tensor in self.importance_scores.items():
        #     for filter_index, score in enumerate(scores_tensor[0]):
        #         # Check if the filter has already been pruned
        #         if (layer_index, filter_index) in self.pruned_filters:
        #             continue

        #         if score < min_value:
        #             min_value = score.item()
        #             min_filter = filter_index
        #             min_layer = layer_index
        #             if min_value == 0:
        #                 break

        # return min_layer, min_filter

    def Prune(self, layer_to_prune, filter_to_prune):
        pruned_layer = self.model.features[layer_to_prune]

        with torch.no_grad():
            pruned_layer.weight.data[filter_to_prune] = 0
            pruned_layer.bias.data[filter_to_prune] = 0

        # After pruning, you can update the pruned_filters set
        self.pruned_filters.add((layer_to_prune, filter_to_prune))

    def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
        logger.info("\n===Fine-tune the model to achieve W_s*===")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self._forward_with_scaling_factors(inputs, labels, criterion)
                loss.backward()
                optimizer.step()

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