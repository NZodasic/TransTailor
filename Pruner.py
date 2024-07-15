import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import itertools
import pickle
import os
import gdown
import argparse

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
        print("Init alpha from scratch!")
        num_layers = len(self.model.features)
        self.scaling_factors = {}

        for i in range(num_layers):
            layer = self.model.features[i]
            if isinstance(layer, torch.nn.Conv2d):
                print(layer, layer.out_channels)
                self.scaling_factors[i] = torch.rand((1, layer.out_channels, 1, 1), requires_grad=True)

    def TrainScalingFactors(self, root, num_epochs, learning_rate, momentum):
        checkpoint_path = os.path.join(root, 'checkpoint/Importance_aware/ia_epoch_{epoch}.pt')

        for param in self.model.parameters():
            param.requires_grad = False

        criterion = torch.nn.CrossEntropyLoss()
        num_layers = len(self.model.features)

        print("\n===Train the factors alpha by optimizing the loss function===")

        params_to_optimize = itertools.chain(self.scaling_factors[sf] for sf in self.scaling_factors.keys())
        optimizer_alpha = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum)

        for epoch in range(num_epochs):
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            iter_count = 0

            for inputs, labels in tqdm(self.train_loader):
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

    def FindFilterToPrune(self):
        min_value = float('inf')
        min_filter = None
        min_layer = None

        for layer_index, scores_tensor in self.importance_scores.items():
            for filter_index, score in enumerate(scores_tensor[0]):
                # Check if the filter has already been pruned
                if (layer_index, filter_index) in self.pruned_filters:
                    continue

                if score < min_value:
                    min_value = score.item()
                    min_filter = filter_index
                    min_layer = layer_index
                    if min_value == 0:
                        break

        return min_layer, min_filter

    def Prune(self, layer_to_prune, filter_to_prune):
        pruned_layer = self.model.features[layer_to_prune]

        with torch.no_grad():
            pruned_layer.weight.data[filter_to_prune] = 0
            pruned_layer.bias.data[filter_to_prune] = 0

        # After pruning, you can update the pruned_filters set
        self.pruned_filters.add((layer_to_prune, filter_to_prune))

    def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
        print("\n===Fine-tune the model to achieve W_s*===")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()

        epoch = checkpoint_epoch

        for epoch in range(epoch, num_epochs):
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
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