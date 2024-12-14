import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import itertools
import pickle
import os
import gdown

##################################
# DEFINE CONSTANT
BATCH_SIZE = 64
NUM_WORKER = 20

FINETUNE_EPOCH = 1
FINETUNE_LR = 0.005
FINETUNE_MOMENTUM = 0.9

FINETUNE_PRUNING_EPOCH = 1

CHECKPOINT_URL = "https://drive.google.com/file/d/1VaQ2-ZVSt2kdMDHER34ghNCEnyJ-lNXu/view?usp=drive_link"
TARGET_AWARE_CHECKPOINT = "./checkpoint/Target_aware"
CHECKPOINT_NAME = "ta_epoch_1.pt"

ALPHA_CHECKPOINT = "ia_epoch_1.pt"
ALPHA_EPOCH = 1
ALPHA_LR = 0.005
ALPHA_MOMENTUM = 0.9

THRESHOLD = 5

ROOT_DIR = "~/TransTailor"

##################################

def LoadData(numWorker, batchSize):
    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the CIFAR10 train_dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    kwargs = {'num_workers': numWorker, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batchSize, shuffle=True, **kwargs)

    # Load test_dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    kwargs = {'num_workers': 12, 'pin_memory': True} if device == 'cuda' else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batchSize, shuffle=False, **kwargs)

    return train_loader, test_loader

def LoadModel(device):
    # Load the VGG16 model
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
   
    # Replace the last layer of the model with a new layer that matches the number of classes in CIFAR10
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(device)

    return model

def ModelFinetune(device, model, train_loader, num_epochs, learningRate, momentum, checkpointEpoch, mode):
    if mode == "Target_aware":
        checkpoint_path = 'checkpoint/Target_aware/ta_epoch_{epoch}.pt'
    elif mode == "Importance_aware":
        checkpoint_path = 'checkpoint/Importance_aware/ia_epoch_{epoch}.pt'

    print("\n===Fine-tune the pre-trained model to generate W_s*===")
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()

    epoch = checkpointEpoch

    for epoch in range(epoch, num_epochs):
        print("Epoch " + str(epoch+1) + "/" + str(num_epochs))
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path.format(epoch=epoch + 1))
    
    return model

def ScalingFactorsInit(model, alphaCheckpoint):
    if os.path.exists(alphaCheckpoint):
        print("Load alpha from checkpoint!")
        with open(alphaCheckpoint, 'rb') as handle:
            scaling_factors = pickle.load(handle)
    else:
        print("Init alpha from scratch!")
        num_layers = len(model.features)
        scaling_factors = {}

        for i in range(num_layers):
            layer = model.features[i]
            if isinstance(layer, torch.nn.Conv2d):
                print(layer,layer.out_channels)
                scaling_factors[i] = torch.rand((1,layer.out_channels,1,1), requires_grad=True)
    return scaling_factors

def ScalingFactorsTraining(model, scaling_factors, num_epochs, learning_rate, momentum):
    for param in model.parameters():
        param.requires_grad = False
   
    criterion = torch.nn.CrossEntropyLoss()
    num_layers = len(model.features)

    print("\n===Train the factors alpha by optimizing the loss function===")

    params_to_optimize = itertools.chain(scaling_factors[sf] for sf in scaling_factors.keys())
    optimizer_alpha = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        print("Epoch " + str(epoch+1) + "/" + str(num_epochs))
        iter_count = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.shape[0]
            optimizer_alpha.zero_grad()
            outputs = inputs
            outputs.requires_grad = False

            for i in range(num_layers):
                if isinstance(model.features[i], torch.nn.Conv2d):
                    outputs = model.features[i](outputs)
                    outputs = outputs*scaling_factors[i].cuda()
                else:
                    outputs = model.features[i](outputs)

            outputs = torch.flatten(outputs, 1)
            classification_output = model.classifier(outputs)
            loss = criterion(classification_output, labels)
            loss.backward()
            optimizer_alpha.step()
    
    return scaling_factors

def ImportanceScoreInit(scaling_factors, model, train_loader, device):
    importance_scores = {}
    num_layers = len(model.features)
    criterion = torch.nn.CrossEntropyLoss()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = inputs
        for i in range(num_layers):
            if isinstance(model.features[i], torch.nn.Conv2d):
                outputs = model.features[i](outputs)
                outputs = outputs*scaling_factors[i].cuda()
            else:
                outputs = model.features[i](outputs)

        outputs = torch.flatten(outputs, 1)
        classification_output = model.classifier(outputs)
        loss = criterion(classification_output, labels)
        
    for i, scaling_factor in scaling_factors.items():
        first_order_derivative = torch.autograd.grad(loss, scaling_factor, retain_graph=True)[0]
        importance_scores[i] = torch.abs(first_order_derivative * scaling_factor).detach() #Freeze importance_scores[i] after calculating

    return importance_scores

def FindFilterToPrune(importance_scores, pruned_filters):
    min_value = float('inf')
    min_filter = None
    min_layer = None

    for layer_index, scores_tensor in importance_scores.items():
        for filter_index, score in enumerate(scores_tensor[0]):
            # Check if the filter has already been pruned
            if (layer_index, filter_index) in pruned_filters:
                continue
            
            if score < min_value:
                min_value = score.item()
                min_filter = filter_index
                min_layer = layer_index
                if min_value == 0:
                    break

    return min_layer, min_filter

def CalculateAccuracy(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(labels)

    accuracy = 100 * total_correct / total_samples
    return accuracy

def CountNonZeroParameters(model):
    non_zero_params = sum(p.nonzero().size(0) for p in model.parameters())
    return non_zero_params

if __name__ == "__main__":
    print("GET DEVICE INFORMATION")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: " + str(device))

    print("LOAD DATASET: CIFAR10")
    train_loader, test_loader = LoadData(NUM_WORKER, BATCH_SIZE)

    print("LOAD PRETRAINED MODEL: VGG-16 (ImageNet)")
    model = LoadModel(device)

    # LOAD MODEL FROM SAVE
    load_saved_model = input("Do you want to download checkpoint? [y/n]: ")
    if load_saved_model == "y": 
        if CHECKPOINT_URL:
            gdown.download(url=CHECKPOINT_URL, output='./checkpoint/', fuzzy=True)

    checkpoint_epoch = 0
    load_saved_model = input("Do you want to load model from checkpoint? [y/n]: ")
    if load_saved_model == "y":
        checkpoint_path = os.path.join(TARGET_AWARE_CHECKPOINT, CHECKPOINT_NAME)
        if os.path.isfile(checkpoint_path):
            # LOAD MODEL FROM SAVE'S PATH
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Load model successfully!")

            # GET CHECKPOINT NUMBER FROM SAVE'S NAME
            checkpoint_epoch = checkpoint['epoch']
            print("Load checkpoint: ", checkpoint_epoch)
        else:
            print("Cannot find saved model, use original pretrained model instead!")

    # CHECK NUMBER OF PARAMETERS
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with open('log.txt', 'a') as f:
        print("Total trainable params: {:,}".format(total_param) , file=f)

    # FINE-TUNE MODEL ON TARGET DATASET
    opt_sub_model = ModelFinetune(device, model, train_loader, FINETUNE_EPOCH, FINETUNE_LR, FINETUNE_MOMENTUM, checkpoint_epoch, "Target_aware")
    pruned_model = opt_sub_model
    
    opt_accuracy = CalculateAccuracy(opt_sub_model, test_loader)
    with open('log.txt', 'a') as f:
        print("Accuracy of finetuned model: ", opt_accuracy, file=f)

    # INIT SCALING FACTOR
    scalingFactors = ScalingFactorsInit(model, ALPHA_CHECKPOINT)

    # MODEL OPTIMIZATION PROCESS
    while(1):
        scalingFactors = ScalingFactorsTraining(opt_sub_model, scalingFactors, ALPHA_EPOCH, ALPHA_LR, ALPHA_MOMENTUM)

        print("Generating importance score")
        importanceScores = ImportanceScoreInit(scalingFactors, model, train_loader, device)

        ### PRUNING PROCESS
        if 'pruned_filters' not in locals():
            pruned_filters = set()
        
        layer_to_prune, filter_to_prune = FindFilterToPrune(importanceScores, pruned_filters)

        with open('log.txt', 'a') as f:
            print("===Ready to prune ", filter_to_prune, "th filter in ", layer_to_prune, "th layer===", file=f)

        pruned_layer = pruned_model.features[layer_to_prune]
        pruned_filter = pruned_layer.weight.data[filter_to_prune]

        with torch.no_grad():
            pruned_layer.weight.data[filter_to_prune] = 0
            pruned_layer.bias.data[filter_to_prune] = 0

        # After pruning, you can update the pruned_filters set
        pruned_filters.add((layer_to_prune, filter_to_prune))

        pruned_params = total_param - CountNonZeroParameters(pruned_model)
        with open('log.txt', 'a') as f:
            print("Pruned params: {:,}".format(pruned_params), file=f)
            print("Pruned ratio: ", 100*pruned_params/total_param, "%", file=f)

        ### END OF PRUNING PROCESS

        # FINE-TUNE AFTER PRUNING
        for param in pruned_model.parameters():
            param.requires_grad = True
        ModelFinetune(device, pruned_model, train_loader, FINETUNE_PRUNING_EPOCH, FINETUNE_LR, FINETUNE_MOMENTUM, 0, "Target_aware")

        # CALCULATE ACCURACY
        pruned_accuracy = CalculateAccuracy(pruned_model, test_loader)
        with open('log.txt', 'a') as f:
            print("Accuracy of pruned model: ", pruned_accuracy, file=f)

        if abs(opt_accuracy - pruned_accuracy) > THRESHOLD:
            print("Optimization done!")
            torch.save(opt_sub_model.state_dict(), 'checkpoint/optimal_model.pt')
            break
        else:
            print("Update optimal model")
            opt_sub_model = pruned_model
            opt_accuracy = pruned_accuracy