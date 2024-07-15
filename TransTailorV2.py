import torch
import torchvision
import torchvision.transforms as transforms
from Pruner import Pruner
import argparse
import os

TA_EPOCH = 1
TA_LR = 0.005
TA_MOMENTUM = 0.9

IA_EPOCH = 1
IA_LR = 0.005
IA_MOMENTUM = 0.9

def LoadModel(device):
    # Load the VGG16 model
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
   
    # Replace the last layer of the model with a new layer that matches the number of classes in CIFAR10
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(device)

    return model

def LoadData(numWorker, batchSize):
    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_path = os.path.join(ROOT_DIR, "data")

    # Load the CIFAR10 train_dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

    kwargs = {'num_workers': numWorker, 'pin_memory': True} if device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batchSize, shuffle=True, **kwargs)

    # Load test_dataset
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    kwargs = {'num_workers': 12, 'pin_memory': True} if device == 'cuda' else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batchSize, shuffle=False, **kwargs)

    return train_loader, test_loader

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

def LoadArguments():
    parser = argparse.ArgumentParser(description="Config cli params")
    parser.add_argument("-r","--root", help="Root directory")
    parser.add_argument("-c","--checkpoint", default="", help="Checkpoint path")
    parser.add_argument("-n", "--numworker", default=1, help="Number of worker")
    parser.add_argument("-b", "--batchsize", default=32, help="Batch size")

    args = parser.parse_args()
    ROOT_DIR = args.root
    CHECKPOINT_PATH = args.checkpoint
    NUM_WORKER = int(args.numworker)
    BATCH_SIZE = int(args.batchsize)

    return ROOT_DIR, CHECKPOINT_PATH, NUM_WORKER, BATCH_SIZE

if __name__ == "__main__":
    # LOAD ARGUMENTS
    ROOT_DIR, CHECKPOINT_PATH, NUM_WORKER, BATCH_SIZE = LoadArguments()
    RESULT_PATH = os.path.join(ROOT_DIR, "checkpoint/optimal_model.pt")
    SAVED_PATH = os.path.join(ROOT_DIR, "checkpoint/pruner/checkpoint_{pruned_count}.pkl")

    # LOAD MODEL
    print("GET DEVICE INFORMATION")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: " + str(device))

    print("LOAD DATASET: CIFAR10")
    train_loader, test_loader = LoadData(NUM_WORKER, BATCH_SIZE)

    print("LOAD PRETRAINED MODEL: VGG-16 (ImageNet)")
    model = LoadModel(device)

    # INIT PRUNING SCHEME
    pruner = Pruner(model, train_loader, device, amount=10)
    if os.path.isfile(CHECKPOINT_PATH):
        print("Load model and pruning info from checkpoint...")
        pruner.LoadState(CHECKPOINT_PATH)
    else:
        pruner.Finetune(40, TA_LR, TA_MOMENTUM, 0)

        pruner.InitScalingFactors()
        pruner.SaveState(SAVED_PATH.format(pruned_count = 0))

    opt_accuracy = CalculateAccuracy(pruner.model, test_loader)
    print("Accuracy of finetuned model: ", opt_accuracy)

    # START PRUNING PROCESS
    while True:
        pruner.TrainScalingFactors(ROOT_DIR, 1, IA_LR, IA_MOMENTUM)
        pruner.GenerateImportanceScores()
        layer_to_prune, filter_to_prune = pruner.FindFilterToPrune()
        pruner.Prune(layer_to_prune, filter_to_prune)
        print("===Prune ", filter_to_prune, "th filter in ", layer_to_prune, "th layer===")

        pruned_count = pruner.pruned_filters
        if len(pruned_count) % 10 == 0:
            pruner.SaveState(SAVED_PATH.format(pruned_count = pruned_count))

        for param in pruner.model.parameters():
            param.requires_grad = True
        pruner.Finetune(TA_EPOCH, TA_LR, TA_MOMENTUM, 0)

        pruned_accuracy = CalculateAccuracy(pruner.model, test_loader)
        print("Accuracy of pruned model: ", pruned_accuracy)

        if abs(opt_accuracy - pruned_accuracy) > pruner.amount:
            print("Optimization done!")
            torch.save(pruner.model.state_dict(), RESULT_PATH)
            break
        else:
            print("Update optimal model")

