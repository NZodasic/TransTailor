import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
import logging
import time

from Pruner import Pruner  # Import the Pruner class

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants for training
TA_EPOCH = 1
TA_LR = 0.005
TA_MOMENTUM = 0.9

IA_EPOCH = 1
IA_LR = 0.005
IA_MOMENTUM = 0.9

PRUNING_AMOUNT = 10

def LoadModel(device):
    # Load the VGG16 model
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

    # Replace the last layer of the model with a new layer that matches the number of classes in CIFAR10
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    model = model.to(device)

    return model


def LoadData(numWorker, batchSize, validation_split=0.1):
    """Load CIFAR10 dataset and split it into train, validation, and test sets."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_path = os.path.join(ROOT_DIR, "data")

    # Load full training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

    # Split into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(full_train_dataset)), test_size=validation_split, random_state=42
    )
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)

    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    kwargs = {"num_workers": numWorker, "pin_memory": True} if device.type == "cuda" else {}

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader

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


def CalculateAccuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy



def PrintDatasetInfo(train_loader, val_loader, test_loader):
    """Print information about the datasets."""
    def loader_info(loader, name):
        data_iter = iter(loader)
        inputs, labels = next(data_iter)
        print(f"Dataset: {name}")
        print(f"  Number of batches: {len(loader)}")
        # print(f"  Input batch shape: {inputs.shape}")
        # print(f"  Label batch shape: {labels.shape}")
        # print(f"  Sample labels: {labels[:10].tolist()}")
        print("-" * 50)

    print("=== Dataset Information ===")
    loader_info(train_loader, "Train")
    loader_info(val_loader, "Validation")
    loader_info(test_loader, "Test")
    print("===========================")


def PlotLossCurve(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def TimeLog():
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    print("Time log:", curr_time)


if __name__ == "__main__":
    # LOAD ARGUMENTS
    logger.info("START MAIN PROGRAM!")

    ROOT_DIR, CHECKPOINT_PATH, NUM_WORKER, BATCH_SIZE = LoadArguments()
    RESULT_PATH = os.path.join(ROOT_DIR, "optimal_model.pt")
    SAVED_PATH = os.path.join(ROOT_DIR, "checkpoint", "pruner", "checkpoint_{pruned_count}.pkl")

    # LOAD MODEL
    logger.info("GET DEVICE INFORMATION")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DEVICE: " + str(device))

    # LOAD DATASET
    logger.info("LOAD DATASET: CIFAR10")
    # train_loader, test_loader = LoadData(NUM_WORKER, BATCH_SIZE)
    # Gọi hàm này ngay sau khi LoadData
    train_loader, val_loader, test_loader = LoadData(NUM_WORKER, BATCH_SIZE)
    PrintDatasetInfo(train_loader, val_loader, test_loader)

    # LOAD MODEL
    logger.info("LOAD PRETRAINED MODEL: VGG-16 (ImageNet)")
    model = LoadModel(device)

    # # INIT PRUNING SCHEME
    pruner = Pruner(model, train_loader, device)
    if os.path.isfile(CHECKPOINT_PATH):
        logger.info("Load model and pruning info from checkpoint...")
        pruner.LoadState(CHECKPOINT_PATH)
    else:
        logger.info("Fine-tuning model before pruning")

        pruner.Finetune(3, TA_LR, TA_MOMENTUM, 0)
        pruner.InitScalingFactors()
        pruner.SaveState(SAVED_PATH.format(pruned_count = 0))

    opt_accuracy = CalculateAccuracy(pruner.model, test_loader)
    print(f"Accuracy of finetuned model: {opt_accuracy:.2f}%")
    logger.info(f"Accuracy of finetuned model: {opt_accuracy:.2f}%")
    logger.info("===DONE EVALUATE===")


    # Record training and validation losses
    train_losses = []
    val_losses = []


    # START PRUNING PROCESS
    while True:
        TimeLog()
        pruner.TrainScalingFactors(IA_EPOCH, IA_LR, IA_MOMENTUM)
        
        TimeLog()
        pruner.GenerateImportanceScores()
        
        TimeLog()
        filters_to_prune = pruner.FindFiltersToPrune(PRUNING_AMOUNT)
        
        TimeLog()
        pruner.PruneAndRestructure(filters_to_prune)
        
        TimeLog()
        pruner.ModifyClassifier()
        
        TimeLog()
        pruner.PruneScalingFactors(filters_to_prune)
        
        TimeLog()
        pruner.PruneImportanceScore(filters_to_prune)
        
        TimeLog()
        pruner.ImportanceAwareFineTuning(IA_EPOCH, IA_LR, IA_MOMENTUM)
        
        # Update and log losses
        train_losses.append(IA_LR)  # Dummy value, replace with actual loss if tracked
        val_losses.append(IA_LR)  # Dummy value, replace with actual loss if tracked

        sum_filters = 0 
        for layer in filters_to_prune:
            number_of_filters = len(filters_to_prune[layer])
            sum_filters += number_of_filters
        print(f"===Number of pruned filters is: ", sum_filters, flush=True)
        logger.info(f"===Number of pruned filters is: {sum_filters}")


        pruned_count = len(pruner.pruned_filters)
        if pruned_count % 5 == 0:
            pruner.SaveState(SAVED_PATH.format(pruned_count = pruned_count))
        
        TimeLog()
        pruner.Finetune(TA_EPOCH, TA_LR, TA_MOMENTUM, 0)
        
        TimeLog()
        pruned_accuracy = CalculateAccuracy(pruner.model, test_loader)
        
        print(f"Accuracy of pruned model: {pruned_accuracy:.2f}%")
        logger.info(f"Accuracy of pruned model: {pruned_accuracy:.2f}%")
        
        if abs(opt_accuracy - pruned_accuracy) > PRUNING_AMOUNT:
            print(f"Optimization done!", flush=True)
            torch.save(pruner.model.state_dict(), RESULT_PATH)
            break
        else:
            print(f"Update optimal model", flush=True)


    # Plot loss curve at the end of pruning
    PlotLossCurve(train_losses, val_losses)