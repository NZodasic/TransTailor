import torch
import torchvision
import torchvision.transforms as transforms
from Pruner import Pruner
import argparse
import os
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TA_EPOCH = 10
TA_LR = 0.001
TA_MOMENTUM = 0.9

IA_EPOCH = 1
IA_LR = 0.005
IA_MOMENTUM = 0.9

def LoadModel(device):
    # Load the VGG16 model
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
   
    # Freeze early layers
    for param in model.features[:10]:
        param.requires_grad = False

    # Replace the last layer of the model with a new layer that matches the number of classes in CIFAR10
    num_classes = 10
    model.classifier[6] = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[6].in_features, 512),
        torch.nn.ReLU(True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, num_classes)
    )
    model = model.to(device)

    return model

def LoadData(numWorker, batchSize):
    # Define the data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  
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

    kwargs = {'num_workers': 4, 'pin_memory': True} if device == 'cuda' else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batchSize, shuffle=False, **kwargs)

    return train_loader, test_loader

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

def TimeLog():
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    print("Time log:", curr_time)


if __name__ == "__main__":
    # LOAD ARGUMENTS
    logger.info("START MAIN PROGRAM!")
    ROOT_DIR, CHECKPOINT_PATH, NUM_WORKER, BATCH_SIZE = LoadArguments()
    RESULT_PATH = os.path.join(ROOT_DIR, "checkpoint/optimal_model.pt")
    SAVED_PATH = os.path.join(ROOT_DIR, "checkpoint/pruner/checkpoint_{pruned_count}.pkl")

    # LOAD MODEL
    logger.info("GET DEVICE INFORMATION")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DEVICE: " + str(device))

    logger.info("LOAD DATASET: CIFAR10")
    train_loader, test_loader = LoadData(NUM_WORKER, BATCH_SIZE)

    logger.info("LOAD PRETRAINED MODEL: VGG-16 (ImageNet)")
    model = LoadModel(device)

    # INIT PRUNING SCHEME
    pruner = Pruner(model, train_loader, device, amount=10)
    if os.path.isfile(CHECKPOINT_PATH):
        logger.info("Load model and pruning info from checkpoint...")
        pruner.LoadState(CHECKPOINT_PATH)
    else:
        pruner.Finetune(40, TA_LR, TA_MOMENTUM, 0)

        pruner.InitScalingFactors()
        pruner.SaveState(SAVED_PATH.format(pruned_count = 0))

    opt_accuracy = CalculateAccuracy(pruner.model, test_loader)
    print("Accuracy of finetuned model: ", opt_accuracy, flush=True)

    patience = 5
    no_improve_count = 0
    best_accuracy = opt_accuracy
    
    # START PRUNING PROCESS
    while True:
        pruner.TrainScalingFactors(ROOT_DIR, 1, IA_LR, IA_MOMENTUM)
        pruner.GenerateImportanceScores()
        filters_to_prune = pruner.FindFiltersToPrune(prune_percentage=5)

        filters_to_prune = pruner.FindFiltersToPrune(
            model_complexity_factor=len(list(model.features))/10
        )
        pruner.PruneAndRestructure(filters_to_prune)

        pruner.ModifyClassifier()

        pruner.PruneScalingFactors(filters_to_prune)

        pruner.PruneImportanceScore(filters_to_prune)
        
        sum_filters = 0 
        for layer in filters_to_prune:
            number_of_filters = len(filters_to_prune[layer])
            sum_filters += number_of_filters
        print("===Number of pruned filters is: ", sum_filters)

        # pruned_count = len(pruner.pruned_filters)
        # if pruned_count % 10 == 0:
        #     pruner.SaveState(SAVED_PATH.format(pruned_count = pruned_count))

        for param in pruner.model.parameters():
            param.requires_grad = True

        pruner.Finetune(TA_EPOCH, TA_LR, TA_MOMENTUM, 0)

        pruned_accuracy = CalculateAccuracy(pruner.model, test_loader)
        print("Accuracy of pruned model: ", pruned_accuracy, flush=True)
        
        if pruned_accuracy > best_accuracy:
            best_accuracy = pruned_accuracy
            no_improve_count = 0
            torch.save(pruner.model.state.dict(), RESULT_PATH)
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print("Early stopping triggered", flush=True)
            break
        

        if abs(opt_accuracy - pruned_accuracy) > pruner.amount:
            print("Optimization done!", flush=True)
        #     torch.save(pruner.model.state_dict(), RESULT_PATH)
        #     break
        # else:
        #     print("Update optimal model", flush=True)