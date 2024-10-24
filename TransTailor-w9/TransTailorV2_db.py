import torch
import torchvision
import torchvision.transforms as transforms
from Pruner import Pruner
import argparse
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TA_EPOCH = 1
TA_LR = 0.005
TA_MOMENTUM = 0.9

IA_EPOCH = 1
IA_LR = 0.005
IA_MOMENTUM = 0.9
PRUNING_ROUNDS = 3

def LoadModel(device):
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    num_classes = 10
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)
    return model

def LoadData(numWorker, batchSize, subset_size):

    transform = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_path = os.path.join(ROOT_DIR, "data")
    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    
    if subset_size is not None:
      subset_indices = torch.randperm(len(train_dataset))[:subset_size]
      train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
      
    kwargs = {'num_workers': numWorker, 'pin_memory': True} if device == 'cuda' else {}
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, **kwargs)
    
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False, **kwargs)
    
    return train_dataset, train_loader, test_loader

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

def CalculateAccuracy(model, test_loader, device):
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


if __name__ == "__main__":
    # LOAD ARGUMENTS
    logger.info("START MAIN PROGRAM!")
    ROOT_DIR, CHECKPOINT_PATH, NUM_WORKER, BATCH_SIZE = LoadArguments()
    
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoint")
    RESULT_PATH = os.path.join(ROOT_DIR, "checkpoint/optimal_model.pt")
    SAVED_PATH = os.path.join(ROOT_DIR, "checkpoint/pruner/checkpoint_{pruned_count}.pkl")

    # LOAD DEVICE
    logger.info("GET DEVICE INFORMATION")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DEVICE: " + str(device))

    # LOAD DATASETS
    logger.info("LOAD DATASET: CIFAR10")
    train_loader, test_loader = LoadData(NUM_WORKER, BATCH_SIZE)
    train_dataset, train_loader, test_loader = LoadData(NUM_WORKER, BATCH_SIZE, subset_size=1000)
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of batches in train_loader: {len(train_loader)}")
    logger.info(f"Number of batches in test_loader: {len(test_loader)}")
    logger.info(f"Batch size: {train_loader.batch_size}")

    # LOAD MODEL
    logger.info("LOAD PRETRAINED MODEL: VGG-16 (ImageNet)")
    model = LoadModel(device)
    
    model, scaling_factors, importance_scores, pruned_filters = LoadState("/content/drive/MyDrive/BCU-documents/checkpoint_0.pkl")

    initial_accuracy = CalculateAccuracy(model, test_loader, device)
    logger.info("Accuracy of finetuned model: ", initial_accuracy, flush=True)


    pruner = Pruner(model, train_loader, device, amount=0.1, prune_batch_size=10)

    # Load checkpoint if available
    if os.path.isfile(CHECKPOINT_PATH):
        logger.info("Load model and pruning info from checkpoint...")
        pruner.LoadState(CHECKPOINT_PATH)
    else:
        pruner.Finetune(40, TA_LR, TA_MOMENTUM, 0)
        pruner.InitScalingFactors()
        pruner.SaveState(SAVED_PATH.format(pruned_count=0))


    opt_accuracy = CalculateAccuracy(pruner.model, test_loader)
    print("Accuracy of finetuned model: ", opt_accuracy, flush=True)

    for round in range(PRUNING_ROUNDS):
        logger.info(f"Starting pruning round {round + 1}/{PRUNING_ROUNDS}")
        
        # Train scaling factors
        pruner.TrainScalingFactors(model, scaling_factors, train_loader, IA_EPOCH, IA_LR, IA_MOMENTUM)
        
        # Generate importance scores
        pruner.GenerateImportanceScores(model, train_loader, importance_scores, scaling_factors)
        
        # Find and prune filters
        filters_to_prune = pruner.FindFiltersToPrune(importance_scores, pruned_filters)
        
        pruned_model_ver1 = pruner.PruneAndRestructure(model, filters_to_prune)
        new_scaling_factors_1 = pruner.PruneScalingFactors(filters_to_prune, scaling_factors)
        new_importance_scores_1 = pruner.PruneImportanceScore(filters_to_prune, importance_scores)



        
        pruner.ModifyClassifier(pruned_model_ver1, train_loader, device)
        
        pruner.Finetune(pruned_model_ver1, train_loader, device, TA_EPOCH, TA_LR, TA_MOMENTUM, 0)
        
        pruner.ImportanceAwareFineTuning(pruned_model_ver1, train_loader, device, new_importance_scores_1, IA_EPOCH, IA_LR, IA_MOMENTUM)
        pruned_accuracy = CalculateAccuracy(pruned_model_ver1, test_loader, device)
        
        logger.info(f"Round {round + 1} accuracy: {pruned_accuracy:.2f}%", flush=True)