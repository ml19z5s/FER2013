import os
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from utils.dataset_loader import load_datasets
from utils.my_logging import get_logger
from utils.config import parse_arguments

# Import the factory function
from models import create_model


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def main(args, test_only=False):
    device = args.device
    if args.model in ['HybridViTCNN', 'CNNBeforeViT', 'ViTBeforeCNN']:
        logger = get_logger(args.log_dir, args.dataset, str(args.cnn_ratio) + '_' + args.model + '_' + args.cnn_model)
    else:
        logger = get_logger(args.log_dir, args.dataset, args.model)
    logger.info(args)

    # Load the dataset
    train_loader, val_loader, test_loader = load_datasets(
        os.path.join(args.data_dir, args.dataset, 'train'),
        os.path.join(args.data_dir, args.dataset, 'test'),
        args,
        args.batch_size,
        test_only=test_only
    )

    logger.info(f'Number of classes: {args.num_classes}')
    logger.info(f'# Train images: {len(train_loader.dataset)}')
    logger.info(f'# Val   images: {len(val_loader.dataset)}')
    logger.info(f'# Test  images: {len(test_loader.dataset)}')

    # Create the model
    model = create_model(args).to(device)

    # Create the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_rate)

    # Initialize variables to track the best epoch and accuracy
    best_epoch = 0
    best_test_acc = 0.0
    best_val_acc = 0.0

    for epoch in range(args.num_epochs):

        # Train the model
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

        # Test the model on validation set
        val_loss, val_acc = test(model, val_loader, criterion, device)

        # Test the model on test set
        test_loss, test_acc = test(model, test_loader, criterion, device)

        logger.info(f'Epoch {epoch + 1}/{args.num_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        logger.info(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        logger.info(f'Test  Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

        # Check if this is the best epoch so far based on validation accuracy
        if val_acc > best_val_acc:
            best_epoch = epoch + 1
            best_test_acc = test_acc
            best_val_acc = val_acc

        # logger.info('-' * 20)

    # Print the best epoch and test accuracy
    logger.info(f'Best epoch: {best_epoch}, Test accuracy: {best_test_acc:.4f}')

    return best_test_acc


def valid_process():
    """
    Test all models
    """
    args = parse_arguments()
    cnn_models = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', ]
    vision_models = ['ViT']
    hybrid_models = ['CNNBeforeViT', 'ViTBeforeCNN', 'HybridViTCNN']
    all_models = cnn_models + vision_models + hybrid_models
    args.epoch = 3
    for model in all_models:
        print(f"===== Testing {model} =====")
        args.model = model
        if model in hybrid_models:
            for cnn in cnn_models:
                args.model = model
                args.cnn_model = cnn
                print(f"  with {cnn}")
                main(args, test_only=True)
        else:
            main(args, test_only=True)


def count_all_parameters():
    """
    count the number of parameters in all models
    """
    args = parse_arguments()
    cnn_models = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', ]
    vision_models = ['ViT']
    hybrid_models = ['CNNBeforeViT', 'ViTBeforeCNN', 'HybridViTCNN']
    all_models = cnn_models + vision_models + hybrid_models
    import csv
    with open('parameters.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model', 'parameters'])
        for model_name in all_models:
            print(f"===== Counting {model_name} =====")
            args.model = model_name
            if model_name in hybrid_models:
                for cnn in cnn_models:
                    args.model = model_name
                    args.cnn_model = cnn
                    model = create_model(args)
                    writer.writerow([f"{model_name}_{cnn}", count_parameters(model)])
            else:
                model = create_model(args)
                writer.writerow([f"{model_name}", count_parameters(model)])


def count_parameters(model):
    """
    count the number of parameters in the model
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return float(f'{num_params / 1e6:.2f}')  # millions


if __name__ == "__main__":
    # valid_process()
    # count_all_parameters()
    main(parse_arguments())
