import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='FER2013 Facial Expression Recognition')

    parser.add_argument('-d', '--dataset', type=str, default='FER2013',
                        choices=['FER2013', 'CK+', 'JAFFE', 'JAFFE_faces'],
                        help='Dataset to use')
    parser.add_argument('-m', '--model', type=str, default='ViTBeforeCNN',
                        choices=['VGG16', 'VGG19',
                                 'ResNet50', 'ResNet101', 'ResNet152',
                                 'ViT',
                                 'HybridViTCNN', 'CNNBeforeViT', 'ViTBeforeCNN'],
                        help='Model to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pre-trained model')
    parser.add_argument('-cnn', '--cnn_model', type=str, default='VGG16',
                        choices=['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152'],
                        help='CNN model to use for hybrid model')

    parser.add_argument('-r', '--cnn_ratio', type=float, default=0.5,
                        help='Ratio of CNN features to use for HybridViTCNN')

    # ViT parameters
    parser.add_argument('-p', '--patch_size', type=int, default=6, help='Patch size for ViT')
    parser.add_argument('-n', '--num_heads', type=int, default=12, help='Number of heads for ViT')
    parser.add_argument('--dim', type=int, default=768, help='Dimension for ViT')
    # parser.add_argument('-mlp', '--mlp_dim', type=int, default=2048, help='MLP dimension for ViT')
    parser.add_argument('-l', '--num_layers', type=int, default=12, help='Number of layers for ViT')
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0.1, help='Dropout rate for ViT')

    # Training parameters
    parser.add_argument('-e', '--num_epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size for training and testing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        choices=[0.01, 0.001, 0.0001],
                        help='Learning rate for the optimizer')
    parser.add_argument('-l2', '--l2_rate', type=float, default=0, help='L2 regularization rate for the optimizer')

    # Other parameters
    parser.add_argument('-ld', '--log_dir', type=str, default='logs', help='Directory to store the log files')
    parser.add_argument('-dd', '--data_dir', type=str, default='datasets', help='Directory to store the dataset')
    parser.add_argument('--device', type=int, default=0, help='Device to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    # Auto set some arguments
    args.device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'FER2013':
        args.image_size = 48
        args.patch_size = 6
        args.num_classes = 7
    else:
        args.image_size = 224
        args.patch_size = 16
        args.num_classes = 7
        # raise NotImplementedError

    return args
