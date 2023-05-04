import torch.nn as nn
import torchvision.models as models

from models.vgg import *
from models.hybrid_models import *
from models.vision_transformer import *
from models.SimpleCNN import SimpleCNN  # For testing purposes


def pretrained_vgg(name, num_classes):
    if name == 'VGG16':
        vgg_model = models.vgg16(pretrained=True)
    elif name == 'VGG19':
        vgg_model = models.vgg19(pretrained=True)
    else:
        raise NotImplementedError

    vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, num_classes)
    return vgg_model


def pretrained_resnet(name, num_classes):
    if name == 'ResNet50':
        resnet_model = models.resnet50(pretrained=True)
    elif name == 'ResNet101':
        resnet_model = models.resnet101(pretrained=True)
    elif name == 'ResNet152':
        resnet_model = models.resnet152(pretrained=True)
    else:
        raise NotImplementedError

    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    return resnet_model


def get_pretrained_cnn(name, num_classes):
    if 'VGG' in name:
        cnn = pretrained_vgg(name, num_classes)
    elif 'ResNet' in name:
        cnn = get_cnn_model(name, num_classes)
    else:
        raise NotImplementedError
    return cnn


def get_cnn_model(name, num_classes):
    if name == 'ResNet50':
        cnn = models.resnet50(num_classes=num_classes, pretrained=False)
    elif name == 'ResNet101':
        cnn = models.resnet101(num_classes=num_classes, pretrained=False)
    elif name == 'ResNet152':
        cnn = models.resnet152(num_classes=num_classes, pretrained=False)
    elif name == 'VGG16':
        cnn = vgg16(num_classes=num_classes)
        # cnn = pretrained_vgg()
    elif name == 'VGG19':
        cnn = vgg19(num_classes=num_classes)
    else:
        raise NotImplementedError
    return cnn


def create_model(args):
    """
    Creates a model based on the model name
    """
    # A simple CNN for testing purposes
    if args.model == 'SimpleCNN':
        model = SimpleCNN(num_classes=args.num_classes)
    # CNNs
    elif args.model in ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152']:
        if args.pretrained:
            model = get_pretrained_cnn(args.model, args.num_classes)
        else:
            model = get_cnn_model(args.model, args.num_classes)
    # ViT
    elif args.model == 'ViT':
        model = vit_fer2013(args)

    # Hybrid models
    else:
        cnn = get_cnn_model(args.cnn_model, args.num_classes)
        if args.model == 'CNNBeforeViT':
            model = default_cnn_before_vit_model(cnn=cnn, cnn_model=args.cnn_model, num_classes=args.num_classes,
                                                 )
        elif args.model == 'ViTBeforeCNN':
            model = default_vit_before_cnn_model(cnn=cnn, cnn_model=args.cnn_model, num_classes=args.num_classes,
                                                 img_size=args.image_size)
        elif args.model == 'HybridViTCNN':
            model = default_hybrid_vit_cnn_model(cnn=cnn, num_classes=args.num_classes, cnn_ratio=args.cnn_ratio,
                                                 img_size=args.image_size)
        else:
            raise NotImplementedError

    return model
