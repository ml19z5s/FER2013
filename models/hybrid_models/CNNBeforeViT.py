import torch.nn as nn

from models.vision_transformer.ViT import VisionTransformer


class CNNBeforeViT(nn.Module):
    def __init__(self, cnn, vision_transformer):
        super().__init__()
        self.cnn = cnn
        self.vision_transformer = vision_transformer

    def forward(self, x):
        x = self.cnn(x)
        # x: [B, 512 * 7 * 7]
        # transform x to [B, 512, 7, 7]
        x = x.view(x.size(0), -1, 7, 7)  # TODO hard code
        x = self.vision_transformer(x)
        return x


def default_cnn_before_vit_model(cnn: nn.Module,
                                 cnn_model='resnet50',
                                 img_size=7,
                                 patch_size=1,
                                 embed_dim=768,
                                 depth=12,
                                 num_heads=12,
                                 representation_size=768,
                                 num_classes=7,
                                 has_logits=True):
    # Instantiate the CNN and ViT
    if cnn_model not in ['ResNet50', 'ResNet101', 'ResNet152', 'VGG16', 'VGG19']:
        raise NotImplementedError
    if 'resnet' in cnn_model.lower():
        # Change the avgpool layer to match the ViT input size
        cnn.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        in_c = 2048  # The last conv layer output channel
    else:
        # VGG
        cnn.classifier = nn.Identity()
        in_c = 512
    cnn.fc = nn.Identity()
    vision_transformer = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_c=in_c,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        representation_size=representation_size if has_logits else None,
        num_classes=num_classes
    )
    return CNNBeforeViT(cnn, vision_transformer)
