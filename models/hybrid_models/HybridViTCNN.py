import torch.nn as nn
import torchvision.models as models
from models.vgg.VGG import vgg16, vgg19
from models.vision_transformer.ViT import VisionTransformer


class HybridViTCNN(nn.Module):

    def __init__(self, cnn, vision_transformer, cnn_ratio=0.5):
        super().__init__()
        self.cnn = cnn
        self.vision_transformer = vision_transformer
        self.cnn_ratio = cnn_ratio

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.vision_transformer(x)
        return self.cnn_ratio * x1 + (1 - self.cnn_ratio) * x2


def default_hybrid_vit_cnn_model(cnn: nn.Module,
                                 img_size=48,
                                 in_c=3,
                                 patch_size=6,
                                 embed_dim=768,
                                 depth=12,
                                 num_heads=12,
                                 representation_size=768,
                                 num_classes=7,
                                 has_logits=True,
                                 cnn_ratio=0.5):
    # Instantiate the CNN and ViT
    vision_transformer = VisionTransformer(img_size=img_size,
                                           in_c=in_c,
                                           patch_size=patch_size,
                                           embed_dim=embed_dim,
                                           depth=depth,
                                           num_heads=num_heads,
                                           representation_size=representation_size if has_logits else None,
                                           num_classes=num_classes,
                                           need_classifier=True,
                                           )
    return HybridViTCNN(cnn, vision_transformer, cnn_ratio)
