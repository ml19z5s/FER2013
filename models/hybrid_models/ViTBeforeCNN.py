import torch.nn as nn
import torchvision.models as models
from models.vision_transformer.ViT import VisionTransformer


class ImageRestorer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, out_c=3):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.ConvTranspose2d(embed_dim, out_c, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, _, C = x.shape
        # transpose: [B, HW, C] -> [B, C, HW]
        # unflatten: [B, C, HW] -> [B, C, H, W]
        x = x.transpose(1, 2).unflatten(2, self.grid_size)
        x = self.proj(x)
        return x


class ViTBeforeCNNModel(nn.Module):
    def __init__(self, cnn, vision_transformer, image_restorer):
        super().__init__()
        self.vision_transformer = vision_transformer
        self.image_restorer = image_restorer
        self.cnn = cnn

    def forward(self, x):
        original_x = x
        x = self.vision_transformer(x)  # [B, (H/patch_size)*(W/patch_size), embed_dim]
        x = self.image_restorer(x)  # [B, C, H, W]
        # print(x.shape, original_x.shape)
        x = 0.5 * x + 0.5 * original_x  # residual connection  # TODO
        x = self.cnn(x)  # [B, num_classes]
        return x


def default_vit_before_cnn_model(cnn: nn.Module,
                                 cnn_model='resnet50',
                                 img_size=48,
                                 in_c=3,
                                 patch_size=6,
                                 embed_dim=768,
                                 depth=12,
                                 num_heads=12,
                                 representation_size=768,
                                 num_classes=7,
                                 has_logits=True):
    # Instantiate the CNN and ViT
    if img_size % patch_size != 0:
        patch_size = img_size // 8
    vision_transformer = VisionTransformer(img_size=img_size,
                                           in_c=in_c,
                                           patch_size=patch_size,
                                           embed_dim=embed_dim,
                                           depth=depth,
                                           num_heads=num_heads,
                                           representation_size=representation_size if has_logits else None,
                                           num_classes=num_classes,
                                           need_classifier=False,
                                           )
    image_restorer = ImageRestorer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, out_c=in_c)
    model = ViTBeforeCNNModel(cnn, vision_transformer, image_restorer)
    return model
