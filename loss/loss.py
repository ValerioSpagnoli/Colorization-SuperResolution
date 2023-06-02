import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LuminanceReconstructionLoss(nn.Module):
    def __init__(self):
        super(LuminanceReconstructionLoss, self).__init__()

    def forward(self, input_image, target_image):
        loss = F.l1_loss(input_image, target_image)
        return loss
    

# class PerceptualLoss(nn.Module):
#     def __init__(self, layers):
#         super(PerceptualLoss, self).__init__()
#         self.layers = layers
#         self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=3 // 2)
#         self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()#.features[:max(layers)+1].eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def forward(self, input_image, target_image):

#         print(input_image.shape)

#         input_image = self.conv(input_image)
#         taget_image = self.conv(target_image)

#         print(input_image.shape)

#         input_features = self.vgg(input_image)
#         target_features = self.vgg(target_image)
#         loss = 0.0

#         for layer in self.layers:
#             input_feat = input_features[layer]
#             target_feat = target_features[layer]
#             loss += F.mse_loss(input_feat, target_feat)

#         return loss

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss