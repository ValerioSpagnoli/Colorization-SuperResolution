import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_dense import RDN
import torchvision.transforms as transforms

class RestorationSubNet(nn.Module):
    def __init__(self, weights=None, scale=2):
        super(RestorationSubNet, self).__init__()
        
        # --------------------------------------------------------------------------------------------- #  
        # Parameters 

        self.num_features = 64 
        self.growth_rate = 64 
        self.num_blocks = 16 
        self.num_layers = 8
        self.scale = scale
        self.weights_file = f"{weights}/rdn_x{self.scale}.pth" 
        
        # --------------------------------------------------------------------------------------------- #
        # Define layers

        # Residual Dense Networks:
        # - RDNx1 take as input the original grayscale image
        # - RDNx4 take as input the original grayscale image scaled by a factor of 4
        # - RDNx8 take as input the original grayscale image scaled by a factor of 8 

        self.RDNx1 = RDN(scale_factor=self.scale, num_channels=3, num_features=self.num_features, growth_rate=self.growth_rate, num_blocks=self.num_blocks, num_layers=self.num_layers)
        self.RDNx4 = RDN(scale_factor=self.scale, num_channels=3, num_features=self.num_features, growth_rate=self.growth_rate, num_blocks=self.num_blocks, num_layers=self.num_layers)
        self.RDNx8 = RDN(scale_factor=self.scale, num_channels=3, num_features=self.num_features, growth_rate=self.growth_rate, num_blocks=self.num_blocks, num_layers=self.num_layers)

        # This conv2d layer takes the concatenated output of RDNx1, RDNx2, RDNx8

        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=3 // 2)

        # --------------------------------------------------------------------------------------------- #
        # Load weights

        state_dict_x1 = self.RDNx1.state_dict()
        state_dict_x4 = self.RDNx4.state_dict()
        state_dict_x8 = self.RDNx8.state_dict()

        for n, p in torch.load(self.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict_x1.keys():
                state_dict_x1[n].copy_(p)
                state_dict_x4[n].copy_(p)
                state_dict_x8[n].copy_(p)
            else:
                raise KeyError(n)



    def forward(self, x):


        # Transform the RGB image to grayscale, but keeping three channels
        to_grayscale = transforms.Grayscale(num_output_channels=3)
        x = to_grayscale(x)

        # Create the downscaled images
        x4 = F.interpolate(x, scale_factor=.25, mode='bilinear')
        x8 = F.interpolate(x, scale_factor=.125, mode='bilinear')

        # Flows into RDNs
        outx1 = self.RDNx1(x)
        outx4 = self.RDNx4(x4)
        outx8 = self.RDNx8(x8)

        # Upscale the output of RDNs
        outx4 = F.interpolate(outx4, scale_factor=4, mode='bilinear')
        outx8 = F.interpolate(outx8, scale_factor=8, mode='bilinear')

        # print(outx1.shape)
        # print(outx4.shape)
        # print(outx8.shape)

        # ---------------------------------------------------------------------- #

        # Denormalize and clip the results, then take only the first channel from each image (are equal)
        # img.shape = [1, h, w]
        # imgx4.shape = [1, h, w]
        # imgx8.shape = [1, h, w]

        denormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                          transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
        
        img = torch.clip(denormalize(outx1[0]), 0, 1)
        img = img[0, :, :]
        img = img.reshape(1, img.shape[0], img.shape[1])

        imgx4 = torch.clip(denormalize(outx4[0]), 0, 1)
        imgx4 = imgx4[0, :, :]
        imgx4 = imgx4.reshape(1, imgx4.shape[0], imgx4.shape[1])

        imgx8 = torch.clip(denormalize(outx8[0]), 0, 1)
        imgx8 = imgx8[0, :, :]
        imgx8 = imgx8.reshape(1, imgx8.shape[0], imgx8.shape[1])

        # print(img.shape)
        # print(imgx4.shape)
        # print(imgx8.shape)

        # ---------------------------------------------------------------------- #

        # concatenate the output -> out.shape = [3, h, w]
        out = torch.cat((img, imgx4, imgx8), dim=0)
        
        # print(out.shape)

        # ---------------------------------------------------------------------- #

        # Flow into conv layer to obtain only one-layer image in output -> out.shape = [1, h, w]
        out = self.conv(out)

        return out