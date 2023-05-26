import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_dense import RDN
import numpy as np


class RestorationSubNet(nn.Module):
    def __init__(self):
        super(RestorationSubNet, self).__init__()
        self.RDNx1 = RDN(scale_factor=4, num_channels=3, num_features=3, growth_rate=3, num_blocks=3, num_layers=4)
        self.RDNx4 = RDN(scale_factor=4, num_channels=3, num_features=3, growth_rate=3, num_blocks=3, num_layers=4)
        self.RDNx8 = RDN(scale_factor=4, num_channels=3, num_features=3, growth_rate=3, num_blocks=3, num_layers=4)

    def forward(self, x):

        # heigth_x = x.shape[2]
        # width_x = x.shape[3]

        # height_x4 = int(np.ceil(heigth_x/4))
        # width_x4 = int(np.ceil(width_x/4))

        # height_x8 = np.ceil(heigth_x/8)
        # width_x8 = np.ceil(width_x/8)


        # x4 = F.interpolate(x, size=(height_x4, width_x4), mode='bilinear')
        # x8 = F.interpolate(x, size=(height_x8, width_x8), mode='bilinear')

        x4 = F.interpolate(x, scale_factor=.25, mode='bilinear')
        x8 = F.interpolate(x, scale_factor=.125, mode='bilinear')

        print(x.shape)
        print(x4.shape)
        print(x8.shape)
        print("-----")

        outx1 = self.RDNx1(x)
        outx4 = self.RDNx4(x4)
        outx8 = self.RDNx8(x8)

        print(outx1.shape)
        print(outx4.shape)
        print(outx8.shape)

        outx4 = F.interpolate(outx4, scale_factor=4, mode='bilinear')
        outx8 = F.interpolate(outx8, scale_factor=8, mode='bilinear')

        print("-----")
        print(outx1.shape)
        print(outx4.shape)
        print(outx8.shape)

        return torch.cat([outx1, outx4, outx8])