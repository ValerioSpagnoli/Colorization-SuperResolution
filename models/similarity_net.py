import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock


class Resnet34(ResNet):
    '''
        Model derived from resnet34 that returns as output the 4 feature maps
    '''
    def forward(self, x):
        # Apply convolution, batch normalization, ReLU, and max pooling
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Extract the intermediate 4 feature maps
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f1, f2, f3, f4]


class SimilarityNet(nn.Module):
    '''
        Whole similarity net
    '''
    def __init__(self, out_channels):
        super(SimilarityNet, self).__init__()
        # Initialize resnet34 and import its weights
        self.resnet = Resnet34(block=BasicBlock, layers=[3, 4, 6, 3])
        self.resnet.load_state_dict(resnet34(weights='DEFAULT').state_dict())

        # Convolutional layers needed to fix channel size (for both img and ref)
        self.conv1_img = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)        
        self.conv2_img = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1)        
        self.conv3_img = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv4_img = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv_img = [self.conv1_img, self.conv2_img, self.conv3_img, self.conv4_img]

        self.conv1_ref = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)        
        self.conv2_ref = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1)        
        self.conv3_ref = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv4_ref = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv_ref = [self.conv1_ref, self.conv2_ref, self.conv3_ref, self.conv4_ref]

        # Batch normalization and relu
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Learnable coefficients for weighted feature maps
        self.A = nn.Parameter(torch.ones((2,4,4)))

        # Convolution filter
        self.conv_filter = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1) 


    def forward(self, x):
        # Extraxt the 4 feature maps using resnet34
        fm_img = self.resnet(x[0].unsqueeze(0))
        print("Resnet out shape: ", [fm_img[i][0].shape for i in range(4)]) # DEBUG PRINT
        fm_ref = self.resnet(x[1].unsqueeze(0))
        print("Resnet out shape: ", [fm_ref[i][0].shape for i in range(4)]) # DEBUG PRINT

        # Apply convolution, instace norm, relu
        fm_img = [self.relu(self.bn(self.conv_img[i](fm_img[i]))) for i in range(4)]
        print("Conv out shape: ", [fm_img[i][0].shape for i in range(4)]) # DEBUG PRINT
        fm_ref = [self.relu(self.bn(self.conv_ref[i](fm_ref[i]))) for i in range(4)]
        print("Conv out shape: ", [fm_ref[i][0].shape for i in range(4)]) # DEBUG PRINT

        # Compute M weighted concatenations
        M_img = [self.conv_filter(torch.cat([self.A[0,i,j]*F.interpolate(fm_img[j], size=fm_img[i].shape[-2:], mode='bilinear') for j in range(4)])) for i in range(4)]
        print("M shapes: ", [M_img[i].shape for i in range(4)]) # DEBUG PRINT
        M_ref = [self.conv_filter(torch.cat([self.A[1,i,j]*F.interpolate(fm_ref[j], size=fm_ref[i].shape[-2:], mode='bilinear') for j in range(4)])) for i in range(4)]
        print("M shapes: ", [M_ref[i].shape for i in range(4)]) # DEBUG PRINT

        # Compute similarity maps
        a = [M_img[i]-torch.mean(M_img[i]) for i in range(4)]
        b = [M_ref[i]-torch.mean(M_ref[i]) for i in range(4)]
        sm = [(a[i] * b[i]) / (torch.norm(a[i], p=2) * torch.norm(b[i], p=2)) for i in range(4)]



        return sm