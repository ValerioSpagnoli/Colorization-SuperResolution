import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock

from models.histvariotest import HistogramLayerLocal, GaussianHistogram

DEBUG = True


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


class WarpNet(nn.Module):
    '''
        Whole warp net
    '''
    def __init__(self, out_channels=64):
        super(WarpNet, self).__init__()
        # Initialize resnet34 and import its weights
        self.resnet = Resnet34(block=BasicBlock, layers=[3, 4, 6, 3])
        self.resnet.load_state_dict(resnet34(weights='DEFAULT').state_dict())

        # Initialize histogram layer
        self.hist_layer = HistogramLayerLocal()

        # Convolutional layers needed to fix channel size (for both img and ref)
        self.layer1 = nn.Sequential(
            # First convolution after resnet
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            # Second convolution (to fix channel size)
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            # First convolution after resnet
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            # Second convolution (to fix channel size)
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

        self.layer3 = nn.Sequential(
            # First convolution after resnet
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            # Second convolution (to fix channel size)
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

        self.layer4 = nn.Sequential(
            # First convolution after resnet
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            # Second convolution (to fix channel size)
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        # Learnable coefficients for weighted feature maps
        self.A = nn.Parameter(torch.ones((2,4,4)))

        # Shared convolutional filters
        self.W_img = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

        self.W_ref = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )


    def forward(self, x):
        # Extract 'L' and 'AB' channels
        tensor_L = x[:,0:1,:,:].repeat(1, 3, 1, 1)
        tensor_AB = x[:,1:3,:,:]
        batch_size = tensor_AB.shape[0]


        ## SIMILARITY MAP COMPUTATION ##

        # Extraxt the 4 feature maps using resnet34
        fm_img = self.resnet(tensor_L[0].unsqueeze(0))
        if DEBUG: print("Resnet out img: ", [t.shape for t in fm_img]) # DEBUG PRINT
        fm_ref = self.resnet(tensor_L[1].unsqueeze(0))
        if DEBUG: print("Resnet out ref: ", [t.shape for t in fm_ref]) # DEBUG PRINT

        # Apply layers (2 convolutions)
        fm_img = [self.layers[i](fm_img[i]) for i in range(4)]
        if DEBUG: print("\nConv out img: ", [t.shape for t in fm_img]) # DEBUG PRINT
        fm_ref = [self.layers[i](fm_ref[i]) for i in range(4)]
        if DEBUG: print("Conv out ref: ", [t.shape for t in fm_ref]) # DEBUG PRINT

        # Compute M weighted concatenations and apply convolution to it
        M_img = [self.W_img(torch.cat([F.interpolate(self.A[0,i,j]*fm_img[j], size=fm_img[i].shape[-2:], mode='bilinear') for j in range(4)], dim=1)) for i in range(4)]
        if DEBUG: print("\nM img: ", [t.shape for t in M_img]) # DEBUG PRINT
        M_ref = [self.W_ref(torch.cat([F.interpolate(self.A[1,i,j]*fm_ref[j], size=fm_ref[i].shape[-2:], mode='bilinear') for j in range(4)], dim=1)) for i in range(4)]
        if DEBUG: print("M res: ", [t.shape for t in M_ref]) # DEBUG PRINT

        # Reshape from 3D to 2D (from HxWxC to HWxC)
        M_img = [M_img[i].view(1, M_img[i].shape[-3], -1) for i in range(4)]
        if DEBUG: print("\nM reshaped img: ", [t.shape for t in M_img]) # DEBUG PRINT
        M_ref = [M_ref[i].view(1, M_ref[i].shape[-3], -1) for i in range(4)]
        if DEBUG: print("M reshaped ref: ", [t.shape for t in M_ref]) # DEBUG PRINT

        # Compute (M - mean(M)) and it's norm + epsilon
        I = [M_img[i]-M_img[i].mean(dim=-1, keepdim=True) for i in range(4)]
        R = [M_ref[i]-M_img[i].mean(dim=-1, keepdim=True) for i in range(4)]
        norm_I = [torch.norm(I[i], p=2, dim=1, keepdim=True) + sys.float_info.epsilon for i in range(4)]
        norm_R = [torch.norm(R[i], p=2, dim=1, keepdim=True) + sys.float_info.epsilon for i in range(4)]

        # Compute softmax((Mi - mean(Mi))*(Mr - mean(Mr))) / (norm(Mi - mean(Mi)) * (norm(Mr - mean(Mr)))
        sm = [F.softmax(torch.matmul(torch.div(I[i], norm_I[i]).permute(0, 2, 1), torch.div(R[i], norm_R[i])).unsqueeze(dim=1), dim=-1) for i in range(4)]
        if DEBUG: print("\nSM: ", [t.shape for t in sm])


        ## HISTOGRAM COMPUTATION ##
            
        # Downsample the AB channels tensors
        AB_down = [F.adaptive_avg_pool2d(tensor_AB, (tensor_AB.shape[2] // 2**(2+i), tensor_AB.shape[3] // 2**(2+i))) for i in range(4)]
        if DEBUG: print("\nAB channels reshape:", [t.shape for t in AB_down])

        # Compute the histograms
        hist = [self.hist_layer(AB_down[i], AB_down[i]) for i in range(4)]
        if DEBUG: print("\nHist shape:", [t.shape for t in hist])

        # Reshape histogram to HWxK and permute (to allow matrix multiplication)            
        reshaped_hist = [hist[i].view(batch_size, 512, -1).permute(0, 2, 1) for i in range(4)]
        if DEBUG: print("Hist reshape to HWxK:", [t.shape for t in reshaped_hist])

        # Multiply similarity map with histogram and permute (to restore order)
        y_hist = [torch.matmul(sm[i][0], reshaped_hist[i]).permute(0, 2, 1).contiguous() for i in range(4)]
        if DEBUG: print("\ny_hist shape:", [t.shape for t in y_hist])

        # Reshape to HxWxK
        reshaped_y_hist = [y_hist[i].view(batch_size, 512, hist[i].shape[2], hist[i].shape[3]) for i in range(4)]
        if DEBUG: print("y_hist reshaped:", [t.shape for t in reshaped_y_hist])

        return (sm, reshaped_y_hist)