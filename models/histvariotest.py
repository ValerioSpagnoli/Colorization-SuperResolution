import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# GaussianHistogram module represents a histogram layer that uses a Gaussian distribution
class GaussianHistogram(nn.Module):
    """
    Use gaussian distribution
    Args:
        bins: number of bins to seperate values
        min: minium vale of the data
        max: maximum value of the data
        sigma: a learable paramerter, init=0.01
    """

    def __init__(self, bins, min, max, sigma, require_grad=False):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max

        self.sigma = torch.tensor([sigma])
        self.sigma = Variable(self.sigma, requires_grad=require_grad)

        self.delta = float(max - min) / float(bins)

        # Create centers of histogram bins
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

    def forward(self, x, attention_mask=None):
        device = x.device
        self.sigma = self.sigma.to(device)
        self.centers = self.centers.to(device)

        # Calculate the distances between input values and histogram centers
        x = torch.unsqueeze(x, dim=1) - torch.unsqueeze(self.centers, 1)

        # Apply the Gaussian function to the distances and normalize the histogram values
        hist_dist = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta

        # Multiply histogram values with attention mask if provided
        if not type(attention_mask) == type(None):
            hist_dist *= torch.unsqueeze(attention_mask, 1)

        # Sum the histogram values along the last dimension
        hist = hist_dist.sum(dim=-1)

        # Normalize the histogram values
        hist = hist / torch.sum(hist, dim=1, keepdim=True)

        return hist, hist_dist


# HistogramLayerLocal module applies the GaussianHistogram module to each channel of a reference tensor
class HistogramLayerLocal(nn.Module):
    def __init__(self):
        super().__init__()
        # Create an instance of GaussianHistogram
        self.hist_layer = GaussianHistogram(bins=256, min=-1., max=1., sigma=0.01, require_grad=False)

    def forward(self, x, ref, attention_mask=None):
        channels = ref.shape[1]
        if len(x.shape) == 3:
            # Interpolate the reference tensor to match the spatial dimensions of the input tensor
            ref = F.interpolate(ref, size=(x.shape[1], x.shape[2]), mode='bicubic')

            if not type(attention_mask) == type(None):
                # Interpolate the attention mask accordingly
                attention_mask = torch.unsqueeze(attention_mask, 1)
                attention_mask = F.interpolate(attention_mask, size=(x.shape[1], x.shape[2]), mode='bicubic')
        else:
            ref = F.interpolate(ref, size=(x.shape[2], x.shape[3]), mode='bicubic')
            if not type(attention_mask) == type(None):
                attention_mask = torch.unsqueeze(attention_mask, 1)
                attention_mask = F.interpolate(attention_mask, size=(x.shape[2], x.shape[3]), mode='bicubic')
                attention_mask = torch.flatten(attention_mask, start_dim=1, end_dim=-1)

        layers = []

        # Loop over each channel of the reference tensor
        for i in range(channels):
            input_channel = torch.flatten(ref[:, i, :, :], start_dim=1, end_dim=-1)
            # Pass the flattened input channel through the GaussianHistogram module
            input_hist, hist_dist = self.hist_layer(input_channel, attention_mask)
            # Reshape the histogram distribution to match the shape of the reference tensor
            hist_dist = hist_dist.view(-1, 256, ref.shape[2], ref.shape[3])
            layers.append(hist_dist)

        # Concatenate the histogram distributions along the channel dimension
        return torch.cat(layers, 1)