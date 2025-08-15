import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn


class HandWrittenDigits(nn.Module):

    """
    For default MNIST 28 x 28 image
    """
    def __init__(self):
        super(HandWrittenDigits, self).__init__()

        """ 
        Layer 1: Determine low level features
        
        Let us try with these parameters
        1. output channels aka low level features = 5
        2. kernel size aka convolutional filter size = 3x3
        """
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=4)

        """
        Layer 2: Determine high level features
        
        Let us try with these parameters
        1. output channels aka high level features = 10
        2. kernel size aka convolutional filter size = 3x3
        """
        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=10, kernel_size=4)

        """
        How linear is calculated?
        The default MNIST images are 28x28 so apply the formula from
        
        
        """
        self.linear1 = torch.nn.Linear(in_features=320, out_features=10)

    def forward(self, x):

        """
        Calculations:
        out = ((in - kernel + 2* padding)/ stride) + 1
        i.e ((28 - 3 + 0)/1) +1 = 26
        so after first Conv the output is a 5x26x26
        """
        conv1 = self.conv1(x)

        """
        Calculations:
        out = in / kernel
        out = (26,26) / 2
        after first max_pooling we get a 5x13x13 output
        """
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        """
        Calculations:
        out = ((in - kernel + 2* padding)/ stride) + 1
        i.e ((13 - 3 + 0)/1) + 1 = 11
        so after first Conv the output is a 10x11x11
        """
        conv2 = self.conv2(pool1)

        """
        Calculations:
        out = out = in / kernel
        i.e (11,11) / 2 = 5.5 (not desirable, going to be missing data)
        after first max_pooling we get a 10x5x5 output
        """
        pool2 = F.max_pool2d(conv2, kernel_size=2)

