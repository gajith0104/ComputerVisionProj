import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from torchvision.datasets import MNIST


class ConvolutionalNN(nn.Module):


    # For default MNIST 28 x 28 image

    def __init__(self):
        super(ConvolutionalNN, self).__init__()

        # Layer 1: Determine low level features
        #
        # Let us try with these parameters
        # 1. output channels aka low level features = 5
        # 2. kernel size aka convolutional filter size = 3x3

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, padding=2, kernel_size=3)

        # Layer 2: Determine high level features
        #
        # Let us try with these parameters
        # 1. output channels aka high level features = 10
        # 2. kernel size aka convolutional filter size = 3x3
        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)

    def forward(self, x):

        # Calculations:
        # out = ((in - kernel + 2* padding)/ stride) + 1
        # i.e ((28 - 3 + 2*1)/1) +1 = 28
        # so after first Conv the output is a 5x26x26

        # Applying ReLU to remove negative values
        conv1 = F.relu(self.conv1(x))

        # Calculations:
        # out = in / kernel
        # out = (28,28) / 2
        # after first max_pooling we get a 5x14x14 output
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        # Calculations:
        # out = ((in - kernel + 2* padding)/ stride) + 1
        # i.e ((14 - 3 + 0)/1) + 1 = 12
        # so after first Conv the output is a 10x11x11
        # 
        # Applying ReLU to remove negative values
        conv2 = F.relu(self.conv2(pool1))

        # Calculations:
        # out = out = in / kernel
        # i.e (12,12) / 2 = 6
        # after first max_pooling we get a 10x6x6 output
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        # Flatten it into a vector
        flatten = F.relu(torch.flatten(pool2, 1))

        # Apply probability distribution on the output
        output = F.log_softmax(flatten, dim=1)
        return output

class MINSTLoader():

    def __init__(self):
        self.file = None
        self.inputType = None
        self.trainSize = 0
        self.testSize = 0
        self._trainDataset = None
        self._testDataset = None

    def load(self, file="./data", trainSize=500, testSize=100):
        self.trainSize = trainSize
        self.testSize = testSize

        # Verify filepath exists otherwise download data

        self.file = file
        trainDataset = None
        testDataset = None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        downloadData = not os.path.exists(self.file+"/MNIST")

        if downloadData:
            print("downloading MNIST dataset")
        else:
            print("running existing MNIST dataset")

        trainDataset = MNIST(root=file, train=True, download=downloadData, transform=transform)
        testDataset = MNIST(root=file, train=False, download=downloadData, transform=transform)

        # Load datasets into memory
        self._trainDataset = DataLoader(trainDataset, batch_size=self.trainSize, shuffle=True)
        self._testDataset = DataLoader(testDataset, batch_size=self.testSize, shuffle=True)

    def getTrainDataset(self):
        return self._trainDataset

    def getTestDataset(self):
        return self._testDataset

class ConvolutionalNNTrainer():
    def __init__(self,model, trainLoader, testLoader, learningRate=0.01, device="cpu"):
        self.model = model.to(device)
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=learningRate)

    def train(self):

        # Set the model to training mode and set loss
        self.model.train()
        lossSum = 0.0

        # Run through training data through backpropagation
        for images, labels in self.trainLoader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            lossSum += loss.item()

        return lossSum / len(self.trainLoader)

    def test(self):
        # Set the model to evaluation mode and set
        self.model.eval()
        correct, total = 0, 0
        totalChange = len(self.testLoader)
        with torch.no_grad():
            for images, labels in self.testLoader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # prediction is going to be the most likely output
                prediction = outputs.argmax(dim=1)

                resultBool = (prediction == labels)
                correct += resultBool.sum().item()
                total += labels.size(0)
        return correct / total

def main():
    model = ConvolutionalNN()
    dataset = MINSTLoader()
    dataset.load()
    trainer = ConvolutionalNNTrainer(model, dataset.getTrainDataset(), dataset.getTestDataset())

    # Run model and evaluation percentages
    trainLoss = trainer.train()
    testLoss = trainer.test()
    print("Loss while training: ", trainLoss)
    print("Loss during testing: ", testLoss)


if __name__ == "__main__":
    main()

