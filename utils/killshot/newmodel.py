import numpy as np
import torch
import torchvision.utils
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import torch.optim as optim


def imshow(img: torch.Tensor):
    img = img / 2 + 0.5
    print(img.size())
    print(np.transpose(img, (1, 2, 0)).size())
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def main():
    print("started")
    # Setup Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = "cpu"
    print(f"Using {device} device")
    # Setup Dataset and Dataloader
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_data = CustomImageDataset(annotations_file="annotations.csv", img_dir="F:/datascience/waldo/images",
                                       transform=transform)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    testing_data = CustomImageDataset(annotations_file="annotations-tests.csv", img_dir="F:/datascience/waldo/images",
                                      transform=transform)
    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=True)

    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)

    # Display test image
    imshow(torchvision.utils.make_grid(images))

    # initialize own neural net
    net = OwnNeuralNet().to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train network here
    for epoch in range(2):  # loop over the dataset multiple times
        print(f"Epoch {epoch + 1}\n-------------------------------")
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network o nthe 10000 test images: {100 * correct // total}')
    #torch.save(net.state_dict(), "valorantclassifier.pth")


# Here is wonderful neural network
class OwnNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels (color) 6 output channels?
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # here simple node reduction, which should work
        self.fc1 = nn.Linear(2037744, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Our own Dataloader
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)  # here image is converted to tensor already
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    main()
