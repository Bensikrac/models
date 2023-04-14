import os
import random

import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader


def main():
    # init device
    #device = (
    #    "cuda"
    #    if torch.cuda.is_available()
    #    else "mps"
    #    if torch.backends.mps.is_available()
    #    else "cpu"
    #)
    device = "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    print(model)

    training_data = CustomImageDataset(annotations_file="annotations.csv", img_dir="images")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Learn here
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    # LOSS FN
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(train_dataloader, model, loss_fn)
    print("Done!")

    # X = torch.rand(1, 1920, 1080, device=device)
    # logits = model(X)
    # logits = model(PIL.Image.open("images/image-0000000020.png", mode="r"))
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    # torch.save(model.state_dict(), "model.pth")
    # model = NeuralNetwork()
    # model.load_state_dict(torch.load("model.pth"))
    # model.eval()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1920 * 1080, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


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
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    main()
