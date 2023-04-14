import os

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.killshot.newmodel import OwnNeuralNet, CustomImageDataset


def imshow(img: torch.Tensor):
    img = img / 2 + 0.5
    print(img.size())
    print(np.transpose(img, (1, 2, 0)).size())
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def main():
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
    model = OwnNeuralNet()
    model.load_state_dict(torch.load("valorantclassifier.pth"))
    model.eval()
    model.to(device)
    img_path = "testimage2.png"
    """
    image = Image.open(img_path)
    image = image.convert('RGB')
    print(transforms.ToTensor()(image).size())
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = transform(image).to(device)
    print(model(image))"""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_data = CustomImageDataset(annotations_file="annotations.csv", img_dir="F:/datascience/waldo/images",
                                       transform=transform)
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    testimage2 = Image.open(img_path).convert("RGB")
    testimage2 = transform(testimage2)
    testimage2 = testimage2.to(device)
    print("Testimage Size")
    print(testimage2.size())
    testimage2 = testimage2[None, :] # add single dimension, because dataloader adds it because it is batch_size why
    # not?

    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    images = images.to(device)
    labels = labels.to(device)
    print("Images Size")
    print(images.size())
    outputs = model(testimage2)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)


if __name__ == "__main__":
    main()
