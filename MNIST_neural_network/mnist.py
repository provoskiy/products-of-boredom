# simple neural network to predict a handwritten number from 28x28 greyscale images

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 784 pixels (28x28 flattened) so 784 inputs
class MNIST(nn.Module):
    def __init__(self ):
        super(MNIST, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
model = MNIST()

optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# get data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# train model
epochs = 9
losses = []

for i in range(epochs):
    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)
        losses.append(loss)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    #if i % 10 == 0:
    print(f"epoch: {i} and loss: {loss}")

# test model
with torch.no_grad():
    ev = model.forward(images)
    loss = criterion(ev, labels)
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        print("predicted:", predicted[:100])
        print("actual:   ", labels[:100])
        break

print(correct)
print(f"accuracy is {correct / total * 100}%")

torch.save(model.state_dict(), "mnist_model.pt")

