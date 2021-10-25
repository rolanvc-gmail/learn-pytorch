import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import seaborn as sns
import pandas

import matplotlib.pyplot as plt

sns.set_theme()

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device: {}".format(device, torch.cuda.get_device_name(0)))


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
accuracies = []
losses = []


def train(dataloader, the_model, the_loss_fn, the_optimizer):
    size = len(dataloader.dataset)
    the_model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = the_model(X)  # As expected, pred is Tensor(64,10). This is (batch-size, num_classes).
        loss = the_loss_fn(pred, y)  # loss is Tensor()

        # Backpropagation
        the_optimizer.zero_grad()
        loss.backward()
        the_optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, the_model, the_loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    the_model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = the_model(X)
            test_loss += the_loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    losses.append(test_loss)
    accuracies.append(100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 30
epoch_arr = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    epoch_arr.append(t)

mydata = {
    'epochs': epoch_arr,
    'accuracy': accuracies,
    'losses': losses
}

myvar = pandas.DataFrame(mydata)
myvar.to_pickle("losses.pkl")
print(myvar)
print("Done!")


