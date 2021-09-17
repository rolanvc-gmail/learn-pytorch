import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from cnn import CNN

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
print("Using {} device".format(device))

tb = SummaryWriter()

model = CNN().to(device)
images, labels = next(iter(train_dataloader))
images, labels = images.to(device), labels.to(device)
grid = torchvision.utils.make_grid(images)
tb.add_image("fashion-mnist images", grid)
tb.add_graph(model, images)
tb.close()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)  # As expected, pred is Tensor(64,10). This is (batch-size, num_classes).
        loss = loss_fn(pred, y)  # loss is Tensor()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, test_loss


epochs = 20
for t in range(epochs):
    tb = SummaryWriter()
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train(train_dataloader, model, loss_fn, optimizer)
    accuracy, test_loss = test(test_dataloader, model, loss_fn)
    tb.add_scalar("Loss", loss, t)
    tb.add_scalar("Accuracy", accuracy, t)
    tb.add_scalar("Test Loss", test_loss, t)
    tb.add_histogram("conv1.bias", model.conv1.bias, t)
    tb.add_histogram("conv1.weight", model.conv1.weight, t)
    tb.add_histogram("conv2.bias", model.conv2.bias, t)
    tb.add_histogram("conv2.weight", model.conv2.weight, t)

print("Done!")
