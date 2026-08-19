"""# Artificial Neural Network

#libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#data loading and preprocessing
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), # convert images to tensor (0-255) range to (0-1) range
        transforms.Normalize((0.5,), (0.5,)) # normalize the images (-1, 1) range
    ])

#download and load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
train_loader, test_loader = get_data_loaders()
#data visualization
def visualize_data(loader):

    # Get a batch of training data
    data_iter = iter(loader)
    images, labels = data_iter.next()

    # Plot the images in the batch, along with the corresponding labels
def visualize_samples(loader, n):
    images, labels = next(iter(loader))
    print(f'Images shape: {images.shape}')
    fig, axes = plt.subplots(1, n, figsize=(10, 5))
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.show()
visualize_samples(train_loader, n=5)

#define ann model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = ANN().to(device)

define_loss_and_optimizer = lambda model: (nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001))
criterion, optimizer = define_loss_and_optimizer(model)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

train_model(model, train_loader, criterion, optimizer, num_epochs=5)

#test
def test_model(model, test_loader):
    model.eval()
    correct = 0 # COUNTER FOR CORRECT PREDICTIONS
    total = 0 # COUNTER FOR TOTAL PREDICTIONS
    with torch.no_grad(): #We don't need to compute gradients during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1) # Get the index of the max log-probability
            total += labels.size(0) # Update total count
            correct += (predicted == labels).sum().item() # Update correct count
    print("Test accuracy: {:.2f}%".format(100 * correct / total))

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
test_model(model, test_loader)

#main
if __name__=="__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_samples(train_loader, n=10)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)
"""
# Artificial Neural Network

# libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# data loading and preprocessing
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert images to tensor (0-255) range to (0-1) range
        transforms.Normalize((0.5,), (0.5,))  # normalize the images (-1, 1) range
    ])

    # download and load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# data visualization
def visualize_data(loader):
    # use the built-in next() function instead
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Plot the images in the batch, along with the corresponding labels
    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.show()


def visualize_samples(loader, n):
    images, labels = next(iter(loader))
    print(f'Images shape: {images.shape}')
    fig, axes = plt.subplots(1, n, figsize=(10, 5))
    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.show()


# define ann model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ANN(nn.Module):
    def __init__(self):
        # class name in super() must match the class itself (ANN, not NeuralNetwork)
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)  # input layer to hidden layer
        self.relu = nn.ReLU()  # activation function
        self.fc2 = nn.Linear(128, 64)  # hidden layer to hidden layer
        self.fc3 = nn.Linear(64, 10)  # hidden layer to output layer

    def forward(self, x):
        # the flatten result must be assigned back to x, otherwise
        # fc1 receives the un-flattened tensor and shapes won't match
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# loss function and optimizer
define_loss_and_optimizer = lambda model: (nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001))

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):

    train_losses = []
    test_losses = []  
    for epoch in range(num_epochs):

        # --- training pass: model learns from train_loader ---
        model.train()  # put model in training mode
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # zero the gradients
            predictions = model(images)  # forward pass
            loss = criterion(predictions, labels)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # update weights

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()  # put model in evaluation mode
        total_test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    _plot_losses(train_losses, test_losses)
    return train_losses, test_losses


def _plot_losses(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, marker="o", linestyle="-", color="b", label="Train Loss")
    plt.plot(epochs, test_losses, marker="s", linestyle="--", color="r", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


# test
def test_model(model, test_loader):
    model.eval()
    correct = 0  # counter for correct predictions
    total = 0  # counter for total predictions
    with torch.no_grad():  # we don't need to compute gradients during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)  # get the index of the max log-probability
            total += labels.size(0)  # update total count
            correct += (predicted == labels).sum().item()  # update correct count

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_samples(train_loader, n=5)

    model = ANN().to(device)

    criterion, optimizer = define_loss_and_optimizer(model)

    # this call now matches train_model's updated signature exactly:
    # (model, train_loader, test_loader, criterion, optimizer, num_epochs=...)
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=10
    )

    test_model(model, test_loader)