import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using: {device}")


#Normalization Of data stage + augmentation functions
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])


#Normalization of test data
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

#Load train and test data
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

#Train loaders to handle batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module): #defines class inherits from nn.Module
    def __init__(self): #Constructor
        super(CNN, self).__init__() #calls the parent contructor for pytorch tracking

        #first layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) #Adds slight regularization speeds up training

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)#Input conv1's output and outputs to 64 filters
        self.bn2 = nn.BatchNorm2d(64)
                                    #Pooling layer wihgt a 2x2 windor and a stride of 2
        self.pool = nn.MaxPool2d(2, 2) #Halves computation (32 -> 16 -> 8)

        self.relu = nn.ReLU()
                                        #Forces network to not rely on single neutrons
                                        #Prevents overfitting and is automatically turned 
                                        #off during evaluation
        self.dropout = nn.Dropout(0.5)  #randomly sets 50 percent of neuron to 0


        self.flatten = nn.Flatten()     #reshapes feature map into a 1D vector

        self.fc1 = nn.Linear(64 * 8 * 8, 128) #First Fully connected layer, outputs 128 neurons 

        self.fc2 = nn.Linear(128, 10) #One Score per class in Cifar

    def forward(self, x):
        x = self.conv1(x)       # (batch, 3, 32, 32) → (batch, 32, 32, 32)
        x = self.bn1(x)         # same shape, just normalized
        x = self.relu(x)        # same shape, negatives become 0
        x = self.pool(x)        # (batch, 32, 32, 32) → (batch, 32, 16, 16)

        x = self.conv2(x)       # (batch, 32, 16, 16) → (batch, 64, 16, 16)
        x = self.bn2(x)         # normalized
        x = self.relu(x)        # activated
        x = self.pool(x)        # (batch, 64, 16, 16) → (batch, 64, 8, 8)

        x = self.flatten(x)
        x = self.fc1(x)         # (batch, 4096) → (batch, 128)
        x = self.relu(x)        # activated
        x = self.dropout(x) 
        x = self.fc2(x)         # (batch, 128) → (batch, 10)
        return x
#Moving on to the training loop

# INIT
model = CNN()
model = model.to(device)                #move to GPU training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 250

#Track Metrics for Plotting
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, lables in train_loader:
        #forward pass

        images, lables = images.to(device), lables.to(device)
        output = model(images)
        loss = criterion(output, lables)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += lables.size(0)
        correct += (predicted == lables).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(test_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss Curves')

ax2.plot(train_accs, label='Train Acc')
ax2.plot(val_accs, label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.set_title('Accuracy Curves')

plt.tight_layout()
plt.show()