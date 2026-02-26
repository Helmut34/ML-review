import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split (
    X, y,
    test_size=.20,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



X_train = torch.tensor(X_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


n_features = X_train.shape[1]

W = torch.randn(n_features, 1, requires_grad=True)

b = torch.zeros(1, requires_grad=True)



def forward(X):
    z = X @ W + b

    return torch.sigmoid(z)

criterion = torch.nn.BCELoss()



learning_rate = 0.01

epoch = 1000

losses = []

for epoch in range(epoch):
    
    y_pred = forward(X_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.item())

    loss.backward()

    with torch.no_grad():

        W -= learning_rate * W.grad

        b -= learning_rate * b.grad
    W.grad.zero_()
    b.grad.zero_()


    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epoch} | Loss: {loss.item():.4f}")


with torch.no_grad():
        y_test_pred = forward(X_test)
        test_loss = criterion(y_test_pred, y_test)
        predictions = (y_test_pred >= 0.5).float()
        accuracy = (predictions == y_test).float().mean()
        print(f"Accuracy: {accuracy.item():.4f}")
        print(f"\nTest BCE: {test_loss.item():.4f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training Loss Curve")
plt.savefig("loss_curve.png")
print("Plot saved to loss_curve.png")

