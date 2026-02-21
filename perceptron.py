import torch
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)


    def forward(self, x):
        return torch.sigmoid(self.linear(x))
                             
model = Perceptron(input_size=4)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


x = torch.randn(100, 4)
y = torch.randint(0, 2, (100, 1)).float()

for epoch in range(1000):
    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


with torch.no_grad():
    predictions = (model(x) > 0.5).float()