#This is a continuation of my Linear regression mini project, Goal is to improve MSE


import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing() #loading dataset

#X are the feature matrixs, this dataset contains 8.  Y, are the target values (median house pricess)
X, y = data.data, data.target


#preprocess training data, standard is 80 to 20 train to test set.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42             #Seed number for reproducibility every run we do the same split.
)


#Now the dataset features must be scaled, without scaling gradient decent will over-correct for larger scale features and under0coirrect for smaller ones. 
#StandarScalar does this automatically. 
#Each feature to mean=0, std= 1.
#This allow the gradient descent function to converge efficiently.

scaler = StandardScaler() #init scalar object

X_train = scaler.fit_transform(X_train) #Transform and FIT on training data only

X_test = scaler.transform(X_test) #ONLY transform test data.


#Because of the Overcomplicated nature of this project, now i need to convert numpy
#arrays to pytorch tensors.

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

y_train = y_train.reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)

#now we can use the W&B from pytorch
# W being a number per feature
#b being a single number that moves the prediction up or down

n_features = X_train.shape[1]

W = torch.randn(n_features, 64, requires_grad=True) #required_grad tells pytorch to track all operations in order to computer Loss duing backpropagation

b = torch.zeros(64, requires_grad=True)

W2 = torch.randn(64, 32, requires_grad=True) #required_grad tells pytorch to track all operations in order to computer Loss duing backpropagation

b2 = torch.zeros(32,requires_grad=True)

W3= torch.randn(32, 1, requires_grad=True) #required_grad tells pytorch to track all operations in order to computer Loss duing backpropagation

b3 = torch.zeros(1,requires_grad=True)

#Now we define our forward layer to actuate learning
relu = nn.ReLU()

def forward(X):
    p = X @ W + b

    a1 = relu(p)

    a2 = a1 @ W2 + b2

    a3 = relu(a2)

    output = a3 @ W3 + b3

    return output

#adding a RELU activation layer



#Calculate MSE LOSS

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()



#Create the training loop

# learning_rate=0.01 #How big of a step forwardis taken per update. 
#                     #cannot be too high or too low.

epochs = 2     # How many times we loop through the entire loop

losses = []             #store each loss in order to plot it

optimizer = torch.optim.Adam(
    [W, b, W2, b2, W3, b3],
    lr=0.001

)
for epoch in range(epochs):

    #call forward pass
    y_pred = forward(X_train)

    #figure out loss
    loss = mse_loss(y_pred, y_train)

    losses.append(loss.item())

    #backwards pass
    loss.backward()         #PyTorch walks backword through the computations and computes the 
                            #How much the W and B contributed to the loss
                            # So then we have gradients pointing to the direction of increasing loss
    # with torch.no_grad(): #must turn off gradient to decrase memory usage (pytorch would try to build a graph of the update)

        # W -= learning_rate * W.grad #The learning rate is what controls the step,
        #                             #we must move in the OPPOSITE direction of the gradient
        # b -= learning_rate * b.grad

    # W.grad.zero_()                  #Must zero the gradients beacuse
                                    #pyTorch accumulates gradients by default, so backward
                                    #Call would combine each epochs gradient
    # b.grad.zero_()
    # W2.grad.zero_()
    # b2.grad.zero_()
    # W3.grad.zero_()
    # b3.grad.zero_()


    optimizer.step()


    optimizer.zero_grad()



    if(epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")


#Final Step is to Evaluate on trained test data. This is the final test to 
#verify model performance on generalized data it has never seen before

with torch.no_grad():
    y_test_pred = forward(X_test)
    test_loss = mse_loss(y_test_pred, y_test)
    print(f"\nTest MSE: {test_loss.item():.4f}")
 

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.savefig("loss_curve.png")
print("Plot saved to loss_curve.png")
