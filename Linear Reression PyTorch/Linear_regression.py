import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



weight = 0.4
bias = 0.7

start = 0
end = 100

X = torch.arange(start, end, 1).unsqueeze(1)
Y = weight * X + bias

plt.plot(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(torch.Tensor.numpy(X), torch.Tensor.numpy(Y),train_size = 0.8)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
Y_train = torch.from_numpy(Y_train)
Y_test = torch.from_numpy(Y_test)



class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad= True, dtype= torch.float))
    
    def forward(self,data : torch.Tensor) -> torch.Tensor:
        predictions = self.weight * data + self.bias
        
        return predictions
    
model_0 = LinearRegression()





loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr= 0.000007)



print(X_train.size())

#tracking the values and training

value_loss =[]
test_value_loss = []
epoch_values = []

epoch = 10000

for a in range(epoch):
    
    #put model in training mode
    model_0.train()
    
    #perform a forward pass    
    Y_pred = model_0(X_train)
    
    #calculate the loss
    loss = loss_function(Y_pred, Y_train)
    
    #clear the gradients 
    optimizer.zero_grad()
    
    #perform a backward pass
    loss.backward()
    
    #calcuate the gradients
    optimizer.step()
    
    #put the mpdel in evaluation mode
    model_0.eval()
    
    with torch.inference_mode():
        
        Y_pred_test = model_0(X_test)
        loss_test = loss_function(Y_pred_test, Y_test)
        
        if a % 100 == 0:
            value_loss.append(loss.item())
            epoch_values.append(a)
            test_value_loss.append(loss_test.item())
            print(f"epoch: {a} | Loss: {loss} | loss in test: {loss_test}")

print(list(model_0.parameters()))

            
    
    
                
    
    
    
    
    
