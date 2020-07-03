import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_data = torch.FloatTensor([[1], [2], [3]])
y_data = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    model = x_data * W + b
    cost = torch.mean((model - y_data) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), b.item(), cost.item()
     ))