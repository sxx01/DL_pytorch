import torch
import matplotlib.pyplot as plt

# x = torch.arange(-8., 8., 0.1, requires_grad=True)
# y = torch.relu(x)
# fig = plt.figure(figsize=(5, 2.5))
# plt.plot(x.detach(), y.detach(), 'x', 'relu(x)')
# y.backward(torch.ones_like(x), retain_graph=True)
# plt.plot(x.detach(), x.grad, 'x', 'grad of relu')
# plt.show()

x = torch.arange(-8., 8., 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x), retain_graph=True)
# plt.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)')
plt.plot(x.detach(), x.grad, 'x', 'grad of sigmoid')
plt.show()
