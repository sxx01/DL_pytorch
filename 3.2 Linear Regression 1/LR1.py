import torch
import random
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print("features: ", features[0], " labels: ", labels[0])


# fig = plt.figure(figsize=(4, 4))
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()


def data_iter(batch_size, features, labels):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_example)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, "\n", y)
#     break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    return X @ w + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
epoch_num = 3
net = linreg
loss = squared_loss

for epo in range(epoch_num):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch: {epo + 1}, train loss: {train_l.mean().item()}')
print(f'w: {w}, b: {b}')
print(f'true_w: {true_w}, true_b: {true_b}')
