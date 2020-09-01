if __name__ == '__main__':
    import torch

    x = torch.FloatTensor([1])
    x.requires_grad = True

    optimizer = torch.optim.SGD([x], lr=1)

    y = x * x * x * x + x * x * x + x * x

    order = 3
    for i in range(order):
        if i == order - 1:
            y.backward()
        else:
            x = torch.autograd.grad(y, x, retain_graph=True)

        print(x,x.grad)