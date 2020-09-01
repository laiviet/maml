import learn2learn as l2l
from model import CNNModel

if __name__ == '__main__':
    model = CNNModel(n_class=N).to(device)
    maml = l2l.algorithms.MAML(model, lr=0.1)
    opt = torch.optim.SGD(maml.parameters(), lr=0.001)
    for iteration in range(10):
        opt.zero_grad()
        task_model = maml.clone()  # torch.clone() for nn.Modules
        adaptation_loss = compute_loss(task_model)
        task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
        evaluation_loss = compute_loss(task_model)
        evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
        opt.step()
