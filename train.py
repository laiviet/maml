import torch
import torch.nn as nn
import torchvision
from data_loader import *
from torch.utils.data import DataLoader
import copy
from collections import OrderedDict
from model import CNNModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

B, N, K, Q = 4, 5, 5, 15

from learn2learn import clone_module, clone_parameters


def accuracy(logits, targets):
    # print('| accuracy: logits ', tuple(logits.shape))
    # print('| accuracy: targets ', tuple(targets.shape))
    logits = logits.view(-1, logits.shape[-1])
    targets = targets.view(-1)

    preds = torch.argmax(logits, dim=-1)

    # print(logits)

    # print(preds)
    # print(targets)
    return (targets == preds).float().mean().item() * 100


# def gradient_update_parameters(model,
#                                loss,
#                                params=None,
#                                step_size=0.1):
#     if params is None:
#         params = OrderedDict(model.meta_named_parameters())
#     grads = torch.autograd.grad(loss,
#                                 params.values(),
#                                 create_graph=True)
#     updated_params = OrderedDict()
#     for (name, param), grad in zip(params.items(), grads):
#         updated_params[name] = param - step_size * grad
#
#     return updated_params


def investigate(v):
    print(v.data, v.grad)


def mse(a, b=None):
    if b is not None:
        err = a - b
        print(torch.mean(err ** 2))
    else:
        print(torch.mean(a ** 2))


def adapt(model, inputs, targets, adaptation_steps, step_size=0.1):
    model.train()
    ce = nn.CrossEntropyLoss()
    for step in range(adaptation_steps):
        # print(model.linear.bias)
        logits = model(inputs)
        inner_loss = ce(logits, targets)
        grads = torch.autograd.grad(inner_loss,
                                    model.parameters(),
                                    create_graph=True,
                                    retain_graph=True)
        # print(type(grads))
        # acc = accuracy(logits, targets)
        # print('Adap@{} {:.2f} {:.4f}'.format(step, acc, inner_loss.item()))
        for v, g in zip(model.parameters(), grads):
            v.data = v.data - step_size * g
    # exit(0)
    return model


def do_train(model, train_dl):
    adaptation_step = 5
    step_size = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ce = torch.nn.CrossEntropyLoss()
    model.train()

    mean = lambda x: sum(x) / len(x)
    for i, batch in enumerate(train_dl):
        optimizer.zero_grad()
        batch_accuracies = []
        batch_losses = []
        for task_id, (support, query, support_targets, query_targets) in enumerate(batch):
            support = support.to(device)  # B,C,W,H
            query = query.to(device)
            support_targets = support_targets.to(device)
            query_targets = query_targets.to(device)

            clone = clone_module(model)
            clone = adapt(clone, support, support_targets,
                          adaptation_steps=adaptation_step,
                          step_size=step_size)

            logits = clone(query)
            outer_loss = ce(logits, query_targets)
            outer_loss.backward()

            acc = accuracy(logits, query_targets)
            batch_accuracies.append(acc)
            batch_losses.append(outer_loss.item())

        if i % 50 == 0:
            print('@ {} | {:.2f} {:.4f}'.format(i, mean(batch_accuracies), mean(batch_losses)))

        optimizer.step()


def do_eval(model, train_dl):
    adaptation_step = 5
    step_size = 0.1
    # model.linear.bias.register_hook(investigate)
    model.eval()
    accuracies = []

    mean = lambda x: sum(x) / len(x)

    for i, batch in enumerate(train_dl):
        for task_id, (support, query, support_targets, query_targets) in enumerate(batch):
            support = support.to(device)  # B,C,W,H
            query = query.to(device)
            support_targets = support_targets.to(device)
            query_targets = query_targets.to(device)

            clone = clone_module(model)
            clone = adapt(clone, support, support_targets,
                          adaptation_steps=adaptation_step,
                          step_size=step_size)

            logits = clone(query)
            acc = accuracy(logits, query_targets)
            accuracies.append(acc)

    return mean(accuracies)


def collate_fn(items):
    return items


def train():
    train_dl = DataLoader(FSLDataset('train', iteration=600), batch_size=B, num_workers=4, collate_fn=collate_fn)
    val_dl = DataLoader(FSLDataset('val', iteration=200), batch_size=B, num_workers=4, collate_fn=collate_fn)
    test_dl = DataLoader(FSLDataset('test', iteration=200), batch_size=B, num_workers=4, collate_fn=collate_fn)

    model = CNNModel(n_class=N).to(device)
    for epoch in range(100):
        do_train(model, train_dl)
        val_acc = do_eval(model, val_dl)
        test_acc = do_eval(model, test_dl)
        print('@ {} Val={:.2f} Test={:.2f}'.format(epoch, val_acc, test_acc))


if __name__ == '__main__':
    train()
