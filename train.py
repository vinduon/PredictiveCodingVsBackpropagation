"""
*    Title: On the relationship between predictive coding and backpropagation source code
*    Author: Robert Rosenbaum
*    Date: 2021
*    Code version: 1.0
*    Availability: https://github.com/RobertRosenbaum/PredictiveCodingVsBackProp
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import TorchSeq2PC as T2PC
from utils import encode_one_hot, rel_diff


def train_pred_coding(model, optim, lr, total_num_steps, num_epoch, device,
                      train_loader, test_loader, criterion, eta, num_iter,
                      test_batch_size, err_type, compute_metric=True):
    if optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    LossesToPlotPC = np.zeros(total_num_steps)
    TestLossesToPlotPC = np.zeros(total_num_steps)
    AccuraciesToPlotPC = np.zeros(total_num_steps)
    TestAccuraciesToPlotPC = np.zeros(total_num_steps)

    GradsRelDiff0 = np.zeros([total_num_steps, len(model)])
    GradsCosSim0 = np.zeros([total_num_steps, len(model)])
    GradsAngle0 = np.zeros([total_num_steps, len(model)])

    CosSim = nn.CosineSimilarity(dim=0, eps=1e-8)
    steps_per_epoch = len(train_loader)
    jj = 0
    for k in range(num_epoch):
        # Re-initializes the training iterator (shuffles data for one epoch)
        training_iterator = iter(train_loader)

        for i in range(steps_per_epoch):  # For each batch

            # Get one batch of training data, reshape it
            # and send it to the current device
            X, Y = next(training_iterator)
            X = X.to(device)
            if err_type == "strict":
                Y = encode_one_hot(Y, 10).to(device)
            else:
                Y = Y.to(device)
            _, Loss, _, _, _ = T2PC.pc_infer(model, criterion, X, Y, err_type, eta, num_iter)

            if compute_metric:
                modelBP = deepcopy(model)  # Copy the model
                YhatBP = modelBP(X)  # Forward pass
                LossBP = criterion(YhatBP, Y)
                LossBP.backward()  # Compute gradients
                for layer in range(len(model)):
                    gradsPC = model[layer][0].weight.grad.cpu().detach().numpy()
                    gradsBP = modelBP[layer][0].weight.grad.cpu().detach().numpy()
                    GradsRelDiff0[jj, layer] = rel_diff(gradsPC, gradsBP)
                    GradsCosSim0[jj, layer] = CosSim(torch.tensor(gradsPC.flatten()),
                                                     torch.tensor(gradsBP.flatten())).item()
                    GradsAngle0[jj, layer] = torch.acos(torch.tensor(GradsCosSim0[jj, layer])).item()
                modelBP.zero_grad()

            # Update parameters
            opt.step()

            # Zero-out gradients
            model.zero_grad()
            opt.zero_grad()

            # Print loss, store loss, compute test loss
            with torch.no_grad():
                if i % 50 == 0:
                    print('k =', k, 'i =', i, 'loss =', Loss.item())
                LossesToPlotPC[jj] = Loss.item()

                if compute_metric:
                    Yhat = model(X)
                    if err_type == "strict":
                        AccuraciesToPlotPC[jj] = (torch.sum(
                            torch.argmax(Y, axis=1) == torch.argmax(Yhat, axis=1)) / test_batch_size).item()
                    else:
                        AccuraciesToPlotPC[jj] = (torch.sum(Y == torch.argmax(Yhat, axis=1)) / test_batch_size).item()
                    model.eval()
                    TestingIterator = iter(test_loader)
                    Xtest, Ytest = next(TestingIterator)
                    Xtest = Xtest.to(device)
                    if err_type == "strict":
                        Ytest = encode_one_hot(Ytest, 10).to(device)
                    else:
                        Ytest = Ytest.to(device)
                    YhatTest = model(Xtest)
                    TestLossesToPlotPC[jj] = criterion(YhatTest, Ytest).item()
                    if err_type == "strict":
                        TestAccuraciesToPlotPC[jj] = (torch.sum(
                            torch.argmax(Ytest, axis=1) == torch.argmax(YhatTest, axis=1)) / test_batch_size).item()
                    else:
                        TestAccuraciesToPlotPC[jj] = (
                                torch.sum(Ytest == torch.argmax(YhatTest, axis=1)) / test_batch_size).item()
                    model.train()
                jj += 1
    return TestLossesToPlotPC, TestAccuraciesToPlotPC, GradsRelDiff0, GradsAngle0


def train_backprop(model, optim, lr, total_num_steps, num_epoch, device,
                   train_loader, test_loader, criterion, test_batch_size, err_type, compute_metric=True):
    if optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    LossesToPlotBP = np.zeros(total_num_steps)
    TestLossesToPlotBP = np.zeros(total_num_steps)
    AccuraciesToPlotBP = np.zeros(total_num_steps)
    TestAccuraciesToPlotBP = np.zeros(total_num_steps)

    jj = 0
    steps_per_epoch = len(train_loader)
    for k in range(num_epoch):
        # Re-initializes the training iterator (shuffles data for one epoch)
        training_iterator = iter(train_loader)

        for i in range(steps_per_epoch):  # For each batch
            # Get one batch of training data, reshape it
            # and send it to the current device
            X, Y = next(training_iterator)
            X = X.to(device)
            if err_type == "strict":
                Y = encode_one_hot(Y, 10).to(device)
            else:
                Y = Y.to(device)
            Yhat = model(X)  # Forward pass
            Loss = criterion(Yhat, Y)
            Loss.backward()  # Compute gradients
            opt.step()  # Update parameters

            # Zero-out gradients
            opt.zero_grad()
            opt.zero_grad()
            with torch.no_grad():
                if i % 50 == 0:
                    print('k =', k, 'i =', i, 'loss =', Loss.item())
                LossesToPlotBP[jj] = Loss.item()
                if compute_metric:
                    if err_type == "strict":
                        AccuraciesToPlotBP[jj] = (torch.sum(
                            torch.argmax(Y, axis=1) == torch.argmax(Yhat, axis=1)) / test_batch_size).item()
                    else:
                        AccuraciesToPlotBP[jj] = (torch.sum(Y == torch.argmax(Yhat, axis=1)) / test_batch_size).item()
                    model.eval()
                    TestingIterator = iter(test_loader)
                    Xtest, Ytest = next(TestingIterator)
                    Xtest = Xtest.to(device)
                    if err_type == "strict":
                        Ytest = encode_one_hot(Ytest, 10).to(device)
                    else:
                        Ytest = Ytest.to(device)
                    YhatTest = model(Xtest)
                    TestLossesToPlotBP[jj] = criterion(YhatTest, Ytest).item()
                    if err_type == "strict":
                        TestAccuraciesToPlotBP[jj] = (torch.sum(
                            torch.argmax(Ytest, axis=1) == torch.argmax(YhatTest, axis=1)) / test_batch_size).item()
                    else:
                        TestAccuraciesToPlotBP[jj] = (
                                    torch.sum(Ytest == torch.argmax(YhatTest, axis=1)) / test_batch_size).item()
                    model.train()
                jj += 1
    return TestLossesToPlotBP, TestAccuraciesToPlotBP
