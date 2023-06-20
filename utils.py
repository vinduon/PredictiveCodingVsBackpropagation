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
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import TorchSeq2PC as T2PC


def encode_one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]


def rel_diff(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def plot_metric(model, TestLossesToPlotPC, TestLossesToPlotBP, TestAccuraciesToPlotPC, TestAccuraciesToPlotBP,
                GradsRelDiff0, GradsAngle0):
    sns.set(context='paper', style='white', font_scale=1.5)

    fig, axes = plt.subplots(figsize=(11, 6))

    plt.subplot(1, 2, 1)
    plt.plot(TestLossesToPlotPC, color=sns.color_palette('dark')[3], label='PC (test)')
    plt.plot(TestLossesToPlotBP, color=sns.color_palette('dark')[0], label='BP (test)')
    plt.xlabel('step number')
    plt.ylabel('loss')
    sns.despine()
    plt.ylim(bottom=0)
    plt.title('A', loc='left')

    plt.subplot(1, 2, 2)
    plt.plot(TestAccuraciesToPlotPC, color=sns.color_palette('dark')[3], label='PC (test)')
    plt.plot(TestAccuraciesToPlotBP, color=sns.color_palette('dark')[0], label='BP (test)')
    plt.xlabel('step number')
    plt.ylabel('accuracy')
    sns.despine()
    plt.legend(loc=(1.04, .25))
    plt.ylim(bottom=0)
    plt.title('B', loc='left')

    plt.tight_layout()
    plt.savefig("result.png")


def plot_param_error(model, train_loader, device, criterion, etas, err_type):
    if err_type == "strict":
        n = 400
    else:
        n = 200
    CosSim = nn.CosineSimilarity(dim=0, eps=1e-8)
    GradsAngle = np.zeros([len(etas), len(model), n])
    GradsCosSim = np.zeros([len(etas), len(model), n])
    GradsRelDiff = np.zeros([len(etas), len(model), n])
    ErrsRelDiff = np.zeros([len(etas), len(model), n])

    for jj in range(len(etas)):
        eta = etas[jj]
        if err_type == "strict":
            modelPC0 = deepcopy(model)
            modelBP0 = deepcopy(model)
        else:
            modelPC = deepcopy(model)
            modelBP0 = deepcopy(model)

        TrainingIterator = iter(train_loader)
        # Get one batch of training data, reshape it
        # and send it to the current device
        X, Y = next(TrainingIterator)
        X = X.to(device)
        if err_type == "strict":
            Y = encode_one_hot(Y, 10).to(device)
        else:
            Y = Y.to(device)

        # Compute BP gradients
        torch.manual_seed(0)
        vhatBP, LossBP, dLdyBP, vBP, epsilonBP = T2PC.pc_infer(modelBP0, criterion, X, Y, "exact")
        # Number of layers, counting the input as layer 0
        DepthPlusOne = len(model) + 1

        if err_type == "fixed_pred":
            modelPC.zero_grad()
            epsilon = [None] * DepthPlusOne
            epsilon[-1] = dLdyBP
            epsilon[-1] = dLdyBP
        else:
            # Initialize epsilons
            torch.manual_seed(0)
            epsilon = [None] * DepthPlusOne

        torch.manual_seed(0)
        if err_type == "strict":
            v = [None] * DepthPlusOne
            for layer in range(DepthPlusOne):
                v[layer] = vhatBP[layer].clone()
            vhat, Loss, dLdy = T2PC.fwd_pass_plus(modelPC0, criterion, X, Y)
            for i in range(n):
                modelPC0.zero_grad()
                layer = DepthPlusOne - 1
                vtilde = modelPC0[layer - 1](v[layer - 1])
                Loss = criterion(vtilde, Y)
                epsilon[layer] = torch.autograd.grad(Loss, vtilde, retain_graph=False)[0]
                for layer in reversed(range(1, DepthPlusOne - 1)):
                    epsilon[layer] = v[layer] - modelPC0[layer - 1](v[layer - 1])
                    _, epsdfdv = torch.autograd.functional.vjp(modelPC0[layer], v[layer], epsilon[layer + 1])
                    dv = -epsilon[layer] + epsdfdv
                    v[layer] = v[layer] + eta * dv

                # Compute new parameter values
                for layer in range(0, DepthPlusOne - 1):
                    with torch.no_grad():
                        vtemp0 = v[layer].clone()
                        vtemp0.requires_grad = True
                    vtemp1 = modelPC0[layer](vtemp0)
                    for p in modelPC0[layer].parameters():
                        dtheta = torch.autograd.grad(vtemp1, p, grad_outputs=epsilon[layer + 1], allow_unused=True,
                                                     retain_graph=True)[0]
                        p.grad = dtheta

                for layer in range(0, DepthPlusOne - 1):
                    gradsPC0 = modelPC0[layer][0].weight.grad.cpu().detach().numpy()
                    gradsBP = modelBP0[layer][0].weight.grad.cpu().detach().numpy()
                    GradsRelDiff[jj, layer, i] = rel_diff(gradsPC0, gradsBP)
                    ErrsRelDiff[jj, layer, i] = rel_diff(epsilon[layer + 1].cpu().detach().numpy(),
                                                        epsilonBP[layer + 1].cpu().detach().numpy())
                    GradsCosSim[jj, layer, i] = CosSim(torch.tensor(gradsPC0.flatten()),
                                                       torch.tensor(gradsBP.flatten())).item()
                    GradsAngle[jj, layer, i] = torch.acos(torch.tensor(GradsCosSim[jj, layer, i])).item()

        else:
            vhat = [None] * DepthPlusOne
            vhat[0] = X
            for layer in range(1, DepthPlusOne):
                f = modelPC[layer - 1]
                vhat[layer] = f(vhat[layer - 1])
            v = [None] * DepthPlusOne
            for layer in range(DepthPlusOne):
                v[layer] = vhat[layer].clone().detach()

                # Iterative updates of v and epsilon using stored values of vhat
            torch.manual_seed(0)
            for i in range(n):
                modelPC.zero_grad()
                for layer in reversed(range(DepthPlusOne - 1)):
                    epsilon[layer] = vhat[layer] - v[layer]
                    _, epsdfdv = torch.autograd.functional.vjp(modelPC[layer], vhat[layer], epsilon[layer + 1])
                    dv = epsilon[layer] - epsdfdv
                    v[layer] = v[layer] + eta * dv

                # Compute new parameter values
                for layer in range(0, DepthPlusOne - 1):
                    for p in modelPC[layer].parameters():
                        dtheta = \
                        torch.autograd.grad(vhat[layer + 1], p, grad_outputs=epsilon[layer + 1], allow_unused=True,
                                            retain_graph=True)[0]
                        p.grad = dtheta

                for layer in range(0, DepthPlusOne - 1):
                    gradsPC = modelPC[layer][0].weight.grad.cpu().detach().numpy()
                    gradsBP = modelBP0[layer][0].weight.grad.cpu().detach().numpy()
                    GradsRelDiff[jj, layer, i] = rel_diff(gradsPC, gradsBP)
                    ErrsRelDiff[jj, layer, i] = rel_diff(epsilon[layer + 1].cpu().detach().numpy(),
                                                        epsilonBP[layer + 1].cpu().detach().numpy())
                    GradsCosSim[jj, layer, i] = CosSim(torch.tensor(gradsPC.flatten()),
                                                       torch.tensor(gradsBP.flatten())).item()
                    GradsAngle[jj, layer, i] = torch.acos(torch.tensor(GradsCosSim[jj, layer, i])).item()

    sns.set(context='paper', style='white', font_scale=1.3)

    fig, axes = plt.subplots(figsize=(16, 6))

    for kk in range(len(etas)):
        plt.subplot(2, 4, kk + 1)
        for layer in range(0, DepthPlusOne - 1):
            plt.plot(np.arange(1, n + 1), GradsRelDiff[kk, layer, :].T, label='layer ' + str(layer + 1))
        plt.title('$\eta=$' + str(etas[kk]))
        plt.xscale('log')
        sns.despine()
        if err_type == "strict":
            if kk >= 2:
                plt.ylim(top=100, bottom=0)
            if kk < 2:
                plt.yticks([0.0, 0.5, 1.0])
            if kk == 0:
                plt.ylabel('relative error between\n' + r'd$\theta$ and gradient')
            if kk == 2:
                plt.legend(loc='best')
        else:
            if kk == 0:
                plt.ylabel('relative error between\n' + r'd$\theta$ and gradient')
            if kk == 2:
                plt.legend()
        plt.xticks([])

    for kk in range(len(etas)):
        plt.subplot(2, 4, 4 + kk + 1)
        for layer in range(0, DepthPlusOne - 1):
            plt.plot(np.arange(1, n + 1), (180 / 3.14159) * GradsAngle[kk, layer, :].T, label='layer ' + str(layer + 1))
        plt.xscale('log')
        plt.xlabel('number of iterations (n)')
        if err_type == "strict":
            if kk == 0:
                plt.ylabel(r'angle between d$\theta$ and' + '\n gradient (degrees)')
            sns.despine()
        else:
            sns.despine()
            if kk == 0:
                plt.ylabel(r'angle between d$\theta$ and' + '\n gradient (degrees)')
            if kk == 3:
                plt.xticks([1, 2, 3, 4, 5, 10, 100], ['$10^0$', '2', '3', '4', '5', '$10^1$', '$10^2$'])
    plt.tight_layout()
    plt.savefig("err_grad.png")
