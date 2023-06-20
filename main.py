"""
*    Title: On the relationship between predictive coding and backpropagation source code
*    Author: Robert Rosenbaum
*    Date: 2021
*    Code version: 1.0
*    Availability: https://github.com/RobertRosenbaum/PredictiveCodingVsBackProp
"""

import torch
import torch.nn as nn
from copy import deepcopy
import argparse
from model import build_original_model, build_modified_model
from dataset import load_dataset
from train import train_pred_coding, train_backprop
from utils import plot_metric, plot_param_error


def parse_args():
    """
    Parses the input arguments for the process of training models.
    :return: argparse.Namespace: The validated input arguments for training models.
    """
    desc = "Pytorch implementation of paper On the relationship between predictive coding and backpropagation."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--train_batch_size', type=int, default=300, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=300, help='Testing batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, default=4, help='Number of epoch')
    parser.add_argument('--eta', type=float, default=0.1, help='Step size')
    parser.add_argument('--num_iter', type=int, default=20, help='Number of iterations')
    parser.add_argument('--opt', type=str, default="adam", help='Choose optimizer')
    parser.add_argument('--compute_metric', default=True, help='Compute training metrics')
    parser.add_argument('--err_type', default="strict", choices=["strict", "fixed_pred"],
                        help='Choose algorithm strict predict or fixed predict')
    parser.add_argument('--model_type', default="original", choices=["original", "modified"],
                        help='Choose different models to do experiment')
    args = parser.parse_args()
    return args


def main():
    torch.manual_seed(84)
    # -- load arguments
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = load_dataset(args.train_batch_size, args.test_batch_size)
    total_num_steps = args.num_epoch * len(train_loader)

    if args.model_type == "original":
        model = build_original_model().to(device)
    else:
        model = build_modified_model().to(device)
    if args.err_type == "strict":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    model_pc = deepcopy(model)
    model_bp = deepcopy(model)
    test_loss_pc, test_acc_pc, grads_rel_diff0, grads_angle0 = train_pred_coding(model=model_pc, optim=args.opt,
                                                                                 lr=args.lr, total_num_steps=total_num_steps,
                                                                                 num_epoch=args.num_epoch,
                                                                                 device=device, train_loader=train_loader,test_loader=test_loader,
                                                                                 criterion=criterion, eta=args.eta, num_iter=args.num_iter,
                                                                                 test_batch_size=args.test_batch_size, err_type=args.err_type)

    test_loss_bp, test_acc_bp = train_backprop(model=model_bp, optim=args.opt, lr=args.lr, total_num_steps=total_num_steps,
                                               num_epoch=args.num_epoch, device=device, train_loader=train_loader, test_loader=test_loader,
                                               criterion=criterion, test_batch_size=args.test_batch_size,
                                               err_type=args.err_type)
    etas = [0.1, 0.2, 0.5, 1]
    plot_metric(model, test_loss_pc, test_loss_bp, test_acc_pc, test_acc_bp, grads_rel_diff0, grads_angle0)
    plot_param_error(model, train_loader, device, criterion, etas, args.err_type)
    if args.err_type == "strict":
        print("Predictive coding accuracy: ", test_acc_pc[-1])
    else:
        print("Predictive coding modified by the fixed prediction assumption accuracy: ", test_acc_pc[-1])
    print("Backpropagation accuracy: ", test_acc_bp[-1])


if __name__ == '__main__':
    main()
