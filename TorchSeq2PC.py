"""
*    Title: On the relationship between predictive coding and backpropagation source code
*    Author: Robert Rosenbaum
*    Date: 2021
*    Code version: 1.0
*    Availability: https://github.com/RobertRosenbaum/PredictiveCodingVsBackProp
"""

import torch


# Perform a forward pass on a Sequential model
# where X,Y are one batch of inputs,labels
# Returns activations for all layers (vhat), loss, and gradient of loss
# wrt last-layer activations (dldy)
# vhat,Loss,dldy=fwd_pass_plus(model,criterion,X,Y)
def fwd_pass_plus(model, criterion, X, Y):
    # Number of layers, counting the input as layer 0
    depth_plus_one = len(model) + 1

    # Forward pass
    vhat = [None] * depth_plus_one
    vhat[0] = X
    for layer in range(1, depth_plus_one):
        f = model[layer - 1]
        vhat[layer] = f(vhat[layer - 1])
    loss = criterion(vhat[-1], Y)

    # Compute gradient of loss with respect to output
    dldy = torch.autograd.grad(loss, vhat[-1])[0]
    return vhat, loss, dldy


# Compute prediction errors (epsilon) and beliefs (v)
# using predictive coding algorithm modified by
# the fixed prediction assumption
# see: Millidge, Tschantz, and Buckley. Predictive coding approximates backprop along arbitrary computation graphs.
# v,epsilon=fixed_pred_pc_pred_errs(model,vhat,dldy,eta=1,n=None)
def fixed_pred_pc_pred_errs(model, vhat, dldy, eta=1, n=None):
    # Number of layers, counting the input as layer 0
    depth_plus_one = len(model) + 1

    if n == None:
        n = len(model)

    # Initialize epsilons
    epsilon = [None] * depth_plus_one
    epsilon[-1] = dldy

    # Initialize v to a copy of vhat with no gradients needed
    # (can this be moved up to the loop above?)
    v = [None] * depth_plus_one
    for layer in range(depth_plus_one):
        v[layer] = vhat[layer].clone().detach()

    # Iterative updates of v and epsilon using stored values of vhat    
    for i in range(n):
        for layer in reversed(range(depth_plus_one - 1)):  # range(depth_plus_one-2,-1,-1)
            epsilon[layer] = vhat[layer] - v[layer]
            _, epsdfdv = torch.autograd.functional.vjp(model[layer], vhat[layer], epsilon[layer + 1])
            dv = epsilon[layer] - epsdfdv
            v[layer] = v[layer] + eta * dv
        # This helps free up memory
        with torch.no_grad():
            for layer in range(1, depth_plus_one - 1):
                v[layer] = v[layer].clone()
                epsilon[layer] = epsilon[layer].clone()
    return v, epsilon


# Compute prediction errors (epsilon) and beliefs (v)
# using a strict interpretation of predictive coding
# without the fixed prediction assumption.
# v,epsilon=strict_pc_pred_errs(model,vinit,criterion,Y,eta,n)
def strict_pc_pred_errs(model, vinit, criterion, Y, eta, n):
    with torch.no_grad():
        # Number of layers, counting the input as layer 0
        depth_plus_one = len(model) + 1

        # Initialize epsilons
        epsilon = [None] * depth_plus_one

    # Initialize v to a copy of vinit with no gradients needed
    # (can this be moved up to the loop above?)
    v = [None] * depth_plus_one
    for layer in range(depth_plus_one):
        v[layer] = vinit[layer].clone()

        # Iterative updates of v and epsilon
    for i in range(n):
        model.zero_grad()
        layer = depth_plus_one - 1
        vtilde = model[layer - 1](v[layer - 1])
        Loss = criterion(vtilde, Y)
        epsilon[layer] = torch.autograd.grad(Loss, vtilde, retain_graph=False)[0]  # -2 ~ depth_plus_one-2
        for layer in reversed(range(1, depth_plus_one - 1)):
            epsilon[layer] = v[layer] - model[layer - 1](v[layer - 1])
            _, epsdfdv = torch.autograd.functional.vjp(model[layer], v[layer], epsilon[layer + 1])
            dv = -epsilon[layer] + epsdfdv
            v[layer] = v[layer] + eta * dv
        # This helps free up memory
        with torch.no_grad():
            for layer in range(1, depth_plus_one - 1):
                v[layer] = v[layer].clone()
                epsilon[layer] = epsilon[layer].clone()

    return v, epsilon


# Compute exact prediction errors (epsilon) and beliefs (v)
# epsilon is defined as the gradient of the loss wrt to
# the activations and v=vhat-epsilon where vhat are the
# activations from a forward pass.
# v,epsilon=exact_pred_errs(model,criterion,X,Y,vhat=None)
def exact_pred_errs(model, criterion, X, Y, vhat=None):
    # Number of layers, counting the input as layer 0
    depth_plus_one = len(model) + 1

    # Forward pass if it wasn't passed in
    if vhat == None:
        vhat = [None] * depth_plus_one
        vhat[0] = X
        for layer in range(1, depth_plus_one):
            f = model[layer - 1]
            vhat[layer] = f(vhat[layer - 1])

    Loss = criterion(vhat[-1], Y)

    epsilon = [None] * depth_plus_one
    v = [None] * depth_plus_one

    for layer in range(1, depth_plus_one):
        epsilon[layer] = torch.autograd.grad(Loss, vhat[layer], allow_unused=True, retain_graph=True)[0]
        v[layer] = vhat[layer] - epsilon[layer]

    return v, epsilon


# Set gradients of model params based on PC approximations
def set_pc_grads(model, epsilon, X, v=None):
    # Number of layers, counting the input as layer 0
    depth_plus_one = len(model) + 1

    # Forward pass if v wasn't passed in
    if v == None:
        v = [None] * depth_plus_one
        v[0] = X
        for layer in range(1, depth_plus_one):
            f = model[layer - 1]
            v[layer] = f(v[layer - 1])

    # Compute new parameter values    
    for layer in range(0, depth_plus_one - 1):
        with torch.no_grad():
            vtemp0 = v[layer].clone()
            vtemp0.requires_grad = True
        vtemp1 = model[layer](vtemp0)
        for p in model[layer].parameters():
            dtheta = torch.autograd.grad(vtemp1, p, grad_outputs=epsilon[layer + 1], allow_unused=True, retain_graph=True)[0]
            p.grad = dtheta

        # Perform a whole PC inference step


# Returns activations (vhat), loss, gradient of the loss wrt output (dldy),
# beliefs (v), and prediction errors (epsilon)
# vhat,Loss,dldy,v,epsilon=pc_infer(model,criterion,X,Y,err_type="fixed_pred",eta=.1,n=20,vinit=None)
def pc_infer(model, criterion, X, Y, err_type, eta=.1, n=20, vinit=None):
    # Fwd pass (plus return vhat and dldy)
    vhat, Loss, dldy = fwd_pass_plus(model, criterion, X, Y)

    # Get beliefs and prediction errors
    if err_type == "fixed_pred":
        v, epsilon = fixed_pred_pc_pred_errs(model, vhat, dldy, eta, n)
        set_pc_grads(model, epsilon, X, vhat)
    elif err_type == "strict":
        if vinit == None:
            vinit = vhat
        v, epsilon = strict_pc_pred_errs(model, vhat, criterion, Y, eta, n)
        set_pc_grads(model, epsilon, X, v)
    elif err_type == "exact":
        v, epsilon = exact_pred_errs(model, criterion, X, Y)
        set_pc_grads(model, epsilon, X, vhat)
    else:
        raise ValueError('err_type must be \"fixed_pred\", \"strict\", or \"exact\"')

    # Set gradients in model
    # set_pc_grads(model,epsilon,X,vhat)

    return vhat, Loss, dldy, v, epsilon
