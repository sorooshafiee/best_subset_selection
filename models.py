from dataclasses import asdict
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import rc
import gurobipy as gp
from gurobipy import GRB
from time import time


rc('font', family='serif')
rc('text', usetex=True)
rc('xtick',labelsize=14)
rc('ytick',labelsize=14)
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', \
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
my_color = ['#1f77b4', '#d62728', '#2ca02c']
colors_light = ['#672870', '#6c939f', '#fddbaa']
line_style = ['solid', 'dashed', 'dashdot']
labels = [r'MISOCP', r'MIQP', r'MIQP $\to$ MISOCP']
labels_2 = ['Lazy', 'Basic']


def box_plot(x, y, xlabel, ylabel, color, label, fname):
    fig, ax = plt.subplots(1)
    for i in range(y.shape[0]):
        ax.boxplot(y[i].T, positions=x+i, patch_artist=True,
            boxprops=dict(facecolor=color[i]),
            capprops=dict(color=color[i]),
            whiskerprops=dict(color=color[i]),
            flierprops=dict(color=color[i], markeredgecolor=color[i]),
            medianprops=dict(color='#000000'),
            )
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_yscale('log')
    ax.legend(label, loc='upper left', fontsize=16)
    ax.set_xticks(x+1)
    fig.savefig(fname, format='pdf', dpi=300)
    return fig, ax


def plot_with_shade(x, y, xlabel, ylabel, color, label, fname):
    y_mean = np.mean(y, axis=2)
    y_max = np.max(y, axis=2)
    y_min = np.min(y, axis=2)
    fig, ax = plt.subplots(1)
    for i in range(y_mean.shape[0]):
        ax.plot(x, y_mean[i], lw=2.5, c=color[i])
        ax.fill_between(x, y_max[i], y_min[i], fc=color[i], alpha=0.05)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_yscale('log')
    ax.set_xticks(x + 1)
    ax.legend(label, loc='best', fontsize=16)
    fig.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')
    return fig, ax


def compute_bigM_knapsack(zeta, b):
    tmp = [np.max(np.sum(zeta[j], axis=1)) - b for j in range(zeta.shape[0])]
    return np.maximum(b, np.array(tmp))
        


def scale_data(X, y):
    scale = np.max([np.max(np.abs(X)), np.max(np.abs(y))])
    return X / scale, y / scale


def print_output(model, msg):
    print(msg, '{:.1f}, {:d}, {:.1f}({:d}), {:d}, {:d}, {:d}, {}'.format(
        model._t, model._user_cut + model._lazy_cut, model._gap[-1], len(model._gap), \
            model._node_count, model._lazy_cut, model._user_cut, model._gap))
    return model._t, model._gap[-1], model._user_cut + model._lazy_cut, model._node_count 


def lazy_polymatroid(model, where):
    if where == GRB.Callback.MIPSOL:
        val_z = model.cbGetSolution(model._var_z)
        ind = np.argsort(-val_z)            
        model._lazy_cut += 1
        model.cbLazy(
            sum(model._var_s) <= model._f0 + sum(model._pi[i] * model._var_z[ind][i] 
            for i in range(len(val_z))))
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            val_z = model.cbGetNodeRel(model._var_z)
            ind = np.argsort(-val_z)
            model._user_cut += 1
            model.cbCut(
                sum(model._var_s) <= model._f0 + sum(model._pi[i] * model._var_z[ind][i] 
                for i in range(len(val_z))))


def MISOCP(
    X, y, f, bigM, \
        TimeLimit=3600, verbose=False, split=1, PreMIQCPForm=-1, MIQCPMethod=0, lazy=1):
    
    t_0 = time()
    X, y = scale_data(X, y)         # This is added for numerical stability

    # Constants
    inf = GRB.INFINITY
    n_samples, n_features = X.shape
    vals = np.array([i for i in range(n_features + 1)])
    f_vals = np.array([f(i) for i in range(n_features + 1)])
    fmax = max(f_vals)
    f0 = f(0)
    pi = np.array([f(i + 1) - f(i) for i in range(n_features)])

    # Create the optimization model
    model = gp.Model()
    
    # Define variables
    var_beta = model.addMVar(n_features, lb=-inf, ub=inf, vtype=GRB.CONTINUOUS)
    var_r = model.addMVar(n_samples, lb=-inf, ub=inf, vtype=GRB.CONTINUOUS)
    var_s = model.addMVar(1, lb=0, ub=fmax, vtype=GRB.CONTINUOUS)
    var_t = model.addMVar(1, lb=0, ub=inf, vtype=GRB.CONTINUOUS)
    var_z = model.addMVar(n_features, vtype=GRB.BINARY)
    if split:
        var_aux = model.addMVar(n_samples, lb=0, ub=inf, vtype=GRB.CONTINUOUS)
    if not lazy:
        var_w = model.addMVar(n_features + 1, vtype=GRB.BINARY)
    
    # Add objective function
    model.setObjective(var_t)

    # Add constraints
    model.addConstr(var_r == y - X @ var_beta)
    
    if split:
        for i in range(n_samples):
            model.addQConstr(var_r[i] * var_r[i] <= var_aux[i] * var_s[0])
        model.addConstr(sum(var_aux) <= var_t[0])
    else:
        model.addQConstr(var_r @ var_r <= var_s @ var_t)
    
    if bigM:
        # Big-M formulation
        model.addConstr(var_beta <= bigM * var_z)
        model.addConstr(var_beta >= -bigM * var_z)
    else:
        # Use conditional statements
        model.addConstrs((var_z[i] == 0) >> (var_beta[i] == 0) 
                         for i in range(n_features))
    
    if not lazy:
        model.addConstr(var_s <= f_vals @ var_w)
        model.addConstr(vals @ var_w == np.ones(n_features) @ var_z)
        model.addConstr(np.ones(n_features + 1) @ var_w == 1)

    # Model parameters
    model.Params.OutputFlag = verbose
    model.Params.lazyConstraints = lazy
    model.Params.MIQCPMethod = MIQCPMethod
    model.Params.PreMIQCPForm = PreMIQCPForm
    model.Params.TimeLimit = TimeLimit
    model._var_z = var_z
    model._var_s = var_s
    model._var_beta = var_beta
    model._f0 = f0
    model._pi = pi
    model._lazy_cut = 0
    model._user_cut = 0

    # Optimize the problem
    if f0 == n_features:      # MSE
        model.addConstr(var_s == pi @ var_z)
        model.optimize()
    else:
        model.optimize(lazy_polymatroid)
           
    # Add new attributes
    model._gap = np.array([model.MIPGap * 100])
    model._node_count = int(model.NodeCount)
    model._t = time() - t_0

    # Return model
    return model


def MIQP_newton(
    X, y, f, bigM, \
        TimeLimit=3600, verbose=False, socp=0, PreMIQCPForm=-1, MIQCPMethod=0, lazy=1):
    
    t_0 = time()
    X, y = scale_data(X, y)         # This is added for numerical stability

    # Constants
    inf = GRB.INFINITY
    n_samples, n_features = X.shape
    vals = np.array([i for i in range(n_features + 1)])
    f_vals = np.array([f(i) for i in range(n_features + 1)])
    fmax = max(f_vals)
    fp = f(n_features) if f(n_features) != 0 else f(n_features-1)
    f0 = f(0)
    pi = np.array([f(i + 1) - f(i) for i in range(n_features)])
    h = lambda beta, z: la.norm(y - X @ beta, 2)**2 / f(sum(z))
    d_hat = lambda beta, z, t: la.norm(y - X @ beta, 2)**2 - t * f(sum(z))
    beta = np.zeros(n_features)
    z = np.zeros(n_features)
    val_tau = h(beta, z)

    # Define the optimization model
    model = gp.Model()

    # Define variables
    var_beta = model.addMVar(n_features, lb=-inf, ub=inf, vtype=GRB.CONTINUOUS)
    var_r = model.addMVar(n_samples, lb=-inf, ub=inf, vtype=GRB.CONTINUOUS)
    var_s = model.addMVar(1, lb=0, ub=fmax, vtype=GRB.CONTINUOUS)
    var_t = model.addMVar(1, lb=0, ub=inf, vtype=GRB.CONTINUOUS)
    var_z = model.addMVar(n_features, vtype=GRB.BINARY)
    if socp:
        var_aux = model.addMVar(1, lb=0, ub=inf, vtype=GRB.CONTINUOUS)
    if not lazy:
        var_w = model.addMVar(n_features + 1, vtype=GRB.BINARY)

    # Add constraints
    model.addConstr(var_r == y - X @ var_beta)
    
    if socp:
        model.addQConstr(var_r @ var_r <= var_t @ var_aux)
        model.addConstr(var_aux == 1)
    else:
        model.addQConstr(var_r @ var_r <= var_t)
    
    if bigM:
        # Big-M formulation
        model.addConstr(var_beta <= bigM * var_z)
        model.addConstr(var_beta >= -bigM * var_z)
    else:
        # Use conditional statements
        model.addConstrs((var_z[i] == 0) >> (var_beta[i] == 0) for i in range(n_features))
    
    if not lazy:
        model.addConstr(var_s <= f_vals @ var_w)
        model.addConstr(vals @ var_w == np.ones(n_features) @ var_z)
        model.addConstr(np.ones(n_features + 1) @ var_w == 1)

    # Model parameters
    model.Params.OutputFlag = verbose
    model.Params.lazyConstraints = lazy
    model.Params.MIQCPMethod = MIQCPMethod
    model.Params.PreMIQCPForm = PreMIQCPForm
    model.Params.TimeLimit = TimeLimit
    model._var_z = var_z
    model._var_s = var_s
    model._var_beta = var_beta
    model._f0 = f0
    model._pi = pi
    model._lazy_cut = 0
    model._user_cut = 0

    gap = []
    model._node_count = 0
    t_1 = time()
    for i in range(n_features+1):
        # Set the time limit
        try:
            model.Params.TimeLimit = TimeLimit - (time() - t_1)
        except:
            break

        # Add objective function
        model.setObjective(var_t - val_tau * var_s)
        
        # Optimize the problem
        model.optimize(lazy_polymatroid)

        # Node count
        model._node_count += int(model.NodeCount)
        # Optimality gap
        gap.append(abs((100 * model.ObjBound) / (val_tau * fp)))
        
        # Update val_tau
        if d_hat(var_beta.x, var_z.x, val_tau) < 0 and gap[-1] > 1e-6:
            val_tau = h(var_beta.x, var_z.x)
        else:
            break

    # Add new attributes
    model._gap = np.array(gap)
    model._t = time() - t_0

    # Return optimal values
    return model

