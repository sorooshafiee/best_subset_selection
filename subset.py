import os
import numpy as np
from models import print_output, plot_with_shade, MISOCP, MIQP_newton, box_plot


# Constants
n_samples = 1000
n_features = 100
n_count = 10
n_model = 3             # MISOCP with 3D splitted cones, MIQP, MIQP to MISOCP
rho = 0.35              # Correlation parameter
nu = 2                  # Signal to noise ratio
all_k = np.array([i * 10 for i in range(1, 10)])
method = 'BIC'
colors = ['#672870', '#6c939f', '#fddbaa']
my_colors = ['#1f77b4', '#d62728', '#2ca02c']
labels = ['MISOCP', 'MIQP + FP', 'MISOCP + FP']

# Save and load data parameters
cwd = os.getcwd()
DIR_SAVE = os.path.join(cwd, 'saved')
DIR_FIG = os.path.join(cwd, 'plots')
results = os.path.join(DIR_SAVE, method + '_result.npz')
load_data = True

if load_data and os.path.isfile(results):
    saved = np.load(results)
    all_t = saved['all_t']
    all_gap = saved['all_gap']
    all_cut = saved['all_cut']
    all_nodes = saved['all_nodes']
else:
    # Parameters
    if method == 'AIC':
        f = lambda i: np.exp(- 2 * i / n_features)
    elif method == 'BIC':
        f = lambda i: np.exp(- np.log(n_features) * i / n_features)
    else:
        f = lambda i: n_features - i
    bigM = 1
    TimeLimit = 360
    verbose = False

    # Start the loop
    all_t = np.zeros([n_model, len(all_k), n_count])
    all_gap = np.zeros([n_model, len(all_k), n_count])
    all_cut = np.zeros([n_model, len(all_k), n_count])
    all_nodes = np.zeros([n_model, len(all_k), n_count])
    print(10 * '*', method, 10 * '*')

    for ind in range(n_count):
        mean_x = np.zeros(n_features)
        cov_x = np.zeros([n_features, n_features])
        for i in range(n_features):
            for j in range(n_features):
                cov_x[i,j] = rho ** np.abs(i-j)
        # Generate data
        X = np.random.multivariate_normal(mean_x, cov_x, n_samples)
        for ind_k, k in enumerate(all_k):
            beta = np.zeros(n_features)
            beta[0:k] = 1
            s2 = beta @ cov_x @ beta
            mean_y = X @ beta
            cov_y = np.diag(s2 / nu * np.ones(n_samples))
            y = np.random.multivariate_normal(mean_y, cov_y)
            
            # MISOCP with splitted 3D cones
            try:
                m1 = MISOCP(X, y, f, bigM, TimeLimit, verbose)
                msg = '{:d}, MISOCP formulation, {:d},'.format(ind+1, k)
                all_t[0, ind_k, ind], all_gap[0, ind_k, ind], \
                all_cut[0, ind_k, ind], all_nodes[0, ind_k, ind] = print_output(m1, msg)
            except:
                print('MISOCP formulation has failed')
            
            # MIQP
            try:
                m2 = MIQP_newton(X, y, f, bigM, TimeLimit, verbose)
                msg = '{:d}, MIQP formulation, {:d},'.format(ind+1, k)
                all_t[1, ind_k, ind], all_gap[1, ind_k, ind], \
                all_cut[1, ind_k, ind], all_nodes[1, ind_k, ind] = print_output(m2, msg)
            except:
                print('MIQP formulation has failed')

            # MIQP to MISOCP
            try:
                m3 = MIQP_newton(X, y, f, bigM, TimeLimit, verbose, 1)
                msg = '{:d}, MIQP to MISOCP formulation, {:d},'.format(ind+1, k)
                all_t[2, ind_k, ind], all_gap[2, ind_k, ind], \
                all_cut[2, ind_k, ind], all_nodes[2, ind_k, ind] = print_output(m3, msg)
            except:
                print('MIQP to MISOCP formulation has failed')
    np.savez(results, \
        all_t = all_t, all_cut = all_cut,  all_gap = all_gap, all_nodes = all_nodes)

plot_with_shade(all_k, all_t, r'$\| \beta_0 \|_0$', 'Execution time (s)', my_colors, \
                labels, os.path.join(DIR_FIG, method + '_time.pdf'))

plot_with_shade(all_k, all_gap, r'$\| \beta_0 \|_0$', 'Gap (\%)', my_colors, \
                labels, os.path.join(DIR_FIG, method + '_gap.pdf'))

plot_with_shade(all_k, all_cut, r'$\| \beta_0 \|_0$', '\# of cuts', my_colors, \
                labels, os.path.join(DIR_FIG, method + '_cut.pdf'))

plot_with_shade(all_k, all_nodes, r'$\| \beta_0 \|_0$', '\# of nodes', my_colors, \
                labels, os.path.join(DIR_FIG, method + '_node.pdf'))