# File containing modules for generating plots
import matplotlib.pyplot as plt
import numpy as np


def generate_scatter_plots(dist1, dist2, path):
    plt.figure()
    plt.scatter(dist1[:, 0], dist1[:, 1], c='r')
    plt.scatter(dist2[:, 0], dist2[:, 1], c='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(path)
    plt.close()


def generate_scatter_plots_with_coupling(dist1, dist2, P_mat, path):
    plt.figure()
    I, J = np.nonzero(P_mat > 1e-5)
    maxval = P_mat.max()
    for k in range(len(I)):
        lwidth = P_mat[I[k]][J[k]] / maxval
        plt.plot(np.hstack((dist1[I[k], 0], dist2[J[k], 0])),
                 np.hstack((dist1[I[k], 1], dist2[J[k], 1])), 'k', lw=lwidth)

    plt.scatter(dist1[:, 0], dist1[:, 1], c='r')
    plt.scatter(dist2[:, 0], dist2[:, 1], c='b')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(path)
    plt.close()


def generate_OT_sensitivity_plot(true_OT, OT_array, fpath):
    plt.figure()

    x_arr = [OT_array[i][0] for i in range(len(OT_array))]
    y_arr_true = [true_OT for i in range(len(OT_array))]
    y_arr = [OT_array[i][1] for i in range(len(OT_array))]

    plt.plot(x_arr, y_arr_true, 'g+-')
    plt.plot(x_arr, y_arr, 'b+-')

    plt.legend(['True OT', 'Robust OT'])
    plt.xlabel(r'$\rho')
    plt.ylabel('OT distance')
    plt.savefig(fpath, dpi=200)