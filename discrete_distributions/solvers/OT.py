import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import plotter
from pathlib import Path
import os


class OTSolver(object):
    def __init__(self, dist1, dist2, marginal1=None, marginal2=None, ground_cost='l2', logdir='results'):
        self.dist1 = dist1
        self.dist2 = dist2
        nsamples1 = dist1.shape[0]
        nsamples2 = dist2.shape[0]
        self.nsamples1 = nsamples1
        self.nsamples2 = nsamples2

        if marginal1 is None:
            self.marginal1 = np.array([1/nsamples1 for i in range(nsamples1)])
        else:
            self.marginal1 = marginal1

        if marginal2 is None:
            self.marginal2 = np.array([1/nsamples2 for i in range(nsamples2)])
        else:
            self.marginal2 = marginal2
        self.marginal1 = np.expand_dims(self.marginal1, axis=1)
        self.marginal2 = np.expand_dims(self.marginal2, axis=1)

        self.ground_cost = ground_cost
        assert ground_cost in ['l2']
        self.logdir = logdir
        Path(self.logdir).mkdir(parents=True, exist_ok=True)

    def form_cost_matrix(self, x, y):
        if self.ground_cost == 'l2':
            return np.sum(x ** 2, 1)[:, None] + np.sum(y ** 2, 1)[None, :] - 2 * x.dot(y.transpose())

    def solve(self, plot=True):
        C = self.form_cost_matrix(self.dist1, self.dist2)
        P = cp.Variable((self.nsamples1, self.nsamples2))

        u = np.ones((self.nsamples2, 1))
        v = np.ones((self.nsamples1, 1))
        constraints = [0 <= P, cp.matmul(P, u) == self.marginal1, cp.matmul(P.T, v) == self.marginal2]

        objective = cp.Minimize(cp.sum(cp.multiply(P, C)))
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        coupling = P.value

        print("Number of non-zero values in P: {} (n + m-1 = %d)".format(len(coupling[coupling > 1e-5]),
                                                                         self.nsamples1 + self.nsamples2 - 1))
        print("Objective function: {}".format(objective.value))

        if plot:
            print('Generating plots ...')
            plotter.generate_scatter_plots(self.dist1, self.dist2,
                                           '{}/orig.png'.format(self.logdir))
            plotter.generate_scatter_plots_with_coupling(self.dist1, self.dist2, coupling,
                                                         '{}/coupling.png'.format(self.logdir))

        OT_cost = objective.value
        return OT_cost






