import os
import data
import argparse
import numpy as np
import solvers
import plotter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['rotated_rings', 'two_moons', 'random_gaussians'])
    parser.add_argument('--logdir', type=str, default='results')
    parser.add_argument('--ano-frac', type=float, default=0.0)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--solver', type=str, default='OT', choices=['OT', 'ROT'])
    parser.add_argument('--experiment', type=str, default='oneshot', choices=['oneshot', 'sensitivity'])
    parser.add_argument('--ground-cost', type=str, default='l2', choices=['l1', 'l2'])
    return parser.parse_args()


def main(args):
    nsamples = 100
    ndim = 2
    if args.dataset == 'rotated_rings':
        nmodes = 4
        rotation = (2 * np.pi / nmodes) * 0.2
        dist1, dist2 = data.rotated_rings_2D(nmodes, nsamples, rotation, rad=10)
    elif args.dataset == 'two_moons':
        dist1, dist2 = data.two_moons(nsamples)
    elif args.dataset == 'random_gaussians':
        ndim = 10
        dist1, dist2 = data.random_gaussians(ndim, nsamples)

    dist1_uncorrupted, dist2_uncorrupted = dist1, dist2

    if args.ano_frac > 0:
        ano_dist1 = data.anomaly_gaussian(ndim, rad=100, nsamples=int(nsamples * args.ano_frac), seed=0)
        ano_dist2 = data.anomaly_gaussian(ndim, rad=100, nsamples=int(nsamples * args.ano_frac), seed=2)
        dist1 = np.concatenate((dist1, ano_dist1), axis=0)
        dist2 = np.concatenate((dist2, ano_dist2), axis=0)

    if args.experiment == 'oneshot':
        print('Computing OT')
        if args.solver == 'OT':
            solver = solvers.OTSolver(dist1, dist2, ground_cost=args.ground_cost, logdir=args.logdir)
            OT_dist = solver.solve()
        elif args.solver == 'ROT':
            solver = solvers.ROTSolver(dist1, dist2, ground_cost=args.ground_cost, logdir=args.logdir, rho=args.rho)
            OT_dist = solver.solve()

        fh = open('{}/log.txt'.format(args.logdir), 'w+')
        fh.write('Optimal transport cost: {}'.format(OT_dist))
    else:
        Path(args.logdir).mkdir(parents=True, exist_ok=True)
        print('Running sensitivity analysis')
        rho_sweep = [i * 0.02 for i in range(10)]

        solver = solvers.OTSolver(dist1_uncorrupted, dist2_uncorrupted, ground_cost=args.ground_cost)
        OT_dist_uncorr = solver.solve(plot=False)

        OT_sensitivity = []
        for rho_val in rho_sweep:
            solver = solvers.ROTSolver(dist1, dist2, ground_cost=args.ground_cost, rho=rho_val)
            OT_dist = solver.solve(plot=False)
            OT_sensitivity.append((rho_val, OT_dist))

        plotter.generate_OT_sensitivity_plot(OT_dist_uncorr, OT_sensitivity, '{}/sensitivity.png'.format(args.logdir))

if __name__ == '__main__':
    args = parse_args()
    main(args)