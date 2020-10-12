# Library of data generators
import numpy as np
import math
import sklearn


def gaussian(nsamples, mean, cov):
    samples = np.random.multivariate_normal(mean, cov, size=(nsamples,))
    return samples


def mixture_of_gaussian(nsamples, means, covs, mode_props):
    assert len(means) == len(covs) and len(means) == len(mode_props)
    nmodes = len(mode_props)
    samples = []

    for i in range(nmodes):
        nsamples_i = math.ceil(mode_props[i] * nsamples)
        samples_i = np.random.multivariate_normal(means[i], covs[i], size=(nsamples_i,))
        samples.append(samples_i)
    samples = np.vstack(samples)
    samples = samples[0:nsamples, ::]
    return samples


def rotated_rings_2D(nmodes, nsamples, rotation, rad=1):
    ndim = 2
    covs = [np.eye(ndim) for i in range(nmodes)]
    means_1 = []
    means_2 = []

    for i in range(nmodes):
        angle = (2 * np.pi * i) / nmodes
        angle_shifted = angle + rotation
        m1 = [rad * np.cos(angle), rad * np.sin(angle)]
        m2 = [rad * np.cos(angle_shifted), rad * np.sin(angle_shifted)]
        means_1.append(m1)
        means_2.append(m2)

    dist1 = mixture_of_gaussian(nsamples, means_1, covs, [1.0 / nmodes] * nmodes)
    dist2 = mixture_of_gaussian(nsamples, means_2, covs, [1.0 / nmodes] * nmodes)
    return dist1, dist2


def two_moons(nsamples):
    data_all, y_all = sklearn.datasets.make_moons(2*nsamples)
    dist1 = data_all[y_all == 0, ::]
    dist2 = data_all[y_all == 1, ::]
    return dist1, dist2


def random_gaussians(ndim, nsamples):
    rad = np.random.rand(1) * 10
    mean1 = np.random.rand(ndim) * rad
    mean2 = np.random.rand(ndim) * rad
    cov1 = np.random.rand(ndim, ndim)
    cov1 = np.matmul(cov1, cov1.T)
    cov2 = np.random.rand(ndim, ndim)
    cov2 = np.matmul(cov2, cov2.T)

    dist1 = gaussian(nsamples, mean1, cov1)
    dist2 = gaussian(nsamples, mean2, cov2)
    return dist1, dist2


def anomaly_gaussian(ndim, rad, nsamples, seed=0):
    # Anomaly gaussian centered at radius rad
    np.random.seed(seed)
    mean = np.random.randn(ndim)
    mean = mean / np.linalg.norm(mean, ord=2)
    mean = mean * rad
    cov = np.random.rand(ndim, ndim) * rad * 0.1
    cov = np.matmul(cov, cov.T)
    ano_dist = gaussian(nsamples, mean, cov)
    return ano_dist


def anomaly_sphere(ndim, rad, nsamples):
    # Anomalous samples on a sphere
    ano_dist = np.random.randn(nsamples, ndim)
    ano_dist = ano_dist / np.linalg.norm(ano_dist, ord=2, axis=1)
    ano_dist = ano_dist * rad
    return ano_dist