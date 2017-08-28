import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gauss_kernel(size=5, sigma=1):
    grid = np.mgrid[:size, :size].T
    center = [size // 2] * 2
    kernel = np.exp(((grid - center) ** 2).sum(axis=2) / (-2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


_gauss_kernel_weights = {}


def conv_gauss_kernel(x, k_size=5, sigma=1, padding=0, stride=1, cuda=False):
    global _gauss_kernel_weights
    if (k_size, sigma, cuda) in _gauss_kernel_weights:
        weights = _gauss_kernel_weights[k_size, sigma, cuda]
    else:
        weights = torch.from_numpy(gauss_kernel(size=k_size, sigma=sigma))
        if cuda:
            weights = weights.cuda()
        weights = Variable(weights, requires_grad=False)
        _gauss_kernel_weights[k_size, sigma, cuda] = weights

    # TODO: same-padding instead of zero?
    n_channels = x.size()[1]
    weights = weights.expand(n_channels, 1, k_size, k_size)
    return F.conv2d(x, weights, stride=stride, padding=padding,
                    groups=n_channels)


def laplacian_pyramid(x, n_levels, k_size=5, sigma=2):
    pyr = []
    current = x
    for level in range(n_levels):
        gauss = conv_gauss_kernel(
            current, k_size=k_size, sigma=sigma, padding=k_size // 2)
        diff = current - gauss
        pyr.append(diff)
        current = F.avg_pool2d(gauss, 2)
    pyr.append(current)
    return pyr


def laplacian_loss(input, target, n_levels=3, k_size=5, sigma=2):
    kw = dict(n_levels=n_levels, k_size=k_size, sigma=sigma)
    pyr_i = laplacian_pyramid(input, **kw)
    pyr_t = laplacian_pyramid(target, **kw)
    loss = 0
    for j, (i, t) in enumerate(zip(pyr_i, pyr_t)):
        loss += torch.norm(i - t, p=1) / 2. ** (2 * j)
    return loss
