#!/usr/bin/env python
from __future__ import division, print_function

from functools import partial
import os

import numpy as np
from numpy import newaxis
from sklearn.utils.extmath import row_norms
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torchvision.utils import save_image
import tqdm

from laplacian_loss import laplacian_loss


def make_generator(latent_dim=100, num_channels=3, image_size=64):
    nz = latent_dim
    ngf = image_size
    nc = num_channels

    # based on DCGAN from pytorch/examples
    generator = nn.Sequential(
        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh(),
        # state size. (nc) x 64 x 64
    )

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    generator.apply(weights_init)

    return generator


def scale_latents(pca_codes, copy=True):
    z = pca_codes.copy() if copy else pca_codes
    norms = row_norms(z)
    z[norms > 1] /= norms[:, newaxis]
    return z


def load_data(data_dir):
    all_imgs = np.load(os.path.join(data_dir, 'imgs.npy'), mmap_mode='r')
    with np.load(os.path.join(data_dir, 'pca.npz')) as d:
        latents = scale_latents(d['codes'], copy=False)
    return all_imgs, latents


def var_from_numpy(x, cuda=False, requires_grad=False):
    v = torch.from_numpy(x)
    if cuda:
        v = v.cuda()
    return Variable(v, requires_grad=requires_grad)


def project_(z, inds=None):
    if inds is None:
        z_norms = torch.norm(z, p=2, dim=1)
        small_enough = z_norms <= 1
        if not small_enough.all():
            z_norms.masked_fill_(small_enough, 1)
            z /= z_norms[:, np.newaxis]
    else:
        z_norms = torch.norm(z[inds], p=2, dim=1)
        which = z_norms > 1
        if which.any():
            z[inds[which], :] /= z_norms[which][:, np.newaxis]


def train(all_imgs, init_latents, out_path='.',
          epochs=50, batch_size=256, cuda=False, latent_dim=100,
          loss_fn='laplacian', optimizer='SGD',
          checkpoint_every=10, sample_every=1):
    make_var = partial(var_from_numpy, cuda=cuda)
    epoch_dir = os.path.join(out_path, 'glo-{}').format
    for e in range(epochs):
        d = epoch_dir(e)
        if os.path.isdir(d):
            if len(os.listdir(d)) > 0:
                raise ValueError("Directory already exists: {}".format(d))
            else:
                os.rmdir(d)

    z = make_var(init_latents.copy(), requires_grad=True)
    project_(z.data)
    generator = make_generator(latent_dim=latent_dim)
    if cuda:
        generator = generator.cuda()

    opt_class = getattr(torch.optim, optimizer)
    opt = opt_class([
        {'params': generator.parameters(), 'lr': 1},
        {'params': [z], 'lr': 10},
    ])

    if loss_fn == 'laplacian':
        loss_fn = partial(laplacian_loss, cuda=cuda)
    elif loss_fn.startswith('laplacian-'):
        n_levels = int(loss_fn[len('laplacian-'):])
        loss_fn = partial(laplacian_loss, n_levels=n_levels, cuda=cuda)
    elif loss_fn == 'mse':
        loss_fn = nn.functional.mse_loss

    samp_latent_stds = np.random.randn(100, latent_dim).astype(np.float32)

    #epoch_t = tqdm.trange(epochs, desc='Epoch')
    epoch_t = range(epochs)
    for epoch in epoch_t:
        do_checkpoint = epoch % checkpoint_every == 0 or epoch == epochs - 1
        do_sample = epoch % sample_every == 0 or epoch == epochs - 1
        if do_checkpoint or do_sample:
            os.makedirs(epoch_dir(epoch))  # save time if not writeable

        # samp = RandomSampler(all_imgs)
        from torch.utils.data.sampler import SequentialSampler
        samp = SequentialSampler(all_imgs)
        batcher = BatchSampler(samp, batch_size=batch_size, drop_last=True)
        #t = tqdm.tqdm(batcher, desc='Batch')
        t = batcher
        for batch_i, inds in enumerate(t):
            inds_v = make_var(np.asarray(inds))
            opt.zero_grad()

            zs = z[inds_v][:, :, newaxis, newaxis]
            recons = generator(zs)
            imgs = make_var(all_imgs[inds])
            loss = loss_fn(recons, imgs)

            loss.backward()
            opt.step()

            print("{}/{}: loss {:.5}; {:.3} {:.3} {:.3} {:.3}".format(
                epoch, batch_i, loss.data.cpu().numpy()[0],
                max(np.max(np.abs(p.data.cpu().numpy())) for p in generator.parameters()),
                max(np.max(np.abs(p.grad.data.cpu().numpy())) for p in generator.parameters()),
                np.max(np.abs(z.data.cpu().numpy())),
                np.max(np.abs(z.grad.data.cpu().numpy())),
                ))

            project_(z.data, inds=inds_v.data)

            loss_val = loss.data.cpu().numpy()[0]
            # t.set_postfix(loss='{:.4f}'.format(loss_val))
            if np.any(np.isnan(loss_val)):
                raise ValueError("loss is nan :(")

        # epoch_t.write("Epoch {}: final loss {}".format(epoch, loss_val))
        if do_checkpoint or do_sample:
            d = partial(os.path.join, epoch_dir(epoch))

            z_numpy = z.data.cpu().numpy()
            mean = np.mean(z_numpy, axis=0)
            cov = np.cov(z_numpy, rowvar=False)

            if do_checkpoint:
                torch.save(generator.state_dict(), d('gen-params.pkl'))
                torch.save(opt.state_dict(), d('opt-state.pkl'))
                np.save(d('latents.npy'), z_numpy)
                np.savez(d('latents-fit.npz'), mean=mean, cov=cov)

            if do_sample:
                try:
                    l_factor = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    u, s, v = np.linalg.svd(cov)
                    l_factor = u * np.sqrt(s)[np.newaxis, :]
                z_samps = samp_latent_stds.dot(l_factor)
                z_samps += mean
                z_samps = z_samps.astype(np.float32)
                img_samps = generator(make_var(z_samps)[:, :, newaxis, newaxis])
                save_image(img_samps.data, d('samples.jpg'), nrow=10)
                np.save(d('samples-latents.npy'), z_samps)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('out_path', default='.', nargs='?')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument(
        '--loss-fn', choices=['laplacian', 'mse'] +
                             ['laplacian-{}'.format(i) for i in range(1, 7)],
        default='laplacian')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam'],
                        default='SGD')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--sample-every', type=int, default=1)
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    all_imgs, init_latents = load_data(args.data_dir)
    kwargs = vars(args)
    del kwargs['data_dir'], kwargs['seed']
    train(all_imgs, init_latents, **kwargs)


if __name__ == '__main__':
    main()
