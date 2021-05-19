import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage as ndi


def plot_2d_density(X, Y=None, bins=200, n_pad=40, w=None, ax=None, gaussian_sigma=0.5, cmap=plt.get_cmap('viridis'), vlim=np.array([0.001, 0.98]), circle_type='bg',  box_off=True, return_matrix=False):
    if Y is not None:
        if w is not None:
            b, _, _ = np.histogram2d(X, Y, bins=bins)
            b = ndi.gaussian_filter(b.T, sigma=gaussian_sigma)

            s, _, _ = np.histogram2d(X, Y, bins=bins, weights=w)
            s = ndi.gaussian_filter(s.T, sigma=gaussian_sigma)

            d = np.zeros_like(b)
            d[b > 0] = s[b > 0] / b[b > 0]
            d = ndi.gaussian_filter(d, sigma=gaussian_sigma)
        else:
            d, _, _ = np.histogram2d(X, Y, bins=bins)
            d /= np.sum(d)
            d = ndi.gaussian_filter(d.T, sigma=gaussian_sigma)
    else:
        d = X

    if return_matrix:
        return d
    else:
        if np.isscalar(vlim):
            vlim = np.array([0, np.quantile(d[d > 0].flatten(), vlim)])
        else:
            if np.all((vlim < 1) & (vlim > 0)):
                vlim = np.quantile(d[d > 0].flatten(), vlim)

        if ax is None:
            _, ax = plt.subplots()

        if np.isscalar(bins):
            n_bins = bins
        else:
            n_bins = len(bins[0]) - 1

        if circle_type == 'bg':
            c = np.meshgrid(np.arange(2 * n_pad + n_bins), np.arange(2 * n_pad + n_bins))
            c = np.sqrt(((c[0] - ((2 * n_pad + n_bins) / 2)) ** 2) + ((c[1] - ((2 * n_pad + n_bins) / 2)) ** 2)) < (0.95 * ((2 * n_pad + n_bins) / 2))
            ax.pcolormesh(np.pad(d, [n_pad, n_pad]) + c, vmin=1, vmax=1 + vlim[1], cmap=cmap, shading='gouraud', alpha=1)
            # ax.pcolormesh(np.log10(np.pad(d, [n_pad, n_pad]) + c + 1), vmin=np.log10(2), vmax=np.log10(2 + vlim[1]), cmap=cmap, shading='gouraud', alpha=1)
        elif circle_type == 'arch':
            c = (n_bins / 2)
            ax.add_artist(plt.Circle((c + n_pad, c + n_pad), 0.95 * (c + n_pad), color='black', fill=False))
            ax.pcolormesh(np.pad(d, [n_pad, n_pad]), vmin=-vlim[1], vmax=vlim[1], cmap=cmap, shading='gouraud', alpha=1)
        else:
            ax.pcolormesh(np.pad(d, [n_pad, n_pad]), vmin=0, vmax=vlim[1], cmap=cmap, shading='gouraud', alpha=1)

        if box_off is True:
            [ax.spines[sp].set_visible(False) for sp in ax.spines]
            ax.set(xticks=[], yticks=[])


def plt_cmap(ax, cmap, extend, width, ylabel):
    cb = mpl.colorbar.ColorbarBase(ax=ax, cmap=cmap, extend=extend)
    cb.set_ticks([])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1], width, pos[3]])
    ax.set(ylabel=ylabel)

