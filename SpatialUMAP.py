import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import draw as skdraw, transform as sktran
import pickle
from time import time
import umap
from multiprocessing import Pool
from functools import partial
from scipy import optimize


class SpatialUMAP:
    @staticmethod
    def construct_arcs(dist_bin_px):
        # set bool mask of the arcs
        arcs = np.zeros([int(2 * dist_bin_px[-1]) + 1] * 2 + [len(dist_bin_px), ], dtype=bool)
        for i in range(len(dist_bin_px)):
            # circle based on radius
            rr, cc = skdraw.disk(center=(np.array(arcs.shape[:2]) - 1) / 2, radius=dist_bin_px[i] + 1, shape=arcs.shape[:2])
            arcs[rr, cc, i] = True
        # difference logic to produce arcs
        return np.stack([arcs[:, :, 0]] + [arcs[:, :, i] != arcs[:, :, i - 1] for i in range(1, arcs.shape[2])], axis=2)

    @staticmethod
    def process_cell_areas(i, cell_positions, cell_labels, dist_bin_px, img_mask, arcs):
        # true bounds to match arcs
        bounds = np.array([cell_positions[i].astype(int) - dist_bin_px[-1].astype(int), dist_bin_px[-1].astype(int) + 1 + cell_positions[i].astype(int)]).T
        # actual coordinate slices given tissue image
        coords = np.stack([np.maximum(0, bounds[:, 0]), np.array([np.minimum(a, b) for a, b in zip(np.array(img_mask.shape) - 1, bounds[:, 1])])], axis=1)
        # padded extract
        areas = np.pad(img_mask[tuple(map(lambda x: slice(*x), coords))], (bounds - coords) * np.array([-1, 1])[np.newaxis, :], mode='constant', constant_values=0)
        # area in square pixels
        areas = (areas[:, :, np.newaxis] & arcs).sum(axis=(0, 1))
        # return i and areas
        return i, areas

    @staticmethod
    def process_cell_counts(i, cell_positions, cell_labels, dist_bin_px):
        # squared distance
        counts = np.sum(np.square(cell_positions[i][np.newaxis, :] - cell_positions), axis=1)
        # inequalities around arcs
        counts = counts[np.newaxis, :] <= np.square(np.concatenate([[0], dist_bin_px]))[:, np.newaxis]
        # matmul to counts
        counts = np.diff(np.matmul(counts.astype(int), cell_labels.astype(int)), axis=0)
        # return index and counts
        return i, counts

    def __init__(self, dist_bin_um, um_per_px, area_downsample):
        # microns per pixel
        self.um_per_px = um_per_px
        # distance arcs
        self.dist_bin_um = dist_bin_um
        # in pixels
        self.dist_bin_px = self.dist_bin_um / self.um_per_px
        # downsampling factor for area calculations
        self.area_downsample = area_downsample
        self.arcs_radii = (self.dist_bin_px * self.area_downsample).astype(int)
        self.arcs_masks = SpatialUMAP.construct_arcs(self.arcs_radii)

    def clear_counts(self):
        self.counts = np.empty((self.cell_positions.shape[0], len(self.dist_bin_um), 5))

    def clear_areas(self):
        self.areas = np.empty((self.cell_positions.shape[0], len(self.dist_bin_um)))

    def start_pool(self, processes):
        self.pool = Pool(processes=processes)

    def close_pool(self):
        self.pool.close()
        del self.pool

    def process_region_counts(self, region_id):
        # get indices of cells from this region
        idx = np.where(region_id == self.cells['TMA_core_id'])[0]
        # get counts if there are cells in region
        if len(idx) > 0:
            # partial for picklable fn for pool for process with data from this region
            args = dict(cell_positions=self.cell_positions[idx], cell_labels=self.cell_labels.values[idx], dist_bin_px=self.dist_bin_px)
            pool_map_fn = partial(SpatialUMAP.process_cell_counts, **args)
            # process
            i, counts = list(map(lambda x: np.stack(x, axis=0), list(zip(*self.pool.map(pool_map_fn, range(len(idx)))))))
            # set results, adjust indexing (just in case)
            self.counts[idx] = counts[i]

    def process_region_areas(self, region_id, area_threshold, plots_directory=None):
        # get indices of cells from this region
        idx = np.where(region_id == self.cells['TMA_core_id'])[0]
        # get counts if cells are in region
        if len(idx) > 0:
            # fit ellipse from point cloud
            fit_ellipse = FitEllipse()
            idx_fit = fit_ellipse.fit(self.cell_positions[idx][:, [1, 0]], px_to_hull=(100 / self.um_per_px))
            # extract binary mask
            img_tissue_mask = fit_ellipse.img_ellipse
            # down sample for area calculations
            img_tissue_mask_dn = sktran.rescale(img_tissue_mask, self.area_downsample).astype(bool)

            # partial for picklable fn for pool for process with data from this region
            args = dict(cell_positions=self.cell_positions[idx][:, [1, 0]] * self.area_downsample, cell_labels=self.cell_labels.values[idx], dist_bin_px=self.arcs_radii, img_mask=img_tissue_mask_dn, arcs=self.arcs_masks)
            pool_map_fn = partial(SpatialUMAP.process_cell_areas, **args)
            # process
            i, areas = list(map(lambda x: np.stack(x, axis=0), list(zip(*self.pool.map(pool_map_fn, range(len(idx)))))))
            # adjust for indexing (just in case)
            areas = areas[i]
            # set filter for cells with adequate area coverage
            filt = ((areas / self.arcs_masks.sum(axis=(0, 1))[np.newaxis, ...]) > area_threshold).all(axis=1)

            # set results
            self.areas[idx] = areas
            self.cells.loc[idx, 'area_filter'] = filt

            if plots_directory is not None:
                plt.ioff()
                f = plt.figure(figsize=(3, 3))
                plt.axes()
                f.axes[0].cla()
                f.axes[0].plot(*self.cell_positions[idx].T, 'k,')
                f.axes[0].plot(*self.cell_positions[idx][idx_fit].T, 'r.', markersize=3, alpha=0.5)
                f.axes[0].plot(*self.cell_positions[idx][filt].T, 'b.', markersize=3, alpha=0.5)
                f.axes[0].imshow(img_tissue_mask, alpha=0.5)
                f.axes[0].axis('off')
                plt.tight_layout(pad=0.1)
                f.savefig('%s/%s.png' % (plots_directory, region_id), format='png')
                plt.close(f)
                del f
                plt.ion()

    def get_counts(self, pool_size=2, save_file=None):
        self.clear_counts()
        self.start_pool(pool_size)
        for region_id in tqdm(self.region_ids):
            self.process_region_counts(region_id)
        self.close_pool()

        if save_file is not None:
            column_names = ['%s-%s' % (cell_type, distance) for distance in self.dist_bin_um for cell_type in self.cell_labels.columns.values]
            pd.DataFrame(self.counts.reshape((self.counts.shape[0], -1)), columns=column_names).to_csv(save_file, index=False)

    def get_areas(self, area_threshold, pool_size=2, save_file=None, plots_directory=None):
        self.clear_areas()
        self.cells['area_filter'] = False
        self.start_pool(pool_size)
        for region_id in tqdm(self.region_ids):
            self.process_region_areas(region_id, area_threshold=area_threshold, plots_directory=plots_directory)
        self.close_pool()

        if save_file is not None:
            pd.DataFrame(self.areas, columns=self.dist_bin_um).to_csv(save_file, index=False)

    def set_train_test(self, n, seed=None):
        region_ids = self.cells['TMA_core_id'].unique()
        self.cells[['umap_train', 'umap_test']] = False
        for region_id, group in self.cells.groupby('Sample_number'):
            if group['area_filter'].sum() >= (n * 2):
                idx_train, idx_test, _ = np.split(np.random.default_rng(seed).permutation(group['area_filter'].sum()), [n, n * 2])
                self.cells.loc[group.index[group.area_filter][idx_train], 'umap_train'] = True
                self.cells.loc[group.index[group.area_filter][idx_test], 'umap_test'] = True


class FitEllipse:
    def __init__(self):
        self.x = None
        self.img_ellipse = None

    @staticmethod
    def ellipse_function(points, x, y, a, b, r):
        t = np.array([np.cos(r), np.sin(r)])
        d = points - np.array([x, y])[np.newaxis, ...]
        return np.square(((t[0] * d[:, 0]) + (t[1] * d[:, 1])) / a) + np.square(((t[1] * d[:, 0]) - (t[0] * d[:, 1])) / b)

    @staticmethod
    def ellipse_area(a, b):
        return np.pi * a * b

    def draw_ellipse(self, x=None):
        assert self.img_ellipse is not None
        _x = x if x is not None else self.x
        xx, yy = skdraw.ellipse(_x[0], _x[1], _x[2], _x[3], self.img_ellipse.shape, _x[4])
        self.img_ellipse[:] = False
        self.img_ellipse[xx, yy] = True

    def fit(self, d, px_to_hull):
        idx_fit = np.ones(d.shape[0], dtype=bool)
        idx_remove = True
        while np.any(idx_remove):
            hull = ConvexHull(d[idx_fit])
            d_h = np.sum(np.square(d[idx_fit][:, np.newaxis, :] - d[idx_fit][hull.vertices][np.newaxis, :, :]), axis=-1)
            idx_remove = np.sum(d_h < np.square(px_to_hull), axis=0) < 5
            idx_fit[np.where(idx_fit)[0][hull.vertices[idx_remove]]] = False
        idx_fit = np.where(idx_fit)[0][np.unique(np.argsort(d_h, axis=0)[:50])]

        self.w, self.h = np.max(d, axis=0).astype(int)
        x_init = np.concatenate([np.array(np.array((self.w, self.h))) / 2, np.log(np.array((self.w, self.h))), [0, ]]).astype(float)
        self.res = optimize.minimize(lambda x: np.mean(np.abs(FitEllipse.ellipse_function(d[idx_fit], x[0], x[1], np.exp(x[2]), np.exp(x[3]), x[4]) - 1)), x_init, method='nelder-mead')
        self.x = self.res.x.copy()
        self.x[2], self.x[3] = np.exp(self.x[[2, 3]])

        self.img_ellipse = np.zeros((self.w, self.h), dtype=bool)
        self.draw_ellipse()

        return idx_fit
