import numpy as np
import pandas as pd
from SpatialUMAP import *


# datafiles to be used
cell_datafile = 'data/Single_cell_data_93specimens_de.identified.csv'
clinical_datafile = 'data/Clinical.data_5.12.2021_de.identified.xlsx'

# instantiate the spatial umap object
spatial_umap = SpatialUMAP(dist_bin_um=np.array([25, 50, 100, 150, 200]), um_per_px=0.5, area_downsample=.2)
# load in cells and patient data
spatial_umap.cells = pd.read_csv(cell_datafile, sep=',').iloc[:, 1:]
spatial_umap.patients = pd.read_excel(clinical_datafile, sheet_name=0)
# set explicitly as numpy array the cell coordinates (x, y)
spatial_umap.cell_positions = spatial_umap.cells[['Xcor', 'Ycor']].values
# set explicitly as one hot data frame the cell labels
spatial_umap.cell_labels = pd.get_dummies(spatial_umap.cells['Lineage'])
# set the region is to be analyzed (a TMA core is treated similar to a region of a interest)
spatial_umap.region_ids = spatial_umap.cells.TMA_core_2.unique()
# clear metrics
spatial_umap.clear_counts()
spatial_umap.clear_areas()

# get the counts per cell and save to pickle file
# spatial_umap.get_counts(pool_size=20, pickle_file='data/counts.pkl')
spatial_umap.counts = pickle.load(open('data/counts.pkl', 'rb'))

# get the areas of cells and save to pickle file
area_threshold = 0.8
# spatial_umap.get_areas(area_threshold, pool_size=20, pickle_file='areas.pkl', plots_directory='data/plots')
spatial_umap.areas = pickle.load(open('areas.pkl', 'rb'))

# calculate density base on counts of cells / area of each arc examine
spatial_umap.density = np.empty(spatial_umap.counts.shape)
spatial_umap.cells['area_filter'] = ((spatial_umap.areas / spatial_umap.arcs_masks.sum(axis=(0, 1))[np.newaxis, ...]) > area_threshold).all(axis=1)
spatial_umap.density[spatial_umap.cells['area_filter'].values] = spatial_umap.counts[spatial_umap.cells['area_filter'].values] / spatial_umap.areas[spatial_umap.cells['area_filter'].values][..., np.newaxis]

# set training and "test" cells for umap training and embedding, respectively
spatial_umap.set_train_test(n=2500, seed=54321)
# generate umap fit and embedding on the "test" cells
spatial_umap.generate_umap()

# save as pickle for plotting
pickle.dump(spatial_umap, open('data/spatial_umap.pkl', 'wb'))
