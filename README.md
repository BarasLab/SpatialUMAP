## SpatialUMAP

### SpatialUMAP.py

Using cell coordinates and cell labels, a table of densities across a defined grid of distances stratified by cell label is first create from the data. These densitis describe the local microenvironment of a cell with respect to the composition of the cellular millieu across different distances. By default, the system examines 5 concentric spatial bounds (0-25, 26-50, 51-100, 101-150, 151-200 um) around each individual cell. Therefore, the total number of features extracted is 5 times the number of unique cell labels (in these data - Tumor, CD163+, CD8+, FoxP3+). This feature vector of densities is then used as input for UMAP to fit the embedding function to over 200,000 cells and then subsequently it is applied to over 200,000 different cells for visualization and analytics. 

### PlottingTool.py

Conventional 2D density ploting is performed using np.histogram2d and gaussian smoothing. Further, we allow weighting by a scalar value for each cell such as by the mean fluorescence intensity (MFI) measured for that cell for things like PD-L1 and PD-1. Note, this data is not part of the feature vector that is used as input for UMAP, described above.

