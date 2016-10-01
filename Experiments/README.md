## Data preparation

We used cytometry single-cell data. The data file is fcs format. We read the file using MATLAB fcs data reader before analyzing the data. We used Laszlo Balkay's FCS data reader.

## Installation

If you look at the largeVis_single_cell folder, there is src folder. It contains C++ implementation of largeVis. By modifying the cpp files, you can tune the largeVis algorithm. For example, you can change the probability function. After modifying the code, you can install your own largeVis package using devtools::install_local(your largeVis sourcode path). 

## Usage

After installing largeVis package, you can use it by making a simple R script. The file single_cell_visualization.R is an example script for visualization of high-dimensional single_cell data. You first need to insert your input and output csv file path. Before you run the script, you have to tune the parameters of largeVis according to your data set.

## MATLAB code

Reading fcs files: We used Laszlo Balkay's fcs file reader (fca_readfcs.zip).
Applying the Hungarian algorithm (also known as Munkres' algorithm): We used Yi Cao's MATLAB code (munkres.zip).
Clustering, computing F1-measures and creating plots: We used our own code.
