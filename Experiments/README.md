## Data preparation

We used cytometry single-cell data. The data file is a fcs format, so in order read the files, we used Laszlo Balkay's FCS data reader.

## Installation

If you see the largeVis_single_cell folder, there is src folder. It contains C++ implementation of largeVis. Using the cpp files, you can modify the largeVis algorithm. For example, you can change the probability function. After modifying the code, you can build the C++ code, or you can install it as your own largeVis R package by using devtools::install_local in R. 

## Usage

After build the code or installing the largeVis R package, you can run your algorithm. In R, you can use it by making a simple R script. The file single_cell_visualization.R is an example script for visualization of high-dimensional single_cell data. You first need to designate your input and output csv file path. Then, you have to tune the parameters of largeVis according to your data set.

## MATLAB code

Reading fcs files: We used Laszlo Balkay's fcs file reader (fca_readfcs.zip). <br />
Applying the Hungarian algorithm (also known as Munkres' algorithm): We used Yi Cao's MATLAB code (munkres.zip). <br />
Clustering, computing F1-measures and creating plots: We used our own code.
