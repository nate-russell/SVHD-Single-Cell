#!/usr/bin/env Rscript
library(optparse)
library(largeVis)
library(ggplot2)
library(MASS)
library(Matrix)



# Command Line options 
option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="dataset file name", metavar="character"),
  
  make_option(c("-o", "--out"), type="character", default="out.txt", 
              help="output file name [default= %default]", metavar="character"),
  
  make_option(c("-d", "--dim"), type="integer", default=2, 
              help="Dimension [default= %default]", metavar="integer"),
  
  make_option(c("-m", "--M"), type="integer", default=5, 
              help="number of negative samples [default= %default]", metavar="integer"),
  
  make_option(c("-k", "--K"), type="integer", default=5, 
              help="number of nearest neighbors [default= %default]", metavar="integer"),
  
  make_option(c("-n", "--N"), type="integer", default=5, 
              help="number of random projection trees [default= %default]", metavar="integer"),
  
  make_option(c("-i", "--I"), type="integer", default=5, 
              help="max number of iterations [default= %default]", metavar="integer"),
  
  make_option(c("-g", "--G"), type="numeric", default=7, 
              help="Gamma [default= %default]", metavar="numeric"),
  
  make_option(c("-a", "--A"), type="numeric", default=1.0, 
              help="Alpha [default= %default]", metavar="numeric"),
  
  make_option(c("-r", "--R"), type="numeric", default=1.0, 
              help="Rho [default= %default]", metavar="numeric")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}


print(opt$file)
print(opt$out)
print(opt$M * 2)
print(opt$K)
print(opt$M)
print(opt$I)
print(opt$G*2)
print(opt$A)
print(opt$R)

set.seed(1974)
#RcppArmadillo::armadillo_set_seed(1974)


# Read in the data
print("Reading Data from disk:")
print(opt$file)
df <- read.csv(opt$file,sep=',',header=F)
N <- nrow(df)
P <- ncol(df)

# Format Data
mydat <- df[1:N,1:P] # modify me to have correct shape
pca <- princomp(mydat)
mydat <- pca$scores[1:N,1:P]
init <- t(pca$scores[1:N,1:2])
print("Data Dimensions:")
print(dim(mydat))
print('Coordinate Initialization Dimensions:')
print(dim(init))

# Check for duplicates
print("Checking for duplicates:")
mydupes <- which(duplicated(mydat))
print(mydupes)
#mydat <- mydat[-mydupes, ]

# Transpose data
mydat <- t(mydat)

cat('\n\n')
neighbors <- randomProjectionTreeSearch(mydat,
                                        K = opt$K,
                                        n_trees = opt$N,
                                        tree_threshold = max(10,nrow(mydat)),
                                        max_iter = opt$I,
                                        distance_method = "Euclidean",
                                        verbose = TRUE
)

neighborIndices <- neighborsToVectors(neighbors)
rm(neighbors)
gc()
distances <- distance(x = mydat, 
                      i = neighborIndices$i, 
                      j = neighborIndices$j)
rm(mydat)
gc()
wij <- buildEdgeMatrix(i = neighborIndices$i, 
                       j = neighborIndices$j, 
                       d = distances)
rm(distances, neighborIndices)
gc()
coords <- projectKNNs(wij$wij,
                      dim = opt$dim,
                      M = opt$M, # negative samples
                      gamma = opt$G, # similar to early exageration .1 -14 likely 7
                      alpha = opt$A,# perplexity 00.01 - 10 likely 1
                      rho = opt$R,# probly dont change
                      min_rho = 0,
                      weight_pos_samples = TRUE,
                      #sgd_batches = 1000,
                      coords = init,
                      verbose = TRUE)

print('Finished Computing LargeVis Embedding')
print('Writing LargeVis Embedding to Disk: ')
print(opt$out)
write.csv(t(coords),opt$out)
print('LargeVis Script Complete')
