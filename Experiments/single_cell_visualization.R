library(largeVis)

# Load data
listData <- read.csv("Insert your input csv file path", FALSE)
matrixData <- do.call(rbind, listData) # matrix type
matrixData <- t(matrixData)
nMarkers <- 39
nolabelData <- matrixData[, c(1:nMarkers)]
data <- t(nolabelData)

# Parameter setting
dim <- 2
K <- 150
n_trees <- 100
tree_threshold <- nrow(data)
max_iter <- 2 
max_depth <- 32
distance_method <- "Euclidean" 
perplexity <- 50
M <- 5
sgd_batches <- 2000
weight_pos_samples <- TRUE
alpha <- 1 
gamma <- 7
rho <- 1
min_rho <- 0
coords <- NULL
verbose <- TRUE

knns <- randomProjectionTreeSearch(data, n_trees = n_trees, tree_threshold = tree_threshold, K = K, max_iter = max_iter,
								   max_depth = max_depth, distance_method = distance_method, verbose = verbose)

if (verbose[1])
   	cat("Calculating edge weights...")
neighbor_indices <- neighborsToVectors(knns)

if (verbose)
   	cat("Calculating neighbor distances.\n")

xs <- distance(data, neighbor_indices$i,neighbor_indices$j,distance_method,verbose)[, 1]

if (verbose)
   	cat("\n")

if ((any(is.na(xs)) + any(is.infinite(xs)) + any(is.nan(xs)) + any(xs == 0)) > 0)
   	stop("An error leaked into the distance calculation - check for duplicates")

if (any(xs > 27)) {
   	warning(paste(
	"The Distances between some neighbors are large enough to cause the calculation of p_{j|i} to overflow.",
   	"Scaling the distance vector."))
   	xs <- scale(xs, center = FALSE)
}

sigwij <- buildEdgeMatrix(i = neighbor_indices$i, j = neighbor_indices$j, d = xs, perplexity = perplexity, verbose = verbose)
rm(neighbor_indices)

end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)

start.time <-Sys.time()
coords <- projectKNNs(wij = sigwij$wij, dim = dim, sgd_batches = sgd_batches, M = M, weight_pos_samples = weight_pos_samples, 
			gamma = gamma, verbose = verbose, alpha = alpha, coords = coords, rho = rho, min_rho = min_rho)

write.table(coords, "Insert your output csv file path", row.names = FALSE, col.names = FALSE)
