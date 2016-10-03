[back to main page](../)

## Performing Dimensionality reduction

Example Python
```python
from largevis import LargeVis
# Initialize Class with default values
lv = LargeVis(K=5,n_trees=5,max_iter=10,dim=2,M=5,gamma=1,alpha=1,rho=1)
# Transform Data, where X is an (n_samples,p_features) numpy array
D_lv = lv.fit_transform(X)
```

Example Command Line
```bash
Rscript largevis_cli.R -f /my/path/in.csv -o /my/path/out.csv -m 5 -k 5 -n 5 -i 10 -g 1 -a 1 -r 1
```

## Launching localhost session of webtool

Example Python
```python
from Visualization.LaunchGPE import launch_localhost
launch_localhost("path/to/dir/with/data/for/plotting",mode='normal',verbose=True,downsample=None)
```

Example Command Line
```bash
bokeh serve Visualization/BokehTool --args --dir path/to/dir/with/data/for/plotting --verbose true
```

## Getting the Single-Cell Data
TODO

## End-to-End Demonstartion
Please use the [demo.py](demo.py) script 
