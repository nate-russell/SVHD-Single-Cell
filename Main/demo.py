from largevis import LargeVis
from Visualization.LaunchGPE import launch_localhost
import pandas as pd
import argparse
import os
from sklearn.decomposition import PCA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Data to be used in demo")
    args = parser.parse_args()

    if not os.path.isfile(args.data): raise ValueError(str(args.data)+" is not a valid file")
    df = pd.read_csv(args.data)

    print(df)




