import subprocess
import os
import numpy as np
import tempfile
import pandas as pd
from sklearn.datasets import make_blobs



class LargeVis:

    def __init__(self,K=5,n_trees=5,max_iter=10,
                 dim=2,M=5,gamma=1,alpha=1,rho=1):

        self.k = K
        self.n_trees = n_trees
        self.max_iter = max_iter
        self.dim = dim
        self.negative_samples = M
        self.gamma = gamma
        self.alpha = alpha
        self.rho = rho



    def fit_transform(self,X=None,path=None):
        """


        :param X:
        :param path:
        :return:
        """

        # Designate Outfile location
        tmpdir = tempfile.mkdtemp()
        print('Temp Dir: ' + tmpdir)

        # Get or make data disk location
        if isinstance(path,str):
            print('found data path')
            in_path = path

        elif isinstance(X,np.ndarray):
            print("Found numpy array")
            # write object to temporary file
            in_path = tmpdir+"/numpy_data.csv"
            np.savetxt(in_path,X,delimiter=',')
            print("Temp X file: " + in_path)

        else:
            raise TypeError('invalid data passed')

        out_path = tmpdir+'/out.csv'


        os.chdir(os.path.dirname(__file__))

        print('\nCurrent Working Director:',os.curdir)

        # Call R command line script via subprocess
        subprocess.call(["Rscript","largevis_cli.R",
                         "-f", in_path,
                         "-o", out_path,
                         "-m", str(self.negative_samples),
                         "-k", str(self.k),
                         "-n", str(self.n_trees),
                         "-i", str(self.max_iter),
                         "-g", str(self.gamma),
                         "-a", str(self.alpha),
                         "-r", str(self.rho),
                         ])

        # Read largevis out file back into python memory
        df = pd.read_csv(out_path)
        df.drop(["Unnamed: 0"],axis=1,inplace=True)

        return df.values


def test_largevis():
    """

    :return:
    """

    X,y = make_blobs(n_samples=100,n_features=10,centers=3)
    lv = LargeVis()
    D = lv.fit_transform(X)
    print(D)
    #scatter_plot(D,['Cluster '+str(i) for i in y],c_type='qual',axis_label=['x','y'],title='largevis demo')






if __name__ == "__main__":
    test_largevis()

