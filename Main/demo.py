from largevis import LargeVis
from Visualization.LaunchGPE import launch_localhost
import pandas as pd
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--data", help="Data to be used in demo")
    #args = parser.parse_args()

    #if not os.path.isfile(args.data): raise ValueError(str(args.data)+" is not a valid file")
    #df = pd.read_csv(args.data)

    # Start server

    # Prep Data Dirs
    demo_path = "/data/work/ntrusse2/PSB2017/singlecelldemo"
    vectors_path = os.path.join(demo_path,"vectors")
    if not os.path.exists(demo_path): os.mkdir(demo_path)
    if not os.path.exists(vectors_path): os.mkdir(vectors_path)

    # Start server
    launch_localhost(demo_path, downsample=1500)

    df = pd.read_csv("/data/work/ntrusse2/PSB2017/singlecelldata.csv",header=None)
    df.columns = ['ter119','cd45.2','ly6g','igd','cd11c','F4/80','CD3','NKp46',
                  'CD23','CD34','CD115','CD19','PDCA-1','CD8alpha','Ly6C','CD4',
                  'CD11b','CD27','CD16/32','Siglec-F','Foxp3','B220','CD5','FcetaR1alpha',
                  'TCRgammadelta','CCR7','Sca1','CD49b','cKit','CD150','CD25','TCRb','CD43',
                  'CD64','CD138','CD103','IgM','CD44','MHC II','Gate']
    df['Gate'] = ['Gate: %d'%i for i in df['Gate'].values]

    df.to_csv(os.path.join(vectors_path,'markers.csv'))

    # Compute and Save PCA data
    X = df.drop(['Gate'],axis=1).values


    # Run PCA
    D_pca = PCA(n_components=2).fit_transform(X)
    # Save PCA
    lvdf = pd.DataFrame(D_pca, columns=['D1', 'D2'])
    lvdf.to_csv(os.path.join(vectors_path, 'pca.csv'))


    # Run largevis
    lv = LargeVis()
    D_lv = lv.fit_transform(X)
    # Save largevis
    lvdf = pd.DataFrame(D_lv,columns=['D1','D2'])
    lvdf.to_csv(os.path.join(vectors_path,'largevis.csv'))


    # Start server
    launch_localhost(demo_path,downsample=1000)













