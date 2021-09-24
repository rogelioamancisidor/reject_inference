def main(idx):
    # import dependencies
    import os
    from sklearn.manifold import TSNE
    import numpy as np
    import theano
    import theano.tensor as T
    import cPickle as pickle
    import time 
    import gzip
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from SS_M1  import SS_M1
    from utils import floatX, shared_dataset, load_dataset, weights_to_gpu
    import argparse
    from sklearn.preprocessing import OneHotEncoder
    import random
    import seaborn as sns
    import json
    import cPickle as pickle
  
    # give the path where you have saved all inputs
    path = "../../output/m1"
    
    parser = argparse.ArgumentParser() 
    args = parser.parse_args()
    with open(path+'/commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    
    # extract weights
    # specify idx (number of cross-validation to use)
    print('loading weights ...')
    weights_path = path+'/weights_'+str(idx)+'.npy'
    weights = np.load(weights_path)

    # order in weights: self.enc_mlp.params + self.dec_mlp.params + self.cls_mlp.params + self.gmm_weights
    # the number of weights needs to be adjusted depending on the architecture. Another option is to pickle the enterie model to
    # avoid loading weights.
    enc_weights = [weights_to_gpu(weights[0:8],args.nlayers_enc,'Gaussian'), True]
    dec_weights = [weights_to_gpu(weights[8:16],args.nlayers_dec,'Gaussian'),True]
    cls_weights = [weights_to_gpu(weights[16:20],args.nlayers_cls,'Classifier'),True]
    gmm_weights = weights_to_gpu(weights[20:],args.nlayers_gmm,'GMM')

    # load data
    with gzip.open(path+'/data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_te  = data[idx]['x_te']
    y_te  = data[idx]['y_te']
    X_cal = data[idx]['x_cal']
    y_cal = data[idx]['y_cal']
    
    # create model
    alpha = 12.0847615979 
    print('loading pre-tranied model')
    model = SS_M1(X_te.shape[1], args.num_k, alpha, args, 
                        enc_weights = enc_weights, 
                        dec_weights = dec_weights,
                        cls_weights = cls_weights,
                        gmm_weights = gmm_weights
                        )

    print('AUC:{} '.format(model.get_performance(X_te,y_te,X_cal,y_cal)['AUC']))

if __name__ == '__main__':
    idxs = [2]
    for idx in idxs:
        main(idx)
