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
    from SS_M2  import SS_M2
    from utils import kld_unit_mvn, floatX, shared_dataset,equally_number,cluster_acc, load_dataset, weights_to_gpu
    import argparse
    from sklearn.preprocessing import OneHotEncoder
    import random
    import seaborn as sns
    import json
    import cPickle as pickle
   
    path = "../../output/m2"
    
    parser = argparse.ArgumentParser() 
    args = parser.parse_args()
    with open(path+'/commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    
    # extract weights
    # specify idx (number of cross-validation to use)
    print('loading weights ...')
    weights_path = path+'/weights_'+str(idx)+'.npy'
    weights = np.load(weights_path)

    # order in weights: self.enc_mlp.params + self.dec_mlp.params + self.cls_mlp.params + self.aux_mlp_inf.params + self.gmm   
    # the number of weights needs to be adjusted depending on the architecture. Another option is to pickle the enterie model to
    # avoid loading weights.
    enc_weights = [weights_to_gpu(weights[0:10],args.nlayers_enc,'Gaussian'), True]
    dec_weights = [weights_to_gpu(weights[10:20],args.nlayers_dec,'Gaussian'),True]
    cls_weights = [weights_to_gpu(weights[20:24],args.nlayers_cls,'Classifier'),True]
    aux_weights = [weights_to_gpu(weights[24:32],args.nlayers_aux_inf,'Gaussian'),True]
    gmm_weights = weights_to_gpu(weights[32:],args.nlayers_gmm,'GMM')

    # load data
    with gzip.open(path+'/data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_te  = data[idx]['x_te']
    y_te  = data[idx]['y_te']
    X_cal = data[idx]['x_cal']
    y_cal = data[idx]['y_cal']
    
    # create model
    alpha = 0.0878891752577 
    print('loading pre-tranied model')
    model = SS_M2(X_te.shape[1], args.num_k, alpha, args, 
                        enc_weights = enc_weights, 
                        dec_weights = dec_weights,
                        cls_weights = cls_weights,
                        gmm_weights = gmm_weights,
                        aux_weights = aux_weights 
                        )

    print('AUC:{} '.format(model.get_performance(X_te,y_te,X_cal,y_cal)['AUC']))

if __name__ == '__main__':
    idxs = [0,1,2,3,4]
    for idx in idxs:
        main(idx)
