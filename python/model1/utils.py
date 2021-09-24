import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
from sklearn.utils import shuffle

'''
helper functions
'''

floatX = theano.config.floatX

def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    D = int(D)

    w = np.zeros((D,D), dtype=np.int64)

    for i in xrange(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    a=sum([w[i,j] for i,j in ind])*1.0/Y_pred.size
    b= w
    
    return a,b

def weights_to_gpu(np_weights,n_layers,distribution,components = 2):
    gpu_weights = []
    if distribution == 'Gaussian':
        k = 4
    elif distribution == 'GMM':
        k = 4 * components
    elif distribution == 'Classifier':
        k = 2
        
    tot_length_w = 2*n_layers + k # get the length of all weights. +4 only for gaussian!!
    
    layer_i = 0
    mu_counter=0
    sigma_counter=0
    for i in range(len(np_weights)):  
        if i%2 == 0:
            layer = 'W'
            layer_i += 1
        else:
            layer = 'b'
        
        if distribution == 'Gaussian':
            if i < (n_layers*2):     
                name_layer = 'GaussianMLP_hidden_'+str(layer_i)+'_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            elif i >= (n_layers*2) and i < (tot_length_w-2):
                name_layer = 'GaussianMLP_mu_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            else:
                name_layer = 'GaussianMLP_logvar_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
        elif distribution == 'Classifier':
            if i < (n_layers*2):     
                name_layer = 'GaussianMLP_hidden_'+str(layer_i)+'_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            else:
                name_layer = 'GaussianMLP_cls_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
        elif distribution == 'GMM':
            if i < (n_layers*2):     
                name_layer = 'GaussianMLP_hidden_'+str(layer_i)+'_'+layer
                gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
            elif i >= (n_layers*2):
                if mu_counter < 2:
                    name_layer = 'GaussianMLP_mu_'+layer
                    gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
                    mu_counter += 1 
		    if mu_counter == 2:
                        sigma_counter = 0
                else:
                    name_layer = 'GaussianMLP_logvar_'+layer
                    gpu_weights.append(theano.shared(value=np_weights[i],name=name_layer,borrow=True))
                    sigma_counter += 1
                    if sigma_counter == 2:
                        mu_counter = 0

    return gpu_weights

# XXX dataset parameters
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    if len(data_xy)==2:
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        #return shared_x, T.cast(shared_y, 'int32')
        return shared_x, shared_y
    if len(data_xy)==1:
        data_x   = data_xy[0]

        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x
    

def load_dataset(dset='paper',scaler=True, num_rej = 35000, num_bads=4570):
    import gzip, cPickle
    from sklearn import preprocessing
    from sklearn.model_selection import StratifiedShuffleSplit
    import random
    from sklearn.decomposition import PCA as sklearnPCA
    
    """setp 1 load data"""
    if dset == 'paper' or dset == 'complete':
        if dset == 'paper':
            with gzip.open('../../data/supervised_lc_paper.pk.gz', 'rb') as f:
                data_supervised = cPickle.load(f)
    
            with gzip.open('../../data/unsupervised_lc_paper.pk.gz', 'rb') as f:
                data_unsupervised = cPickle.load(f)
        elif dset == 'complete':
            with gzip.open('../../data/supervised_lc_dta.pk.gz', 'rb') as f:
                data_supervised = cPickle.load(f)
    
            with gzip.open('../../data/unsupervised_lc_dta.pk.gz', 'rb') as f:
                data_unsupervised = cPickle.load(f)
    
    
        X_a = data_supervised['data']

        # XXX merging to scale all together
        '''
        X_r = data_unsupervised['data']
        X = np.r_[X_a[:,1:],X_r[:,1:]]
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X_a = np.c_[X_a[:,0],X[0:149125,:]]
        X_r = np.c_[X_r[:,0],X[149125:,:]]
        '''

        # first column is the date
        date_a = X_a[:,0]
        X_a = X_a[:,1:]
    
        y_a = data_supervised['labels']
        # first column is the regular flag, second column is to make it one-hot-enc
        y_a = y_a[:,0]
    
        if scaler == True:
            # Create the Scaler object
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            X_a = scaler.fit_transform(X_a)
    
        # no sepparate calibration samples
        # after October 2012 and getting only 2000
        X_cal = X_a[date_a>=20121001,:]
        y_cal = y_a[date_a>=20121001]
        idx_c = random.sample(range(0,X_cal.shape[0]),2000)
        X_cal = X_cal[idx_c]
        y_cal = y_cal[idx_c]
    
        ####################################################################################
        #XXX now rejected apps
        X_r = data_unsupervised['data']
        
        # first column is the date
        date_r = X_r[:,0]
        X_r = X_r[:,1:]
        y_r = np.array([-1]*X_r.shape[0]) # just for the splitting in step 2
    
        if scaler == True:
            # Fit your data on the scaler object
            X_r = scaler.fit_transform(X_r)
    
    
        #all data before Oct 2012 as in the paper
        X_a = X_a[date_a<20121001,:]
        X_r = X_r[date_r<20121001,:]
        y_a = y_a[date_a<20121001]
    
    """ step 2 split 7:3"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index_a, test_index_a in sss.split(X_a, y_a):
        X_tr_a, X_te_a  = X_a[train_index_a], X_a[test_index_a]
        y_tr_a, y_te_a  = y_a[train_index_a], y_a[test_index_a]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index_r, test_index_r in sss.split(X_r,y_r):
        X_tr_r, X_te_r  = X_r[train_index_r], X_r[test_index_r]


    # create balanced set preserving all bads in training set
    X_a_1 = X_tr_a[y_tr_a==1,:]
    y_a_1 = y_tr_a[y_tr_a==1]

    if num_bads > y_a_1.sum():
        raise RuntimeError('the number of bads is larger thant possible. Max possible number is ' % y_a_1.sum() )

    X_a_0 = X_tr_a[y_tr_a==0,:]
    y_a_0 = y_tr_a[y_tr_a==0]

    # we need the idx to randomly select the 0's in the training set
    idx_a = random.sample(range(0,X_a_0.shape[0]),int(1.0*num_bads))
    idx_a_1 = random.sample(range(0,X_a_1.shape[0]),int(1.0*num_bads))

    idx_r = random.sample(range(0,X_r.shape[0]),int(num_rej))

    X_tr_a = np.r_[X_a_1[idx_a_1,:],X_a_0[idx_a,:]]
    X_tr_r = X_r[idx_r,:]
    y_tr_a = np.r_[y_a_1[idx_a_1],y_a_0[idx_a]]

    f.close()
    X_tr_a,y_tr_a = shuffle(X_tr_a,y_tr_a)
    data = (X_tr_a,y_tr_a,X_tr_r,X_te_a,y_te_a,X_cal,y_cal)

    return data

# costs
def kld_unit_mvn(mu, var,flag=None):
    # KL divergence from N(0, I)
    if flag == None:
        return (mu.shape[1] + T.sum(T.log(var), axis=1) - T.sum(T.square(mu), axis=1) - T.sum(var, axis=1)) / 2.0
    elif flag == 1:
        return (mu.shape[1] + np.sum(np.log(var), axis=1) - np.sum(np.square(mu), axis=1) - np.sum(var, axis=1)) / 2.0
    else:
        raise RuntimeError('the flag pass into the function is not recognized')

def log_diag_mvn(mu, var):
    def f(x):
        # expects batches
        k = mu.shape[1]

        logp = (-k / 2.0) * np.log(2 * np.pi) - 0.5 * T.sum(T.log(var), axis=1) - T.sum(0.5 * (1.0 / var) * (x - mu) * (x - mu), axis=1)
        return logp
    return f


