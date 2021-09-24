def main():
    # import dependencies
    import numpy as np
    import theano
    import theano.tensor as T
    import cPickle as pickle
    import time 
    import matplotlib.pyplot as plt
    from SS_M1 import SS_M1
    from utils import kld_unit_mvn, floatX, shared_dataset,cluster_acc, load_dataset
    import argparse
    import gzip, json
    from sklearn.preprocessing import OneHotEncoder
    import random, os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--n_sup', default=1000, type=int,help='number of observations from the minority class, i.e. y=1')
    parser.add_argument('--n_unsup', default=30000, type=int,help='number of unsupervised  observations')
    parser.add_argument('--n_cv', default=1, type=int)
    parser.add_argument('--nlayers_enc', default=2, type=int, help='number of hidden layers in encoder MLP before output layers.') 
    parser.add_argument('--nlayers_dec', default=2, type=int, help='number of hidden layers in decoder MLP before output layers.') 
    parser.add_argument('--nlayers_gmm', default=1, type=int, help='number of hidden layers in gmm MLP before output layers.') 
    parser.add_argument('--nlayers_cls', default=1, type=int, help='number of hidden layers in classifier MLP before output layers.') 
    parser.add_argument('--num_k', default=2, type=int, help='number of elements in the gaussian mixture.') 
    parser.add_argument('--hdim_enc', default=[10,10],nargs='+', type=int, help='dimension of hidden layer in enc. Must be a list')
    parser.add_argument('--hdim_dec', default=[10,10],nargs='+', type=int, help='dimension of hidden layer in enc. Must be a list')
    parser.add_argument('--hdim_gmm', default=[10],nargs='+', type=int, help='dimension of hidden layer in enc. Must be a list')
    parser.add_argument('--hdim_cls', default=[70],nargs='+', type=int, help='dimension of hidden layer in classifier mlp')
    parser.add_argument('--zdim', default=10, type=int, help='dimension of latent variable')
    parser.add_argument('--lr_u', default=0.0001, type=float, help='learning rate for unsupervised loss')
    parser.add_argument('--lr_l', default=0.0001, type=float, help='learning rate for supervised loss')
    parser.add_argument('--epochs', default=10, type=int, help='how often to print cost')
    parser.add_argument('--save_every', default=50, type=int, help='how often to save model (in terms of epochs)')
    parser.add_argument('--outfile', default='vae_model', help='output file to save model to')
    parser.add_argument('--dset', default = 'cards', help='dataset to use')
    parser.add_argument('--dec',default='gaussian',help='Choose decoder type')
    parser.add_argument('--beta', default=1.1, type=float, help='beta param in alpha')
    parser.add_argument('--dropout_rate', default=0.01, type=float, help='dropout probability')
    parser.add_argument('--hl_activation',default='softplus',help='activation function for hidden layers in all networks')
    args = parser.parse_args()
    print(args)

    start= time.time()
    

    path = "../../output/"+str(args.outfile)
    try:    
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    # select number of runs, unlabeled, bads and classes in the dataset
    R = args.n_cv
    unlabeled_obs = args.n_unsup
    num_bads = args.n_sup
    labeled_obs = num_bads*2
    tot_obs = unlabeled_obs + labeled_obs
    num_classes = 2

    # place_holders for stats 
    auc  = np.zeros((args.epochs,R)) 
    cluster_acc  = np.zeros((args.epochs,R)) 
    cost = np.zeros((args.epochs,R))
    data = {}

    for kk in range(R):
        print('training cross-validation {} out of {}'.format(kk+1,R))
        labeled_tr_x, labeled_tr_y, unlabeled_tr_x, X_te, y_te,X_cal,y_cal = load_dataset(dset=args.dset,num_rej=unlabeled_obs,num_bads=num_bads)
        print('no. of obs. y=1 is {} out of {} '.format(labeled_tr_y.sum(),labeled_tr_y.shape[0]))
        
        # transforming labeled_tr_y to one-hot-encoding
        make_ohe = OneHotEncoder(n_values=num_classes) 
        labeled_tr_y = labeled_tr_y.reshape(labeled_tr_y.shape[0],1)
        labeled_tr_y = make_ohe.fit_transform(labeled_tr_y).toarray()
        labeled_te_y = y_te.reshape(y_te.shape[0],1)
        labeled_te_y = make_ohe.fit_transform(labeled_te_y).toarray()

        # alpha parameters
        alpha = args.beta*tot_obs/labeled_obs
        print('alpha is {}'.format(alpha))
                        
        # number of batchs, both labeled and unlabeled
        num_train_batches = labeled_tr_x.shape[0] / args.batch_size
        num_train_unlabeled_batches = unlabeled_tr_x.shape[0] / args.batch_size
        print('no. of labeled and unlabeled batches are: {} and {}'.format(num_train_batches,num_train_unlabeled_batches))

        data_x, data_y = shared_dataset((labeled_tr_x,labeled_tr_y))

        # create model
        model = SS_M1(labeled_tr_x.shape[1],num_classes,alpha,args) 
        
        # move unlabeled data to gpu
        data_x_u = shared_dataset((unlabeled_tr_x,))

        # compile training functions n
        train_labeled   = model.labeled_training(data_x,data_y, args.batch_size)
        train_unlabeled = model.unlabeled_training(data_x_u, args.batch_size)

        # cost_l and cost_u are lists, so we need the [0]
        cost_l = 0
        cost_u = 0
        

        # in the training we alternate unlabeled batches and labeled batches according to periodic_interval_batches
        periodic_interval_batches = int((labeled_tr_x.shape[0]+unlabeled_tr_x.shape[0]) / (1.0 * labeled_tr_x.shape[0]))
        print('1 supervised batch every {} unsupervised batches'.format(periodic_interval_batches))

        batches_per_epoch = num_train_batches + num_train_unlabeled_batches
        # training: for each epoch we go trough all labeled and unlabeld batches, in this order. 
        for e in xrange(args.epochs):
            ctr_sup   = 0
            ctr_unsup = 0
            
            # use learning rate decay
            model.optimizer_l.learning_decay(decay_rate=0.9, every_epoch=5, epoch=e, min_lr=0.0000001)
            model.optimizer_u.learning_decay(decay_rate=0.9, every_epoch=10, epoch=e, min_lr=0.0000001)
            model.alpha_annealing(decay_rate=0, every_epoch=1, epoch=e, max_epoch=100, min_alpha=0.0, max_alpha=alpha, mode='increase')
	    

            for i in range(batches_per_epoch):
                # whether this batch is supervised or not
                is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < num_train_batches

                # run corresponding batch
                if is_supervised:
                    eps  = np.random.randn(args.batch_size, args.zdim).astype(floatX)
                    cost_l, cls_cost, rec_cost = train_labeled(ctr_sup, eps, 1)
                    
                    ctr_sup += 1

                    if np.isnan(cost_l):
                        raise RuntimeError('supervised cost is nan')

                else:                
                    eps  = np.random.randn(args.batch_size, args.zdim).astype(floatX)
                    cost_u = train_unlabeled(ctr_unsup,eps, 1)
                    
                    ctr_unsup += 1

                    if np.isnan(cost_u):
                        raise RuntimeError('unsupervised cost is nan')

            model.summary(X_te,y_te,X_cal,y_cal,cost_u,cost_l,cls_cost,epoch=e,every=50)

            # uncomment to plot latent space every 100 epochs
            if e % 100 == 0 and e >0:
                model.plot_gmm_space(X_te, y_te, X_cal, y_cal, args, plot_name = path + '/latent_space_'+str(kk)+'_'+str(e)+'.pdf')
            
            cost[e,kk] = cost_u + cost_l
            auc[e,kk] = model.get_performance(X_te,y_te,X_cal,y_cal)['AUC']
        
        data[kk]={'y_te':y_te,'x_te':X_te,'y_cal':y_cal,'x_cal':X_cal}
        
        print('saving weights ...')
        all_weights = model.all_params()
        np.save(path+'/weights_'+str(kk)+'.npy',all_weights)

        # generate latent space
        model.plot_gmm_space(X_te, y_te, X_cal, y_cal, args, plot_name = path + '/latent_space_'+str(kk)+'.pdf' )
    
    f = gzip.open(path+'/data.pkl','wb')
    pickle.dump(data, f, protocol=2)

    plt.figure()
    plt.plot(cost,label='cost')
    plt.legend()
    plt.savefig(path+'/cost_m1.pdf')

    plt.figure()
    plt.plot(auc,label='auc')
    plt.grid()
    plt.title('Avg AUC {:.4f}'.format(np.mean(auc[-1,:])))
    plt.savefig(path+'/auc_m1.pdf')
    
    with open(path+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    end = time.time()
    elapsed = end - start   

    print('Elapsed  time: %.1fs'%elapsed)   
    
if __name__ == '__main__':
    main()
