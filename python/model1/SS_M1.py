import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import time 
import matplotlib.pyplot as plt
from dropout_layers import GaussianMLP, BernoulliMLP, CLS
from utils import kld_unit_mvn, floatX, shared_dataset,log_diag_mvn, floatX
from sklearn.preprocessing import OneHotEncoder
from optimizers import optimizer
from rpy2.robjects.packages import importr
from rpy2 import robjects     
hmeasure = importr("hmeasure")

e = 1e-8
class SS_M1(object):
    def __init__(self, xdim,num_classes,alpha, args,enc_weights=None,dec_weights=None, cls_weights=None, gmm_weights=None):
        self.alpha = alpha
        self.lr_u = args.lr_u
        self.lr_l = args.lr_l
        self.batch_size = args.batch_size
        self.num_classes = num_classes
        self.xdim = xdim
        self.hdim_cls = args.hdim_cls
        self.hdim_enc = args.hdim_enc
        self.hdim_dec = args.hdim_dec
        self.hdim_gmm = args.hdim_gmm
        self.hdim_cls = args.hdim_cls
        self.zdim = args.zdim
        self.dec  = args.dec
        self.x = T.matrix('x', dtype=floatX)
        self.eps = T.matrix('eps', dtype=floatX)
        self.labels = T.matrix('labels',dtype=floatX)
        self.y_in_gmm = T.matrix('y_in_gmm',dtype=floatX)
        self.y_in_enc = T.matrix('y_in_enc',dtype=floatX)
        self.y_in_dec = T.matrix('y_in_dec',dtype=floatX)
        self.num_k = args.num_k
        self.nlayers_enc = args.nlayers_enc
        self.nlayers_dec = args.nlayers_dec
        self.nlayers_cls = args.nlayers_cls
        self.nlayers_gmm = args.nlayers_gmm
        self.training_mode = T.scalar('t_mode')
        self.dropout_rate = args.dropout_rate
        
        if args.hl_activation == 'softplus':
            self.activation = T.nnet.nnet.softplus
        elif args.hl_activation == 'relu':
            self.activation = T.nnet.relu
        
        # create list of gmm_mlp for outputs \mu_k and \sigma_k
        self.gmm_weights = [] 
        self.all_gmm     = [] 
        
        # defining the input to the encoder symbolically, since y_in_dec (labels) vary depending on the kind of the mini-batch
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_enc)
        input_to_enc = T.concatenate([self.x,self.y_in_enc],axis=1)
        self.enc_mlp = GaussianMLP(input_to_enc, self.xdim+self.num_classes, self.hdim_enc, self.zdim,self.training_mode,dropout_rate=dropout_rate, nlayers=self.nlayers_enc, eps=self.eps, Weights=enc_weights,activation=self.activation)

        
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_cls)
        self.cls_mlp = CLS(self.x, self.xdim, self.hdim_cls, self.num_k, self.training_mode, nlayers=self.nlayers_cls,Weights=cls_weights, dropout_rate=dropout_rate,activation=self.activation)


        dropout_rate = np.tile(0,args.nlayers_gmm)
        for i in range(self.num_k):
            if gmm_weights is not None:
                hl = gmm_weights[0:2]
                idx_start = i*4 + len(hl)
                idx_end = idx_start + 4
                ol = gmm_weights[idx_start:idx_end]
                both_weights = hl + ol
                weights = [both_weights,True]
            else:
                weights = None

            if i == 0:
                self.gmm_mlp = GaussianMLP(self.y_in_gmm, self.num_k, self.hdim_gmm, self.zdim, self.training_mode,dropout_rate=dropout_rate, nlayers=self.nlayers_gmm,eps=self.eps,Weights=weights,activation=self.activation)

                self.gmm_weights.extend(self.gmm_mlp.params)        
            else:
                if weights is None:
                    weights = [self.gmm_mlp.params,False]
                self.gmm_mlp = GaussianMLP(self.y_in_gmm, self.num_k, self.hdim_gmm, self.zdim, self.training_mode,dropout_rate=dropout_rate, nlayers=self.nlayers_gmm,eps=self.eps,Weights=weights,activation=self.activation)

                self.gmm_weights.extend(self.gmm_mlp.params[-4:])        
            
            
            self.all_gmm.append(self.gmm_mlp)        

        input_to_dec  = T.concatenate([self.enc_mlp.draw_sample],axis=1)
        dropout_rate = np.tile(args.dropout_rate,args.nlayers_dec)
        if self.dec == 'bernoulli':
            self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.num_classes, self.hdim_dec, self.xdim,self.training_mode,dropout_rate=dropout_rate, nlayers=self.nlayers_dec, y=self.x,Weights=dec_weights,activation=self.activation)
        elif self.dec == 'gaussian':
            self.dec_mlp = GaussianMLP(input_to_dec, self.zdim, self.hdim_dec, self.xdim,self.training_mode,dropout_rate=dropout_rate, nlayers=self.nlayers_dec, y=self.x,Weights=dec_weights,activation=self.activation)
        else:
            raise RuntimeError('unrecognized decoder %' % dec)

        # Model parameters
        self.params = self.enc_mlp.params + self.dec_mlp.params + self.cls_mlp.params + self.gmm_weights

        # use this to sample pi from the classifier MLP
        self.draw_pi = theano.function(
            inputs=[self.x, self.training_mode],
            outputs=self.cls_mlp.pi
        )

        # use this to predict label from the classifier MLP
        self.predict_label = theano.function(
            inputs=[self.x, self.training_mode],
            outputs=self.cls_mlp.y_pred
        )

        # use this to get decoder cost
        self.dec_cost = theano.function(
            inputs=[self.x,self.enc_mlp.draw_sample, self.training_mode],
            outputs=self.dec_mlp.cost
        )
        
        # use this to sample samples from the encoder
        self.draw_sample = theano.function(
            inputs = [self.x,self.eps,self.y_in_enc, self.training_mode],
            outputs = self.enc_mlp.draw_sample
        )
        
        # use this to get encoder parameters 
        self.all_params = theano.function(inputs=[],outputs=self.params)

        # use this to get decode parameters 
        self.dec_params = theano.function(inputs=[],outputs=self.dec_mlp.params)
        
        # use this to get classifier parameters 
        self.cls_params = theano.function(inputs=[],outputs=self.cls_mlp.params)

        # use this to get gmm parameters 
        self.gmm_params = theano.function(inputs=[],outputs=self.gmm_weights)

       
    
    def labeled_cost(self):
        if self.num_classes != self.num_k:
            raise RuntimeError('When having a supervised objective function the number of Gaussian components must be equal to the number of classes in the data')

        # computational graph 
        # Labeled cost
        # log p(x,y)
        self.p_y          = np.log(1.0/self.num_classes)
      
        ii=0
        for mdl in self.all_gmm:
            var = mdl.var
            mu  = mdl.mu  
            var = var.reshape((self.batch_size,self.zdim))
            mu  = mu.reshape((self.batch_size,self.zdim))
            if ii == 0: 
                all_var = var
                all_mu = mu
            else:
                all_var = T.concatenate([all_var,var],axis=1)
                all_mu = T.concatenate([all_mu,mu],axis=1)
            ii += 1
        

        indicator   = T.repeat(self.labels,self.zdim,axis=1)
        var_components = all_var * indicator
        mu_components  = all_mu * indicator
        gmm_vars = var_components[np.nonzero(var_components)].reshape((self.batch_size,self.zdim))
        gmm_mus  = mu_components[np.nonzero(mu_components)].reshape((self.batch_size,self.zdim))

        L = 0.5*(T.sum(T.log(gmm_vars+e),axis=1) + \
             T.sum(self.enc_mlp.var/(gmm_vars+e),axis=1) + \
             T.sum((1.0 / (gmm_vars+e))*(self.enc_mlp.mu - gmm_mus)*(self.enc_mlp.mu - gmm_mus),axis=1)) - \
            0.5*(T.sum(1+T.log(self.enc_mlp.var+e),axis=1))

        L_Y = self.dec_mlp.cost - self.p_y

        self.cls_cost = T.sum(self.alpha*T.nnet.categorical_crossentropy(self.cls_mlp.pi+e,self.labels))/self.batch_size

        self.labeled_cost = T.sum(L+L_Y)/self.batch_size + self.cls_cost
        
    def unlabeled_cost(self):
        # computational graph
	# Unlabeld cost

        # First we loop through all states of y. I call y_us to the Unlabeld Simulated y label
        # L_u is a mtrix of size[batch_size,num_classes] where each column is one state of y
        make_ohe = OneHotEncoder(n_values=self.num_k,dtype=np.float32)
        self.p_y          = np.log(1.0/self.num_k)

        for i in xrange(self.num_k):
            y_us = i * np.ones((self.batch_size,1))
            one_hot_y_us = make_ohe.fit_transform(y_us).toarray()

            dropout_rate = np.tile(self.dropout_rate, self.nlayers_enc)
            input_to_enc = T.concatenate([self.x,one_hot_y_us],axis=1)
            self.enc_mlp = GaussianMLP(input_to_enc, self.xdim+self.num_classes, self.hdim_enc, self.zdim,self.training_mode,dropout_rate=dropout_rate, nlayers=self.nlayers_enc, eps=self.eps, Weights=[self.enc_mlp.params,True], activation=self.activation)
            
            gmm_model_k = self.all_gmm[i]
            self.gmm_mlp = GaussianMLP(one_hot_y_us, self.num_k, self.hdim_gmm, self.zdim, self.training_mode,dropout_rate=dropout_rate,nlayers=self.nlayers_gmm, eps = self.eps, Weights=[gmm_model_k.params,True], activation=self.activation)
            
            dropout_rate = np.tile(self.dropout_rate, self.nlayers_dec)
            input_to_dec  = self.enc_mlp.draw_sample
            if self.dec == 'bernoulli':
                self.dec_mlp = BernoulliMLP(input_to_dec, self.zdim+self.num_classes, self.hdim_dec, self.xdim, self.training_mode, dropout_rate=dropout_rate, y=self.x, nlayers=self.nlayers_dec, Weights = [self.dec_mlp.params,True], activation=self.activation)
            elif self.dec == 'gaussian':
                self.dec_mlp = GaussianMLP(input_to_dec, self.zdim, self.hdim_dec, self.xdim, self.training_mode,y=self.x, dropout_rate=dropout_rate, nlayers=self.nlayers_dec, Weights = [self.dec_mlp.params,True], activation=self.activation)
            else:
                raise RuntimeError('unrecognized decoder %' % dec)

            L = 0.5*(T.sum(T.log(self.gmm_mlp.var+e),axis=1) + T.sum(self.enc_mlp.var/(self.gmm_mlp.var+e),axis=1) + T.sum((1.0 / (self.gmm_mlp.var+e)) * (self.enc_mlp.mu - self.gmm_mlp.mu)*(self.enc_mlp.mu - self.gmm_mlp.mu),axis=1)) - 0.5*(T.sum(1+T.log(self.enc_mlp.var+e),axis=1)) + self.dec_mlp.cost 
                        
            L = L.reshape((self.batch_size,1))
            
            if i==0:
                L_u = L
            else:
                L_u = T.concatenate([L_u,L],axis=1)

        U = T.sum(self.cls_mlp.pi * L_u,axis=1)
        
        L_Y = T.sum(self.cls_mlp.pi*T.log(self.cls_mlp.pi+e),axis=1) - T.sum(self.cls_mlp.pi*self.p_y,axis=1) 

        self.unlabeled_cost = T.sum(U+L_Y)/self.batch_size

        
    def labeled_training(self,data_x,data_y,batchsize):
        index = T.lscalar()

        self.labeled_cost()

        clip_max = 0 
        self.gparams_l = [T.grad(self.labeled_cost, p) for p in self.params]
        if clip_max > 0.0:
            norm = T.sqrt(sum([T.sum(T.square(g)) for g in self.gparams_l]))
            self.gparams_l = [clip_norm(g, clip_max, norm) for g in self.gparams_l]
            #self.gparams_l = [T.clip(g,-clip_max,clip_max) for g in self.gparams_l] 


        self.optimizer_l = optimizer(self.params,self.gparams_l,self.lr_l)
        self.updates_l = self.optimizer_l.adam()
        
     
        train_l = theano.function(
            inputs=[index, self.eps, self.training_mode],
            outputs=[self.labeled_cost, self.cls_cost, self.dec_mlp.cost],
            updates=self.updates_l,
            givens={
                    self.x:             data_x [index*batchsize : (index+1)*batchsize],
                    self.labels:        data_y [index*batchsize : (index+1)*batchsize],
                    self.y_in_enc:      data_y [index*batchsize : (index+1)*batchsize],
                    self.y_in_gmm:      data_y [index*batchsize : (index+1)*batchsize]
                    }
        ) 

        return train_l

    def unlabeled_training(self,data_x, batchsize):
        index = T.lscalar()

        self.unlabeled_cost()
        
        clip_max = 0.5
        self.gparams_u = [T.grad(self.unlabeled_cost, p) for p in self.params]
        if clip_max > 0:
            norm = T.sqrt(sum([T.sum(T.square(g)) for g in self.gparams_u]))
            self.gparams_u = [clip_norm(g, clip_max, norm) for g in self.gparams_u]
            #self.gparams_u = [T.clip(g,-clip_max,clip_max) for g in self.gparams_u] 

        self.optimizer_u = optimizer(self.params,self.gparams_u,self.lr_u)
        self.updates_u = self.optimizer_u.adam()
        
        train_u = theano.function(
            inputs=[index, self.eps, self.training_mode],
            outputs=self.unlabeled_cost,
            updates=self.updates_u,
            givens={
                    self.x: data_x[index*batchsize : (index+1)*batchsize]
                    }
        ) 

        return train_u

    def get_performance(self,X,y,X_cal,y_cal,scores=None, calibrate = True):
        if scores is None:          
            scores = self.draw_pi(X,0)
            scores = scores[:,1]
            if calibrate == True:
                from betacal import BetaCalibration
                            
                scores_cal = self.draw_pi(X_cal,0)
                bc =  BetaCalibration(parameters="abm")
                bc.fit(scores_cal[:,1],y_cal)
                scores = bc.predict(scores)
            
        d = {'Segment 0': robjects.FloatVector(scores)}
        dataf = robjects.DataFrame(d)
        rho = 1.0*sum(y)/float(y.shape[0])
        results = hmeasure.HMeasure(robjects.IntVector(y),dataf,threshold=rho)
    
        H    = results[0][0][0]
        Gini = results[0][1][0]
        AUC  = results[0][2][0]
        TP   = results[0][18][0]
        FP   = results[0][19][0]
        FN   = results[0][21][0]
        Recall  = 1.0*TP/(TP+FN+e)
        Precision  = 1.0*TP/(TP+FP+e)

        return {'AUC':AUC, 'Gini':Gini, 'H':H, 'Recall':Recall, 'Precision': Precision, 'scores':scores}

    def plot_gmm_space(self, X, y, X_cal, y_cal, args, num_vectors=5000, plot_name='../../output/latent_space.pdf'):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import  numpy as np
        import random
        from sklearn.preprocessing import OneHotEncoder
        from betacal import BetaCalibration
        from awsome_plot import scatter_hist
        
        print('generating latent space...')
        idx = random.sample(range(0,X.shape[0]),num_vectors)
        eps = np.random.randn(num_vectors, args.zdim).astype(floatX)
        make_ohe = OneHotEncoder(n_values=args.num_k)
        y_onehot = make_ohe.fit_transform(y[idx].reshape(y[idx].shape[0],1)).toarray().astype(np.float32)
        z = self.draw_sample(X[idx,:],eps,y_onehot,0)
        pi   = self.draw_pi(X[idx,:],0)
        # calibrate
        scores_cal = self.draw_pi(X_cal,0)
        bc =  BetaCalibration(parameters="abm")
        bc.fit(scores_cal[:,1],y_cal)
        pi = bc.predict(pi[:,1])
        y = y[idx]
        # now tsne
        print('Fitting tsne...')
        z_ta = TSNE(n_components=2).fit_transform(z)

        scatter_hist(z_ta[y==0,0],z_ta[y==0,1],z_ta[y==1,0],z_ta[y==1,1],pi[y==0],pi[y==1],plot_name)

    def summary(self,X_te,y_te,X_cal,y_cal,unlabeled_cost, labeled_cost, cls_cost, epoch, every=100):
        if epoch % every == 0 and epoch > 0:
            print 'Costs at epoch ',  str(epoch) , ' are:'
            print 'Supervised: ', labeled_cost
            print 'Unsupervised: ', unlabeled_cost
            print 'Classifier: ', cls_cost
            print 'Performace at epoch ',  str(epoch) , ' are:'
            print 'AUC: ', self.get_performance(X_te,y_te,X_cal,y_cal)['AUC']
	    #print self.print_alpha(flag=False)

    def set_alpha(self,alpha):
        self.alpha = alpha

    def print_alpha(self,flag=False):
        if flag:
            print 'current alpha is ', self.alpha
    
    def alpha_annealing(self, every_epoch, epoch, decay_rate=0, max_epoch=50, flag=False, min_alpha=0.001, max_alpha=500, mode='decay'):
        if mode == 'decay':
            if epoch % every_epoch == 0 and epoch > 0 :        
                new_alpha = max(self.alpha * decay_rate,min_alpha)
                self.set_alpha(new_alpha)
                self.print_alpha(flag=flag)
        elif mode == 'increase':
            self.set_alpha(e)
            if epoch % every_epoch == 0:        
                ratio = (np.log(max_alpha)/np.log(e))/max_epoch
                new_alpha = min(e**((epoch+1)*ratio),max_alpha)
                self.set_alpha(new_alpha)
                self.print_alpha(flag=flag)


def clip_norm(g, c, n):
    """Clip the gradient `g` if the L2 norm `n` exceeds `c`.
    # Arguments
        g: Tensor, the gradient tensor
        c: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        n: Tensor, actual norm of `g`.
    # Returns
        Tensor, the gradient clipped if required.
    """
    if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
        return g

    # tf require using a special op to multiply IndexedSliced by scalar
    g = T.switch(T.lt(c,n), g * c / n, g)

    return g
  
