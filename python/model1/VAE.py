import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import time 
import matplotlib.pyplot as plt
from mlp import GaussianMLP, BernoulliMLP
from utils import kld_unit_mvn, floatX, shared_dataset,log_diag_mvn, floatX
from sklearn.preprocessing import OneHotEncoder
from optimizers import optimizer


class VAE(object):

    def __init__(self, xdim,num_classes,args):
        self.lr = args.lr_l
        self.batch_size = args.batch_size
        self.xdim = xdim
        self.num_classes = num_classes
        self.hdim_enc = args.hdim_enc
        self.hdim_dec = args.hdim_dec
        self.zdim = args.zdim
        self.dec  = args.dec
        self.x = T.matrix('x', dtype=floatX)
        self.y = T.matrix('y', dtype=floatX)
        self.eps = T.matrix('eps', dtype=floatX)
        self.nlayers_enc = args.nlayers_enc
        self.nlayers_dec = args.nlayers_dec
        
        
        input_to_enc = T.concatenate([self.x,self.y],axis=1)                           
        

        self.enc_mlp = GaussianMLP(input_to_enc, self.xdim+self.num_classes, self.hdim_enc, self.zdim, nlayers=self.nlayers_enc, eps=self.eps)
	    #self.enc_mlp = GaussianMLP(self.x, self.xdim, self.hdim_enc, self.zdim, nlayers=self.nlayers_enc, eps=self.eps)

        if self.dec == 'bernoulli':
	        self.dec_mlp = BernoulliMLP(self.enc_mlp.out, self.zdim, self.hdim_dec, self.xdim,nlayers=self.nlayers_dec, y=self.x)
        elif self.dec == 'gaussian':
	        self.dec_mlp = GaussianMLP(self.enc_mlp.out, self.zdim, self.hdim_dec, self.xdim,nlayers=self.nlayers_dec, y=self.x)
        else:
	        raise RuntimeError('unrecognized decoder %' % dec)


        # Model parameters
        self.params = self.enc_mlp.params + self.dec_mlp.params
        
        # use this to get encoder parameters 
        self.enc_params = theano.function(inputs=[],outputs=self.enc_mlp.params)

        # use this to get decode parameters 
        self.dec_params = theano.function(inputs=[],outputs=self.dec_mlp.params)
        
    
        self.vae_cost = T.sum(-kld_unit_mvn(self.enc_mlp.mu, self.enc_mlp.var) + self.dec_mlp.cost) / args.batch_size

        zeros = np.zeros((self.batch_size,self.zdim))
        ones  = np.ones((self.batch_size,self.zdim))
        self.vae_cost2 =-log_diag_mvn(zeros,ones)(self.enc_mlp.draw_z)+self.dec_mlp.cost+log_diag_mvn(self.enc_mlp.mu,self.enc_mlp.var)(self.enc_mlp.draw_z) 

        self.vae_cost2 = T.sum(self.vae_cost2)/args.batch_size
        
    
    def training(self,data_x,data_y, epochs=50):
        index = T.lscalar()
        print '<<<<<<<<<<< Pre-training encoder and decoder using  a VAE... >>>>>>>>>>'
        self.gparams = [T.grad(self.vae_cost, p) for p in self.params]

        self.optimizer = optimizer(self.params,self.gparams,self.lr)
        self.updates   = self.optimizer.adam()
        n_batches = data_x.get_value().shape[0]/self.batch_size        

        train = theano.function(
            inputs=[index, self.eps],
            outputs=[self.vae_cost,self.vae_cost2],
            updates=self.updates,
            givens={self.x: data_x[index*self.batch_size : (index+1)*self.batch_size],
                    self.y: data_y[index*self.batch_size : (index+1)*self.batch_size]     
                    }
        ) 
        
        for e in range(epochs):
            for i in range(n_batches):
                eps  = np.random.randn(self.batch_size, self.zdim).astype(floatX)
                vae_cost,cost2 = train(i,eps)

            if e % 100 == 0 and e > 0:           
                print 'vae cost at epoch ', e, ' is ' ,vae_cost
