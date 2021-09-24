import numpy as np
import theano
import theano.tensor as T
from utils import log_diag_mvn, floatX
from sklearn.preprocessing import OneHotEncoder
from theano.tensor.shared_randomstreams import RandomStreams

# XXX
rng = np.random.RandomState()
srng = RandomStreams()
e = 1e-8

class HiddenLayer(object):

    # adapted from http://deeplearning.net/tutorial/mlp.html
    # finds last weights to be assigend in putput layer: last_weights = Weights[-4:]
    # finds the weights positions in the list list(set(Weights)-set(last_weights))

    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, prefix=''):
        self.n_in = n_in
        self.n_out = n_out
        method = 'random'      
        if W is None:
            # NOTE tried glorot init and randn and glorot init worked better
            # after 1 epoch with adagrad
            if method == 'glorot':
                W_values = np.asarray(
                            rng.uniform(
                                low=-np.sqrt(2. / (n_in + n_out)),
                                high=np.sqrt(2. / (n_in + n_out)),
                                size=(n_in, n_out)
                            ),dtype=theano.config.floatX)
            elif method == 'random':            
                W_values = np.asarray(
                         rng.normal(size=(n_in,n_out),loc=0,scale=0.001),
                        dtype=floatX
                        )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'_W', borrow=True)
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'_b', borrow=True)
        
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        
        if activation is None:
            output = lin_output
        elif activation == T.nnet.relu:
            output = activation(lin_output,0.0)
        else:
            output = activation(lin_output)
        self.output = output
            
        # parameters of the model
        self.params = [self.W, self.b]

class _MLP(object):
    # building block for MLP instantiations defined below
    def __init__(self, x, n_in, n_hid, nlayers=1, prefix='',Weights=None):
        self.nlayers = nlayers
        self.hidden_layers = list()
        inp = x

        factor = 2  # the number of parameters per layer. 
    
        if Weights == None:        
            Weights = np.tile([None],self.nlayers*factor) # create an array with Nones so I can pass None weihts in the loop below
        
        for k in xrange(self.nlayers): 
            # keep track of the weight index as function of k            
            idx_weight = k*factor
            idx_bias = idx_weight + 1
            idx_gamma = idx_bias + 1
            idx_beta  = idx_gamma + 1

            hlayer = HiddenLayer(
                W=Weights[idx_weight],
                b=Weights[idx_bias],
                input=inp,
                n_in=n_in,
                n_out=n_hid[k],
                activation=T.nnet.nnet.softplus,
                #activation=T.nnet.relu,
                prefix=prefix + ('_%d' % (k + 1))
                )

            n_in = n_hid[k]
            inp = hlayer.output
            self.hidden_layers.append(hlayer)

        self.params = [param for l in self.hidden_layers for param in l.params]
        self.input = input
        # NOTE output layer computed by instantations


class GaussianMLP(_MLP):
    # Weight is a list with W's and b's to be pass into HiddenLayer, mu_layer and log_var_layer
    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, y=None, eps=None,Weights=None): 

        if Weights is not None:
            # the pass_output flag controls whether to pass output layers weights
            # pass_output = True:  assigns weights to both hidden and output layers
            # pass_output = False: assigns weights only to hidden layers
            pass_output  = Weights[1]    
            Weights      = Weights[0]

            if pass_output == True:
                if y == None and eps == None:
                    output_weights =  Weights[-2:] # this extracts the last two set of weights in the list. 
                    W = output_weights[0]    
                    b = output_weights[1]       
                else:
                    output_weights =  Weights[-4:] # this extracts the last four set of weights in the list. 
                    W_mu = output_weights[0]    
                    b_mu = output_weights[1]       
                    W_var = output_weights[2]    
                    b_var = output_weights[3]       
            else:
                if y == None and eps == None:
                    output_weights =  Weights[-2:]
                    W = None   
                    b = None
                else:
                    output_weights =  Weights[-4:]
                    W_mu = None
                    b_mu = None
                    W_var = None
                    b_var = None
            
            hidden_weights = Weights[0:len(Weights)-len(output_weights)]
            Weights = hidden_weights 
        else:
            if y == None and eps == None:
                W = None   
                b = None
            else:
                W_mu = None
                b_mu = None
                W_var = None
                b_var = None

            Weights = None
 
        super(GaussianMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='GaussianMLP_hidden',Weights=Weights)
        # for use as classifier
        if y == None and eps == None:
            self.cls_layer = HiddenLayer(
                input=self.hidden_layers[-1].output,
                n_in=self.hidden_layers[-1].n_out,
                n_out=n_out,
                activation=T.nnet.softmax,
                prefix='GaussianMLP_cls',
                W=W,
                b=b
            )
            self.params = self.params + self.cls_layer.params 
            self.pi = self.cls_layer.output                 # this will be a list of size equal to the minibatch, and each 
                                                            # entry has k elements, where k is the number of classes
            self.y_pred = T.argmax(self.pi, axis=1)         # this is a list of size equal to the minibatch with the predicted class

        # for use as decoder, encoder or gmm        
        else:
            self.mu_layer = HiddenLayer(
                input=self.hidden_layers[-1].output,
                n_in=self.hidden_layers[-1].n_out,
                n_out=n_out,
                activation=None,
                prefix='GaussianMLP_mu',
                W=W_mu,
                b=b_mu
            )
            # log(sigma^2)
            self.logvar_layer = HiddenLayer(
                input=self.hidden_layers[-1].output,
                n_in=self.hidden_layers[-1].n_out,
                n_out=n_out,
                activation=None,
                prefix='GaussianMLP_logvar',
                W=W_var,
                b=b_var
            )
            self.mu = self.mu_layer.output
            self.var = T.exp(self.logvar_layer.output)
            self.sigma = T.sqrt(e+self.var)
            self.params = self.params + self.mu_layer.params + self.logvar_layer.params 
            # for use as encoder
            if eps:
                assert(y is None)
                # XXX separate reparametrization
                self.out = (self.mu + self.sigma * eps) + srng.normal(eps.shape,0,0.05,dtype=floatX)
                self.draw_sample = self.mu + self.sigma * eps 
            # for use as decoder
            if y:
                assert(eps is None)
                # XXX specific to [0, 1] outputs
                #self.out = T.nnet.sigmoid(self.mu)
                self.out = self.mu
                self.cost = -log_diag_mvn(self.out, self.var)(y)
                

    # change the run_mode in all layers with BatchNormalization 
    def all_layers_runmode(self, run_mode=1):
        for i in range(len(self.hidden_layers)):
            current_layer = self.hidden_layers[i]
            current_layer.set_runmode(run_mode)



class BernoulliMLP(_MLP):
    # Weight is a list with W's and b's to be pass into HiddenLayer and output layer
    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, y=None, Weights = None):

        if Weights is not None:
            pass_output  = Weights[1]    
            Weights      = Weights[0]
            output_weights =  Weights[-2:] # this extracts the last two set of weights.

            if pass_output == True:
                W = output_weights[0]    
                b = output_weights[1]       

                hidden_weights = Weights[0:len(Weights)-(len(output_weights))]
                Weights = hidden_weights
            else:
                W = None
                b = None
            
                hidden_weights = Weights[0:len(Weights)-(len(output_weights))]
                Weights = hidden_weights
        else:
            W = None
            b = None
            Weights = None
        
        super(BernoulliMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='BernoulliMLP_hidden',Weights=Weights)
        self.out_layer = HiddenLayer(
            W=W,
            b=b,
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=T.nnet.sigmoid,
            prefix='BernoulliMLP_x_hat'
        )
        self.params = self.params + self.out_layer.params 
        if y:
            self.out = self.out_layer.output
            self.cost = T.sum(T.nnet.binary_crossentropy(self.out, y),axis=1)

    # change the run_mode in all layers with BatchNormalization 
    def all_layers_runmode(self, run_mode=1):
        for i in range(len(self.hidden_layers)):
            current_layer = self.hidden_layers[i]
            current_layer.set_runmode(run_mode)

