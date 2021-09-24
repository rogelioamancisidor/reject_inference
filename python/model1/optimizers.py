"""
The MIT License (MIT)
Copyright (c) 2015 Alec Radford
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""  
import theano
import numpy as np
from theano import config
from collections import OrderedDict
import theano.tensor as T

class optimizer(object):
    def __init__(self,params, grads, lr):
        self.lr = lr
        self.params = params
        self.grads = grads

    #def __init__(self,params, loss, lr,clip_norm=0):
    #    self.lr = lr
    #    self.params = params
    #    self.loss = loss
    #    self.clip_norm = clip_norm
    
    '''def get_grads(self):
        grads = [T.grad(self.loss, p) for p in self.params]
        if self.clip_norm > 0:
            norm = T.sqrt(sum([T.sum(T.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clip_norm, norm) for g in grads]
        return grads'''

    def set_lr(self,lr):
        self.lr = lr    

    def print_lr(self,flag=False):
        self.flag = flag
        if flag:
            print 'current lr is ', self.lr
    
    def learning_decay(self, decay_rate, every_epoch, epoch, flag=False, min_lr=0.00008):
        if epoch % every_epoch == 0 and epoch > 0 :        
            new_lr = max(self.lr * decay_rate,min_lr)
            self.set_lr(new_lr)
            self.print_lr(flag=flag)

    def adam(self, b1=0.1, b2=0.001, e=1e-8, opt_params=None):
        """
        ADAM Optimizer
        cost (to be minimized)
        params (list of parameters to take gradients with respect to)
        .... parameters specific to the optimization ...
        opt_params (if available, used to intialize the variables
        """

        updates = []
        #self.grads = self.get_grads()

        restartOpt = False
        if opt_params is None:
            restartOpt = True
            opt_params=OrderedDict()
        
        #Track the optimization variable
        if restartOpt:
            i = theano.shared(np.asarray(0).astype(config.floatX),name ='opt_i',borrow=True)
            opt_params['opt_i'] = i
        else:
            i = opt_params['opt_i']
        
        #No need to reload these theano variables
        g_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'g_norm',borrow=True)
        p_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'p_norm',borrow=True)
        opt_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'opt_norm',borrow=True)
        
        #Initialization for ADAM
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        
        for p, g in zip(self.params, self.grads):
            if restartOpt:
                m = theano.shared(np.array(p.get_value() * 0.).astype(config.floatX),name = 'opt_m_'+p.name,borrow=True)
                v = theano.shared(np.array(p.get_value() * 0.).astype(config.floatX),name = 'opt_v_'+p.name,borrow=True)
                opt_params['opt_m_'+p.name] = m
                opt_params['opt_v_'+p.name] = v
            else:
                m = opt_params['opt_m_'+p.name] 
                v = opt_params['opt_v_'+p.name]
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t+e))
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
            #Update norms
            g_norm += (g**2).sum()
            p_norm += (p**2).sum() 
            opt_norm+=(m**2).sum() + (v**2).sum()
        updates.append((i, i_t))
        
        #return updates, [T.sqrt(p_norm), T.sqrt(g_norm), T.sqrt(opt_norm)], opt_params 
        return updates

    def rmsprop(self,rho=0.9, epsilon = 1e-8, opt_params = None,grad_range=None):
        """
        RMSPROP Optimizer
        cost (to be minimized)
        params (list of parameters to take gradients with respect to)
        .... parameters specific to the optimization ...
        opt_params (if available, used to intialize the variables
        returns: update list of tuples, 
                 list of norms [0]: parameters [1]: gradients [2]: opt. params [3]: regularizer
                 opt_params: dictionary containing all the parameters for the optimization
        """
        updates = []
        #self.grads = self.get_grads()

        restartOpt = False
        if opt_params is None:
            restartOpt = True
            opt_params=OrderedDict()
        
        #No need to reload these
        g_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'g_norm',borrow=True)
        p_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'p_norm',borrow=True)
        opt_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'opt_norm',borrow=True)
        
        for p, g in zip(self.params,self.grads):
            if grad_range is not None:
                print '<<<<<< RMSPROP: Truncating Gradients in Range +-(',grad_range,') >>>>>>'
                g = T.clip(g,-grad_range, grad_range)
            
            if restartOpt:
                f_prev   = theano.shared(p.get_value()*0.,name = 'opt_fprev_'+p.name)
                r_prev   = theano.shared(p.get_value()*0.,name = 'opt_rprev_'+p.name)
                opt_params['opt_rprev_'+p.name] = r_prev
                opt_params['opt_fprev_'+p.name] = f_prev
            else:
                r_prev   = opt_params['opt_rprev_'+p.name]
                f_prev   = opt_params['opt_fprev_'+p.name]
            f_cur    = rho*f_prev+(1-rho)*g  
            r_cur    = rho*r_prev+(1-rho)*g**2
            updates.append((r_prev,r_cur))
            updates.append((f_prev,f_cur))
            
            lr_t = self.lr/T.sqrt(r_cur+f_cur**2+epsilon)
            p_t = p-(lr_t*g)
            updates.append((p,p_t))
            
            #Update norms
            g_norm += (g**2).sum()
            p_norm += (p**2).sum() 
            opt_norm+=(r_prev**2).sum()
        return updates

        
    def adagrad(self,epsilon=1e-8):
        #self.grads = self.get_grads()

        gaccums = [theano.shared(value=np.zeros(p.get_value().shape, dtype=config.floatX)) for p in self.params]

        updates = [(param, param - self.lr * gparam / T.sqrt(gaccum + T.square(gparam) + epsilon))
                    for param, gparam, gaccum in zip(self.params, self.grads, gaccums)]
                    
        updates += [(gaccum, gaccum + T.square(gparam)) for gaccum, gparam in zip(gaccums, self.grads)]

        return updates


    def adam2(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        t_prev = theano.shared(np.asarray(0.).astype(config.floatX))
        updates = OrderedDict()
        #self.grads = self.get_grads()

        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)

        t = t_prev + 1
        a_t = self.lr*T.sqrt(one-beta2**t)/(one-beta1**t)
        
        for param, g_t in zip(self.params, self.grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = beta1*m_prev + (one-beta1)*g_t
            v_t = beta2*v_prev + (one-beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates


'''def clip_norm(g, c, n):
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

    return g'''
