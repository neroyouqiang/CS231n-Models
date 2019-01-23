#import minpy.numpy as np
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

class Loss:
    """
    Parent loss class. The loss can be used on both CPU and GPU.
    """
    def __init__(self, reg=0, device='cpu'):
        self.device = device
        self.reg = reg
        
    def loss(self, scores, y):
        loss, _ = self.backward(scores, y)
        return loss
    
    def backward(self, scores, y):
        if self.device == 'cpu':
            # to numpy
            if type(scores) == nd.ndarray.NDArray: 
                scores = scores.asnumpy()
                
            if type(y) == nd.ndarray.NDArray: 
                y = y.asnumpy()
                
            loss, dscores = self._backward(scores, y)
        
        elif self.device == '' or self.device == 'gpu':
            if self.device == 'gpu':
                dvc = mx.gpu()
            else:
                dvc = mx.cpu()
                
            # to ndarray
            if type(scores) == np.ndarray: 
                scores = nd.array(scores, dvc)
                
            if type(y) == np.ndarray: 
                y = nd.array(y, dvc)
                
            loss, dscores = self._backward_mx(scores, y, dvc)
            
            # to numpy
            loss.wait_to_read()
#            loss = loss.asnumpy()
            dscores.wait_to_read()
#            dscores = dscores.asnumpy()
            
        # return
        return loss, dscores
    
    def add_reg(self, loss, params, dparams):
        for i in range(len(params)):
            if 'W' in params[i]:
                dparams[i]['W'] += self.reg * params[i]['W']
                
                if self.device == 'cpu':
                    loss += self.reg * np.sum(np.square(params[i]['W'])) / 2.
                    
                elif self.device == '' or self.device == 'gpu':
                    loss += self.reg * nd.sum(nd.square(params[i]['W'])) / 2.
                    
        if self.device == '' or self.device == 'gpu':
            loss.wait_to_read()
        
        return loss, dparams
    
    def _backward(self, scores, y):
        loss = None
        dscores = None
        return loss, dscores
    
    def _backward_mx(self, scores, y, dvc):
        loss = None
        dscores = None
        return loss, dscores
    

class Softmax:
    def backward(self, scores, y):
        N = scores.shape[0]
        
        # calculate loss
        probs = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        loss = -np.sum(np.log(probs[np.arange(N), y] + 1e-10)) / N
        
        # calculate gradient
        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores = dscores / N
            
        # return
        return loss, dscores
    
    
class SVM:
    def backward(self, scores, y):
        N = scores.shape[0]
        
        # calculate loss
        margins = scores - scores[np.arange(N), y].reshape(-1, 1) + 1
        margins[margins < 0] = 0
        margins[np.arange(N), y] = 0
        loss = np.sum(margins) / N
    
        # calculate gradient
        dscores = margins
        dscores[dscores > 0] = 1.
        dscores[np.arange(N), y] = -np.sum(dscores, axis=1)
        dscores = dscores / N
        
        # return
        return loss, dscores
    
    
class MSE:
    def backward(self, scores, y):
        N = scores.shape[0]
    
        # calculate gradient
        if scores.shape == y.shape:
            dscores = scores - y
        else:
            dscores[np.arange(N), y] -= 1
            
        dscores = dscores / N
        
        # calculate loss
        loss = np.sum(dscores) / N / 2.
        
        # return
        return loss, dscores
        
    
class SoftmaxForRNN(Loss):
    def _backward(self, scores, y):
        mask = (y != 0)
        
        # reshape data
        N, T, D = scores.shape
        
        scores = scores.reshape(N * T, D)
        y = y.reshape(N * T)
        mask = mask.reshape(N * T)
        
        # calculate loss
        probs = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        loss = -np.sum(mask * np.log(probs[np.arange(N * T), y] + 1e-10)) / N # mask.sum()
        
        # calculate gradient
        dscores = probs
        dscores[np.arange(N * T), y] -= 1
        dscores = mask[:, None] * dscores
        dscores = dscores / N # mask.sum()
        
        # reshape result
        dscores = dscores.reshape(N, T, D)
            
        # return
        return loss, dscores
    
    def _backward_mx(self, scores, y, device):
        mask = (y != 0)
        
        # reshape data
        N, T, D = scores.shape
        
        scores = scores.reshape(N * T, D)
        y = y.reshape(N * T)
        mask = mask.reshape(N * T)
        
        # calculate loss
        probs = scores - nd.max(scores, axis=1, keepdims=True)
        probs = nd.exp(probs)
        probs = probs / nd.sum(probs, axis=1, keepdims=True)
        loss = -nd.sum(mask * nd.log(probs[nd.arange(N * T, ctx=device), y] + 1e-10)) / N # mask.sum()
        
        # calculate gradient
        dscores = probs
        dscores[nd.arange(N * T, ctx=device), y] -= 1
        dscores = mask.reshape(-1, 1) * dscores
        dscores = dscores / N # mask.sum()
        
        # reshape result
        dscores = dscores.reshape(N, T, D)
            
        # return
        return loss, dscores