import numpy as np

class Loss:
    def loss(self, scores, y):
        loss, _ = self.backward(scores, y)
        return loss
    
    def backward(self, scores, y):
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
    
    
class SoftmaxForRNN:
    def backward(self, scores, y, mask=None):
        # init mask
        if mask is None: mask = (y != 0)
        
        # reshape data
        N, T, D = scores.shape
        
        scores = scores.reshape(N * T, D)
        y = y.reshape(N * T)
        mask = mask.reshape(N * T)
        
        # calculate loss
        probs = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        loss = -np.sum(mask * np.log(probs[np.arange(N * T), y] + 1e-10)) / mask.sum()
        
        # calculate gradient
        dscores = probs
        dscores[np.arange(N * T), y] -= 1
        dscores = mask[:, None] * dscores
        dscores = dscores / mask.sum()
        
        # reshape result
        dscores = dscores.reshape(N, T, D)
            
        # return
        return loss, dscores