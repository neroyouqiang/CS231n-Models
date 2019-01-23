#import minpy.numpy as np
import numpy as np

class KNN:
    def __init__(self, hyperparams={}):
        
        if 'K' in hyperparams:
            self.k = hyperparams['K']
        else:
            self.k = 1e-5
            
        self.dists = None
    
    def train(self, x, y):
        self.x_train = x.reshape([x.shape[0], -1])
        self.y_train = y
        self.num_class = np.max(y) + 1
    
    
    def predict(self, x):
        # distance
        dists = self.distance(x)
        
        # nearest data ids
        idxs = np.argsort(dists, axis=1)[:, 0: self.k]
        
        # socres
        scores = np.zeros([x.shape[0], self.num_class])
        for i in range(x.shape[0]):
            count = np.bincount(self.y_train[idxs][i])
            scores[i, 0: count.shape[0]] = count
        
        # return
        return scores
        
        
    def distance(self, x):
        x = x.reshape([x.shape[0], -1])
        x2 = np.sum(np.square(x), axis=1, keepdims=True)
        x2_train = np.sum(np.square(self.x_train), axis=1, keepdims=False)
        return x2 - 2 * np.dot(x, self.x_train.T) + x2_train
    

if __name__ == '__main__':
    model = KNN(hyperparams={'K': 1})
    model.train(np.array([[1,2,3], [2,3,4], [1,2,4]]), np.array([0, 1, 0]))
    
    scores = model.predict(np.array([[1,2,4]]))