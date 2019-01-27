import numpy as np

class DataToy:
    def __init__(self, num_train=None, num_test=None, num_val=0, 
                 norm_dis_mean=False, norm_div_std=False):
        """
        Generating toy data by function. 
        2*1 for one data. 3 classes for output.
        """
        self.x_train = np.random.rand(num_train + num_val, 1)
        self.y_train = self._func(self.x_train)
        self.x_test = np.random.rand(num_test, 1)
        self.y_test = self._func(self.x_test)
        
        # normalization and split validation set
        self._norm_train_val(num_train, num_val, norm_dis_mean, norm_div_std)
        
        # record test/train/val data numbers
        self._record_nums()
    
    
    def _norm_train_val(self, num_train, num_val, norm_dis_mean, norm_div_std):
        """
        Split data into training and validating data.
        Nomalization.
        """
        self.x_val = None
        self.y_val = None
        self.norm_mean = 0
        self.norm_std = 1
        
        idxs = np.arange(num_train, num_train + num_val)
        
        self.x_val = self.x_train[idxs]
        self.y_val = self.y_train[idxs]
        
        self.x_train = np.delete(self.x_train, idxs, axis=0)
        self.y_train = np.delete(self.y_train, idxs, axis=0)
        
        if norm_dis_mean:
            self.norm_mean = np.mean(self.x_train, axis=0)
            self.x_train = self.x_train - self.norm_mean
            self.x_val = self.x_val - self.norm_mean
            self.x_test = self.x_test - self.norm_mean
            
        if norm_div_std:
            self.norm_std = np.std(self.x_train, axis=0)
            self.x_train = self.x_train / self.norm_std
            self.x_val = self.x_val / self.norm_std
            self.x_test = self.x_test / self.norm_std
            
    
    def _record_nums(self):
        """
        Record the numbers of different datasets.
        """
        if self.x_train is not None:
            self.num_train = self.x_train.shape[0]
        else:
            self.num_train = 0
            
        if self.x_val is not None:
            self.num_val = self.x_val.shape[0]
        else:
            self.num_val = 0
        
        if self.x_test is not None:
            self.num_test = self.x_test.shape[0]
        else:
            self.num_test = 0
    
        
    def _func(self, x):
        y = (x[:, 0] - x[:, 0] * x[:, 0]) * 4
        y[y > 0.5] = 1
        y[y <= 0.5] = 0
        return y.astype(np.int)
    
            
    def get_batch(self, num, seed=None):
        """
        Get a minibatch of data.
        """
        if seed: 
            np.random.seed(seed)
        else:
            np.random.seed()
            
        idxs = np.random.choice(self.x_train.shape[0], num, replace=False)
        
        return self.x_train[idxs], self.y_train[idxs]
    
    def show_info(self):
        if self.x_train is not None: print('Training data shape: ', self.x_train.shape)
        if self.y_train is not None: print('Training labels shape: ', self.y_train.shape)
        if self.x_val is not None: print('Validating data shape: ', self.x_val.shape)
        if self.y_val is not None: print('Validating labels shape: ', self.y_val.shape)
        if self.x_test is not None: print('Testing data shape: ', self.x_test.shape)
        if self.y_test is not None: print('Testing labels shape: ', self.y_test.shape)
    
    
if __name__ == '__main__':
    dataloader = DataToy(num_val=1000, num_train=5000, num_test=500)
#    label_names = dataloader.label_names
    print('Training data shape: ', dataloader.x_train.shape)
    print('Training labels shape: ', dataloader.y_train.shape)
    print('Validating data shape: ', dataloader.x_val.shape)
    print('Validating labels shape: ', dataloader.y_val.shape)
    print('Testing data shape: ', dataloader.x_test.shape)
    print('Testing labels shape: ', dataloader.y_test.shape)
    
    x_train = dataloader.x_train
    y_train = dataloader.y_train