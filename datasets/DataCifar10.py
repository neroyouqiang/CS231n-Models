import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from datasets.DataToy import DataToy

class DataCifar10(DataToy):
    def __init__(self, filedir, num_train=None, num_test=None, num_val=0, 
                 norm_dis_mean=False, norm_div_std=False, order_by='default'):
        """
        50000 training data, 10000 testing data
        32*32*3 uint8 for one data, 3kB for one data, 180MB for the whole dataset
        10 classes for output
        """
        x_train, y_train, x_test, y_test, label_names = DataCifar10.load_data(filedir)
        
        self.label_names = label_names
        
        self.x_train = x_train.transpose(0, 3, 1, 2)
        self.y_train = y_train
        self.x_test = x_test.transpose(0, 3, 1, 2)
        self.y_test = y_test
        
        if num_train is not None:
            if order_by == 'random':
                np.random.seed(10)
                idxs = np.random.choice(self.x_train.shape[0], num_train + num_val, replace=False)
            else:
                idxs = np.arange(num_train + num_val)
                
            self.x_train = self.x_train[idxs]
            self.y_train = self.y_train[idxs]
            
        if num_test is not None:
            if order_by == 'random':
                np.random.seed(11)
                idxs = np.random.choice(self.x_test.shape[0], num_test, replace=False)
            else:
                idxs = np.arange(num_test)
                
            self.x_test = self.x_test[idxs]
            self.y_test = self.y_test[idxs]
            
        # normalization and split validation set
        self._norm_train_val(num_train, num_val, norm_dis_mean, norm_div_std)
    
    
    def show_by_data(self, x, y=None):
        x = (x * self.norm_std + self.norm_mean).astype(np.uint8)
        
        
        name = None
        if y:
            name = self.label_names[y]
        
        plt.figure()
        if len(x.shape) == 2:
            plt.imshow(x, cmap='gray')
        elif x.shape[0] == 1:
            plt.imshow(x.transpose([1, 2, 0])[:, :, 0], cmap='gray')
        elif x.shape[0] == 3:
            plt.imshow(x.transpose([1, 2, 0]))
        else:
            plt.imshow(x)
            
        if name:
            plt.title(name)
            
        plt.axis('off')
        plt.show()
            
        
    def show_by_index(self, index, is_train=True):
        x = None
        y = None
        
        if is_train:
            x = (self.x_train[index] * self.norm_std + self.norm_mean).astype(np.uint8)
            y = self.y_train[index]
        else:
            x = (self.x_test[index] * self.norm_std + self.norm_mean).astype(np.uint8)
            y = self.y_test[index]
            
        self.show_by_data(x, y)
        
    
    @staticmethod
    def unpickle(filename):
        with open(filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict
    
    @staticmethod
    def load_label_names(filename):
        datadict = DataCifar10.unpickle(filename)
        names = datadict['label_names']
        return names
        
        
    @staticmethod
    def load_data_batch(filename):
        datadict = DataCifar10.unpickle(filename)
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        y = np.array(y)
        return x, y
    
    @staticmethod
    def load_data(filedir):
        xs = []
        ys = []
        for i in range(1, 6):
            x, y = DataCifar10.load_data_batch(os.path.join(filedir, 'data_batch_%d' % i))
            xs.append(x)
            ys.append(y)
        x_train = np.concatenate(xs)
        y_train = np.concatenate(ys)
        
        x_test, y_test = DataCifar10.load_data_batch(os.path.join(filedir, 'test_batch'))
        
        label_names = DataCifar10.load_label_names(os.path.join(filedir, 'batches.meta'))
        
        del x, y
        
        return x_train, y_train, x_test, y_test, label_names


if __name__ == '__main__':
    dataloader = DataCifar10('./cifar-10-batches-py', num_val=1000, num_train=5000, num_test=500)
#    label_names = dataloader.label_names
    print('Training data shape: ', dataloader.x_train.shape)
    print('Training labels shape: ', dataloader.y_train.shape)
    print('Validating data shape: ', dataloader.x_val.shape)
    print('Validating labels shape: ', dataloader.y_val.shape)
    print('Testing data shape: ', dataloader.x_test.shape)
    print('Testing labels shape: ', dataloader.y_test.shape)
    
    dataloader.show_by_index(1)