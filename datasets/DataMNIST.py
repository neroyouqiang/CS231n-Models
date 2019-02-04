import struct
import numpy as np
import os

from datasets.DataCifar10 import DataCifar10
#from DataCifar10 import DataCifar10


class DataMNIST(DataCifar10):
    def __init__(self, file_dir='./datasets/mnist_data/', 
                 num_train=None, num_val=0, num_test=None, order_by='random'):
        """
        60000 training data, 10000 testing data
        28*28*1 uint8 for one data, 0.8kB for one data, 56MB for the whole dataset
        10 classes for output
        """
        x_train, y_train, x_test, y_test = DataMNIST.load_data(file_dir)
        
        self.label_names = None
    
        self.x_train = x_train.reshape(-1, 1, 28, 28)
        self.y_train = y_train
        self.x_test = x_test.reshape(-1, 1, 28, 28)
        self.y_test = y_test
        
        if num_train is not None:
            if order_by == 'random':
#                np.random.seed(10)
                idxs = np.random.choice(self.x_train.shape[0], num_train + num_val, replace=False)
            else:
                idxs = np.arange(num_train + num_val)
                
            self.x_train = self.x_train[idxs]
            self.y_train = self.y_train[idxs]
        else:
            num_train = self.x_train.shape[0]
            
        if num_test is not None:
            if order_by == 'random':
#                np.random.seed(11)
                idxs = np.random.choice(self.x_test.shape[0], num_test, replace=False)
            else:
                idxs = np.arange(num_test)
                
            self.x_test = self.x_test[idxs]
            self.y_test = self.y_test[idxs]
        else:
            num_test = self.x_test.shape[0]
            
        # split validation set
        self._split_train_val(num_train, num_val)
        
        # normalization
        self._norm_max_min(m='-1_1')
        
        # record test/train/val data numbers
        self._record_nums()
    
    
    @staticmethod
    def load_data(file_dir):
        x_train = DataMNIST.read_images(os.path.join(file_dir, 'train-images-idx3-ubyte'))
        y_train = DataMNIST.read_labels(os.path.join(file_dir, 'train-labels-idx1-ubyte'))
        
        x_test = DataMNIST.read_images(os.path.join(file_dir, 't10k-images-idx3-ubyte'))
        y_test = DataMNIST.read_labels(os.path.join(file_dir, 't10k-labels-idx1-ubyte'))
        
        return x_train, y_train, x_test, y_test
    
        
    @staticmethod
    def read_images(filename):
        # get file buffer
        f = open(filename, 'rb')
        buf = f.read()
        f.close()
    
        # init index
        index = 0
    
        # get head info
        magic, num, rows, columns = struct.unpack_from('>IIII', buf, index)
        index += struct.calcsize('>IIII')
    
        # mydata
        data = np.zeros([num, rows, columns])
    
        # get image mydata
        for ii in range(num):
            for x in range(rows):
                for y in range(columns):
                    data[ii, x, y] = int(struct.unpack_from('>B', buf, index)[0])
                    index += struct.calcsize('>B')
    
        # return
        return data
    
    
    @staticmethod
    def read_labels(filename):
        # get file buffer
        f = open(filename, 'rb')
        buf = f.read()
        f.close()
    
        # init index
        index = 0
    
        # get head info
        magic, num = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')
    
        # mydata
        data = np.zeros([num])
    
        # get label mydata
        for ii in range(num):
            data[ii] = int(struct.unpack_from('>B', buf, index)[0])
            index += struct.calcsize('>B')
    
        # return
        return data


if __name__ == '__main__':
    dataloader = DataMNIST('./datasets/mnist_data/') #, num_val=1000, num_train=5000, num_test=500)
    dataloader.show_info()
    
#    dataloader.show_by_index(1)
