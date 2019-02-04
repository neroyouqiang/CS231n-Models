import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

from datasets.DataToy import DataToy
#from DataToy import DataToy

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
        self._norm_mean_std(norm_dis_mean, norm_div_std)
        
        # record test/train/val data numbers
        self._record_nums()
    
    
    def show_by_data(self, xs, ys=None, maxflen=5, cmap=None):
        flen = min(len(xs), maxflen)
        
        for i in range(len(xs)):
            name = None
            if ys is not None:
                y = ys[i]
                if self.label_names is not None:
                    name = self.label_names[y]
                
            x = xs[i]
            
            if cmap != plt.cm.hot:
                if self.norm_type == 'mean_std':
                    x = (x * self.norm_std + self.norm_mean).astype(np.uint8)
                elif self.norm_type == 'max_min_0_1':
                    x = (x * (self.norm_max - self.norm_min) + self.norm_min).astype(np.uint8)
                elif self.norm_type == 'max_min_-1_1':
                    x = (x + 1) / 2
                    x = (x * (self.norm_max - self.norm_min) + self.norm_min).astype(np.uint8)
            
            if i % flen == 0:
                plt.figure()
                plt.gcf().tight_layout()
                
            plt.subplot(1, flen, i % flen + 1)
            
            if len(x.shape) == 2:
                if cmap is None: 
                    cmap = 'gray'
                plt.imshow(x, cmap=cmap)
            elif x.shape[0] == 1:
                if cmap is None: 
                    cmap = 'gray'
                plt.imshow(x.transpose([1, 2, 0])[:, :, 0], cmap=cmap)
            elif x.shape[0] == 3:
                if cmap is None: 
                    plt.imshow(x.transpose([1, 2, 0]))
                else:
                    plt.imshow(x.transpose([1, 2, 0]), cmap=cmap)
            else:
                if cmap is None: 
                    plt.imshow(x)
                else:
                    plt.imshow(x, cmap=cmap)
                
            if name:
                plt.title(name)
                
            plt.axis('off')
        plt.show()
        
#        return x, name
            
        
    def show_by_index(self, index, data_type='train'):
        if data_type == 'train':
            x = (self.x_train[index] * self.norm_std + self.norm_mean).astype(np.uint8)
            y = self.y_train[index]
        elif data_type == 'test':
            x = (self.x_test[index] * self.norm_std + self.norm_mean).astype(np.uint8)
            y = self.y_test[index]
        elif data_type == 'val':
            x = (self.x_val[index] * self.norm_std + self.norm_mean).astype(np.uint8)
            y = self.y_val[index]
            
        return self.show_by_data([x], [y])
        
    
    @staticmethod
    def unpickle(file_name):
        with open(file_name, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict
    
    @staticmethod
    def load_label_names(file_name):
        datadict = DataCifar10.unpickle(file_name)
        names = datadict['label_names']
        return names
        
        
    @staticmethod
    def load_data_batch(file_name):
        datadict = DataCifar10.unpickle(file_name)
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        y = np.array(y)
        return x, y
    
    @staticmethod
    def load_data(file_dir):
        xs = []
        ys = []
        for i in range(1, 6):
            x, y = DataCifar10.load_data_batch(os.path.join(file_dir, 'data_batch_%d' % i))
            xs.append(x)
            ys.append(y)
        x_train = np.concatenate(xs)
        y_train = np.concatenate(ys)
        
        x_test, y_test = DataCifar10.load_data_batch(os.path.join(file_dir, 'test_batch'))
        
        label_names = DataCifar10.load_label_names(os.path.join(file_dir, 'batches.meta'))
        
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