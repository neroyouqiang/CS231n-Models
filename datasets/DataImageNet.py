import numpy as np
import os

from datasets.DataCifar10 import DataCifar10
#from DataCifar10 import DataCifar10


class DataImageNet(DataCifar10):
    def __init__(self, file_dir='./datasets/imagenet_val_25/', 
                 num_train=None, order_by='default'):
        x_train, y_train, class_names = DataImageNet._load_data(file_dir)
        
        self.x_train = x_train.transpose(0, 3, 1, 2)
        self.y_train = y_train
        self.label_names = class_names
        
        self.x_val = None
        self.y_val = None
        
        self.x_test = None
        self.y_test = None
        
        if num_train is not None:
            if order_by == 'random':
                np.random.seed(11)
                idxs = np.random.choice(self.x_train.shape[0], num_train, replace=False)
            else:
                idxs = np.arange(num_train)
                
            self.x_train = self.x_train[idxs]
            self.y_train = self.y_train[idxs]
        
        # normalization
        self._norm_max_min()
        
        # record test/train/val data numbers
        self._record_nums()
        
    
    def _load_data(file_dir):
        imagenet_fn = os.path.join(file_dir, 'imagenet_val_25.npz')
        if not os.path.isfile(imagenet_fn):
          print('file %s not found' % imagenet_fn)
          print('Run the following:')
          print('cd cs231n/datasets')
          print('bash get_imagenet_val.sh')
          assert False, 'Need to download imagenet_val_25.npz'
          
        f = np.load(imagenet_fn)
        X = f['X']
        y = f['y']
        class_names = f['label_map'].item()
            
        return X, y, class_names
    

if __name__ == '__main__':
    data_loader = DataImageNet()
    x = data_loader.x_train
    y = data_loader.y_train
    label_name = data_loader.label_name