import os, json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import urllib.request, urllib.error, urllib.parse, tempfile
import imageio
import ssl
#from scipy.misc import imread

#from DataToy import DataToy
from datasets.DataToy import DataToy


class DataCocoFeat(DataToy):
    """
    About 80,000 images and 400,000 captions in the training set.
    About 40,000 images and 200,000 captions in the validation set.
    """
    def __init__(self, file_dir='./coco_captioning/', pca_features=False, 
                 num_train=None, num_val=None, order_by='default'):
        
        self.data = DataCocoFeat._load_data(file_dir, pca_features)
        
        self.x_train = self.data['train_features'][self.data['train_image_idxs']]
        self.y_train = self.data['train_captions']
        
        self.x_val = self.data['val_features'][self.data['val_image_idxs']]
        self.y_val = self.data['val_captions']
        
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
            
        if num_val is not None:
            if order_by == 'random':
                np.random.seed(11)
                idxs = np.random.choice(self.x_train.shape[0], num_val, replace=False)
            else:
                idxs = np.arange(num_val)
                
            self.x_val = self.x_val[idxs]
            self.y_val = self.y_val[idxs]
        
        # record test/train/val data numbers
        self._record_nums()
        
        
    def show_by_url(self, url, caption):
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # image
            f = urllib.request.urlopen(url)
            _, fname = tempfile.mkstemp()
            with open(fname, 'wb') as ff:
                ff.write(f.read())
            img = imageio.imread(fname)
#            img = imread(fname)
            os.remove(fname)
            
            # caption
            caption_str = self.decode_captions(caption)
            
            # show image
            plt.figure()
            plt.imshow(img)
            plt.title(caption_str)
            plt.axis('off')
            plt.show()
        
            # return
            return img, caption
            
        except urllib.error.URLError as e:
            print('URL Error: ', e.reason, url)
        except urllib.error.HTTPError as e:
            print('HTTP Error: ', e.code, url)
        
        
    def show_by_index(self, index, caption=None, data_type='train'):
        if data_type == 'train':
            image_idx = self.data['train_image_idxs'][index]
            url = self.data['train_urls'][image_idx]
            if caption is None:
                caption = self.data['train_captions'][index]
        else:
            image_idx = self.data['val_image_idxs'][index]
            url = self.data['val_urls'][image_idx]
            if caption is None:
                caption = self.data['val_captions'][index]
            
        return self.show_by_url(url, caption)


    def decode_captions(self, captions):
        idx_to_word = self.data['idx_to_word']
        
        singleton = False
        if captions.ndim == 1:
            singleton = True
            captions = captions[None]
            
        decoded = []
        N, T = captions.shape
        for i in range(N):
            words = []
            for t in range(T):
                word = idx_to_word[captions[i, t]]
                if word != '<NULL>':
                    words.append(word)
                if word == '<END>':
                    break
            decoded.append(' '.join(words))
            
        if singleton:
            decoded = decoded[0]
            
        return decoded
    
    def show_info(self):
        print('Training data shape: ', self.x_train.shape)
        print('Training labels shape: ', self.y_train.shape)
        print('Validation data shape: ', self.x_val.shape)
        print('Validation labels shape: ', self.y_val.shape)
        print('Number of training images: ', len(self.data['train_urls']))
        print('Number of validation images: ', len(self.data['val_urls']))
        print('Number of words: ', len(self.data['idx_to_word']))
    
    
    @staticmethod
    def _load_data(file_dir, pca_features=False):
        data = {}
        
        # ['train_captions', 'train_image_idxs', 'val_captions', 'val_image_idxs']
        caption_file = os.path.join(file_dir, 'coco2014_captions.h5')
        with h5py.File(caption_file, 'r') as f: 
            for k, v in f.items():
                data[k] = np.asarray(v)
        
        # ['train_features', 'val_features']
        if pca_features:
            train_feat_file = os.path.join(file_dir, 'train2014_vgg16_fc7_pca.h5')
        else:
            train_feat_file = os.path.join(file_dir, 'train2014_vgg16_fc7.h5')
        with h5py.File(train_feat_file, 'r') as f:
            data['train_features'] = np.asarray(f['features'])
    
        if pca_features:
            val_feat_file = os.path.join(file_dir, 'val2014_vgg16_fc7_pca.h5')
        else:
            val_feat_file = os.path.join(file_dir, 'val2014_vgg16_fc7.h5')
        with h5py.File(val_feat_file, 'r') as f:
            data['val_features'] = np.asarray(f['features'])
        
        # ['train_urls', 'val_urls']
        train_url_file = os.path.join(file_dir, 'train2014_urls.txt')
        with open(train_url_file, 'r') as f:
            data['train_urls'] = np.asarray([line.strip() for line in f])
    
        val_url_file = os.path.join(file_dir, 'val2014_urls.txt')
        with open(val_url_file, 'r') as f:
            data['val_urls'] = np.asarray([line.strip() for line in f])
            
        # ['idx_to_word'], ['word_to_idx']
        dict_file = os.path.join(file_dir, 'coco2014_vocab.json')
        with open(dict_file, 'r') as f:
            dict_data = json.load(f)
            for k, v in dict_data.items():
                data[k] = v
            
        # return
        return data


if __name__ == '__main__':
    dataloader = DataCocoFeat('./coco_captioning/', pca_features=True)
    img, caption = dataloader.show_by_index(5)