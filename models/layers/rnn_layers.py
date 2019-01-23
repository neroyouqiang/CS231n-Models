import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from models.layers import Layer

class RNN:
    def __init__(self, num_input, num_hidden, init_scale=None):
        self.num_input = num_input
        self.num_hidden = num_hidden
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input)
            
    def forward(self, x, h0, param):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        N, T, D = x.shape
        N, H = h0.shape
        
        h = np.zeros([N, T + 1, H])
        h[:, 0, :] = h0
        
        for t in range(0, T, 1):
            h[:, t + 1, :], _ = self._step_forward(x[:, t, :], h[:, t, :], None, Wx, Wh, b)
            
        hs = h[:, 1: T + 1, :]
        
        return hs, (x, h, Wx, Wh, b)
    
    def forward_step(self, x, c0, h0, param):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        
        h1, c1 = self._step_forward(x, h0, c0, Wx, Wh, b)
        
        return h1, c1
        
    
    def backward(self, dh, cache):
        x, h, Wx, Wh, b = cache
        N, T, D = x.shape
        N, _, H = h.shape
        
        dx = np.zeros([N, T, D])
        dh0 = np.zeros([N, H])
        dWx = np.zeros([D, H])
        dWh = np.zeros([H, H])
        db = np.zeros(H)
        
        for t in range(T - 1, -1, -1):
            cache = (x[:, t, :], h[:, t, :], h[:, t + 1, :], Wx, Wh, b)
            dx[:, t, :], dh0, _, dWxt, dWht, dbt = self._step_backward(dh0 + dh[:, t, :], None, cache)
            dWx += dWxt
            dWh += dWht
            db += dbt
            
        return dx, dh0, {'Wx': dWx, 'Wh': dWh, 'b': db}
           
    def _step_forward(self, x, h0, c0, Wx, Wh, b):
        h1 = np.dot(x, Wx) + np.dot(h0, Wh) + b
        h1 = np.tanh(h1)
        c1 = c0
        
        return h1, c1
            
    def _step_backward(self, dh1, dc1, cache, active='tanh'):
        x, h0, h1, Wx, Wh, b = cache
        
        dh1 = dh1 * (1 - h1 ** 2)
        dc0 = dc1
        
        dh0 = np.dot(dh1, Wh.T)
        dx = np.dot(dh1, Wx.T)
        
        dWx = np.dot(x.T, dh1)
        dWh = np.dot(h0.T, dh1)
        db = np.sum(dh1, axis=0)
        
        return dx, dh0, dc0, dWx, dWh, db
    
    
    def get_init_param(self):
        Wx = np.random.randn(self.num_input, self.num_hidden) * self.init_scale
        Wh = np.random.randn(self.num_hidden, self.num_hidden) * self.init_scale
        b = np.zeros(self.num_hidden)
        
        return {'Wx': Wx, 'Wh': Wh, 'b': b,
                'info': {'num_input': self.num_input, 
                         'num_output': self.num_hidden,
                         'init_scale': self.init_scale}}
        

class LSTM(Layer):
    def __init__(self, num_input, num_hidden, init_scale=None, device='cpu'):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.device = device
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input)
            
    def forward(self, x, h0, param):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        N, T, D = x.shape
        N, H = h0.shape
        
        # Use numpy on CPU
        if self.device == 'cpu':
            # to numpy
            if type(x) != np.ndarray: x = x.asnumpy()
            if type(h0) != np.ndarray: h0 = h0.asnumpy()
            
            h = np.zeros([N, T + 1, H])
            c = np.zeros([N, T + 1, H])
            
            i = np.zeros([N, T, H])
            f = np.zeros([N, T, H])
            o = np.zeros([N, T, H])
            g = np.zeros([N, T, H])
            
            h[:, 0, :] = h0
            for t in range(0, T, 1):
                h[:, t + 1, :], c[:, t + 1, :], cache = self._step_forward(x[:, t, :], h[:, t, :], c[:, t, :], Wx, Wh, b)
                i[:, t, :], f[:, t, :], o[:, t, :], g[:, t, :] = cache
            
            hs = h[:, 1: T + 1, :]
        
        # Use mxnet.ndarray on GPU or CPU
        elif self.device == 'gpu' or self.device == '':
            if self.device == 'gpu':
                dvc = mx.gpu()
            else:
                dvc = mx.cpu()
            
            # to ndarray
            if type(x) != nd.ndarray.NDArray: x = nd.array(x, dvc)
            if type(h0) != nd.ndarray.NDArray: h0 = nd.array(h0, dvc)
            
            h = nd.zeros([N, T + 1, H], dvc)
            c = nd.zeros([N, T + 1, H], dvc)
            
            i = nd.zeros([N, T, H], dvc)
            f = nd.zeros([N, T, H], dvc)
            o = nd.zeros([N, T, H], dvc)
            g = nd.zeros([N, T, H], dvc)
            
            h[:, 0, :] = h0
            for t in range(0, T, 1):
                h[:, t + 1, :], c[:, t + 1, :], cache = self._step_forward_mx(x[:, t, :], h[:, t, :], c[:, t, :], Wx, Wh, b, dvc)
                i[:, t, :], f[:, t, :], o[:, t, :], g[:, t, :] = cache
                
            hs = h[:, 1: T + 1, :]
                
            # to numpy
            hs.wait_to_read()
#            hs = hs.asnumpy()
            
        # return
        return hs, (x, h, c, i, f, o, g, Wx, Wh, b)
    
    def backward(self, dh, cache):
        x, h, c, i, f, o, g, Wx, Wh, b = cache
        N, T, D = x.shape
        N, _, H = h.shape
        
        # Use numpy on CPU
        if self.device == 'cpu':
            # to numpy
            if type(dh) == nd.ndarray.NDArray: dh = dh.asnumpy()
            
            # function
            dx = np.zeros([N, T, D])
            dh0 = np.zeros([N, H])
            dc0 = np.zeros([N, H])
            dWx = np.zeros([D, H * 4])
            dWh = np.zeros([H, H * 4])
            db = np.zeros(H * 4)
            
            for t in range(T - 1, -1, -1):
                cache = (x[:, t, :], h[:, t, :], h[:, t + 1, :], c[:, t, :], c[:, t + 1, :], 
                         i[:, t, :], f[:, t, :], o[:, t, :], g[:, t, :], Wx, Wh, b)
                dx[:, t, :], dh0, dc0, dWxt, dWht, dbt = self._step_backward(dh0 + dh[:, t, :], dc0, cache)
                dWx += dWxt
                dWh += dWht
                db += dbt
                
        # Use mxnet.ndarray on GPU
        elif self.device == 'gpu' or self.device == '':
            if self.device == 'gpu':
                dvc = mx.gpu()
            else:
                dvc = mx.cpu()
                
            # to ndarray
            if type(dh) == np.ndarray: dh = nd.array(dh, dvc)
            
            # function
            dx = nd.zeros([N, T, D], dvc)
            dh0 = nd.zeros([N, H], dvc)
            dc0 = nd.zeros([N, H], dvc)
            
            dWx = nd.zeros([D, H * 4], dvc)
            dWh = nd.zeros([H, H * 4], dvc)
            db = nd.zeros(H * 4, dvc)
            
            for t in range(T - 1, -1, -1):
                cache = (x[:, t, :], h[:, t, :], h[:, t + 1, :], c[:, t, :], c[:, t + 1, :], 
                         i[:, t, :], f[:, t, :], o[:, t, :], g[:, t, :], Wx, Wh, b)
                dx[:, t, :], dh0, dc0, dWxt, dWht, dbt = self._step_backward_mx(dh0 + dh[:, t, :], dc0, cache, dvc)
                dWx += dWxt
                dWh += dWht
                db += dbt
                
            # to numpy
            dx.wait_to_read()
            dh0.wait_to_read()
#            dx, dh0 = dx.asnumpy(), dh0.asnumpy()
            
        return dx, dh0, {'Wx': dWx, 'Wh': dWh, 'b': db}
    
    def forward_step(self, x, h0, c0, param):
        Wx, Wh, b = param['Wx'], param['Wh'], param['b']
        
        # Use numpy on CPU
        if self.device == 'cpu':
            # to numpy
            if type(x) == nd.ndarray.NDArray: x = x.asnumpy()
            if type(h0) == nd.ndarray.NDArray: h0 = h0.asnumpy()
            if type(c0) == nd.ndarray.NDArray: c0 = c0.asnumpy()
            
            # function
            h1, c1, cache = self._step_forward(x, h0, c0, Wx, Wh, b)
            
        # Use mxnet.ndarray on GPU
        elif self.device == 'gpu' or self.device == '':
            if self.device == 'gpu':
                dvc = mx.gpu()
            else:
                dvc = mx.cpu()
                
            # to ndarray
            if type(x) == np.ndarray: x = nd.array(x, dvc)
            if type(h0) == np.ndarray: h0 = nd.array(h0, dvc)
            if type(c0) == np.ndarray: c0 = nd.array(c0, dvc)
            
            # function
            h1, c1, cache = self._step_forward_mx(x, h0, c0, Wx, Wh, b, dvc)
            
            # to numpy
            h1.wait_to_read()
            c1.wait_to_read()
        
        return h1, c1
    
    
    def _step_forward(self, x, h0, c0, Wx, Wh, b):
        N, D = x.shape
        N, H = h0.shape
        
        y = np.dot(x, Wx) + np.dot(h0, Wh) + b
        i = y[:, 0: H]
        f = y[:, H: H * 2]
        o = y[:, H * 2: H * 3]
        g = y[:, H * 3: H * 4]
        
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        o = self.sigmoid(o)
        g = np.tanh(g)
        
        c1 = f * c0 + i * g
        h1 = o * np.tanh(c1)
        
        return h1, c1, (i, f, o, g)
    
    
    def _step_forward_mx(self, x, h0, c0, Wx, Wh, b, device):
        N, D = x.shape
        N, H = h0.shape
        
        y = nd.dot(x, Wx) + nd.dot(h0, Wh) + b
        i = y[:, 0: H]
        f = y[:, H: H * 2]
        o = y[:, H * 2: H * 3]
        g = y[:, H * 3: H * 4]
        
        i = nd.sigmoid(i)
        f = nd.sigmoid(f)
        o = nd.sigmoid(o)
        g = nd.tanh(g)
        
        c1 = f * c0 + i * g
        h1 = o * nd.tanh(c1)
        
        return h1, c1, (i, f, o, g)
            
    
    def _step_backward(self, dh1, dc1, cache):
        x, h0, h1, c0, c1, i, f, o, g, Wx, Wh, b = cache
        
        c1_tanh = np.tanh(c1)
        
        do = dh1 * c1_tanh
        dc1 = dh1 * o * (1 - c1_tanh ** 2) + dc1
        
        dc0 = dc1 * f
        df = dc1 * c0
        di = dc1 * g
        dg = dc1 * i
        
        di = di * i * (1 - i)
        df = df * f * (1 - f)
        do = do * o * (1 - o)
        dg = dg * (1 - g ** 2)
        
        dy = np.concatenate((di, df, do, dg), axis=1)
        db = np.sum(dy, axis=0)
        dWx = np.dot(x.T, dy)
        dWh = np.dot(h0.T, dy)
        
        dx = np.dot(dy, Wx.T)
        dh0 = np.dot(dy, Wh.T)
        
        return dx, dh0, dc0, dWx, dWh, db
    
    
    def _step_backward_mx(self, dh1, dc1, cache, device):
        x, h0, h1, c0, c1, i, f, o, g, Wx, Wh, b = cache
        
        c1_tanh = nd.tanh(c1)
        
        do = dh1 * c1_tanh
        dc1 = dh1 * o * (1 - c1_tanh ** 2) + dc1
        
        dc0 = dc1 * f
        df = dc1 * c0
        di = dc1 * g
        dg = dc1 * i
        
        di = di * i * (1 - i)
        df = df * f * (1 - f)
        do = do * o * (1 - o)
        dg = dg * (1 - g ** 2)
        
        dy = nd.concatenate([di, df, do, dg], axis=1)
        db = nd.sum(dy, axis=0)
        dWx = nd.dot(x.T, dy)
        dWh = nd.dot(h0.T, dy)
        
        dx = nd.dot(dy, Wx.T)
        dh0 = nd.dot(dy, Wh.T)
        
        return dx, dh0, dc0, dWx, dWh, db
    
            
    def sigmoid(self, x):
        """
        A numerically stable version of the logistic sigmoid function.
        """
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    def _get_init_param(self):
        Wx = np.random.randn(self.num_input, self.num_hidden * 4) * self.init_scale
        Wh = np.random.randn(self.num_hidden, self.num_hidden * 4) * self.init_scale
        b = np.zeros(self.num_hidden * 4)
        
        return {'Wx': Wx, 'Wh': Wh, 'b': b}
#                'info': {'num_input': self.num_input, 
#                         'num_output': self.num_hidden,
#                         'init_scale': self.init_scale, 
#                         'device': self.device}}
        

class LinearForRNN(Layer):
    def __init__(self, num_input, num_output, init_scale=None, device='cpu'):
        self.num_input = num_input
        self.num_output = num_output
        self.device = device
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 1. / np.sqrt(self.num_input)
    
    def _forward_train(self, x, param):
        W, b = param['W'], param['b']
        N, T, D = x.shape
                    
        x = x.reshape(N * T, D)
        y = np.dot(x, W) + b
        y = y.reshape(N, T, -1)
        
        cache = (x, W)
        
        return y, cache
    
    def _forward_test(self, x, param):
        W, b = param['W'], param['b']
        N, D = x.shape
                
        x = x.reshape(N, D)
        y = np.dot(x, W) + b
        y = y.reshape(N, -1)
        
        cache = (x, W)
        
        return y, cache
    
    def _forward_train_mx(self, x, param, device):
        W, b = param['W'], param['b']
        N, T, D = x.shape
                    
        x = x.reshape(N * T, D)
        y = nd.dot(x, W) + b
        y = y.reshape(N, T, -1)
        
        cache = (x, W)
        
        return y, cache
    
    def _forward_test_mx(self, x, param, device):
        W, b = param['W'], param['b']
        N, D = x.shape
                
        x = x.reshape(N, D)
        y = nd.dot(x, W) + b
        y = y.reshape(N, -1)
        
        cache = (x, W)
        
        return y, cache
    
    def _backward(self, dy, cache):
        x, W = cache
        N, T, M = dy.shape
        
        dy = dy.reshape(N * T, M)
            
        db = np.sum(dy, axis=0)
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)
            
        dx = dx.reshape(N, T, -1)
        
        dparam = {'W': dW, 'b': db}
            
        return dx, dparam
    
    def _backward_mx(self, dy, cache, device):
        x, W = cache
        N, T, M = dy.shape
        
        dy = dy.reshape(N * T, M)
            
        db = nd.sum(dy, axis=0)
        dW = nd.dot(x.T, dy)
        dx = nd.dot(dy, W.T)
            
        dx = dx.reshape(N, T, -1)
        
        dparam = {'W': dW, 'b': db}
            
        return dx, dparam
    
    def _get_init_param(self):
        W = np.random.randn(self.num_input, self.num_output) * self.init_scale
        b = np.zeros(self.num_output)
        
        return {'W': W, 'b': b}
#                'info': {'num_input': self.num_input, 
#                         'num_output': self.num_output,
#                         'init_scale': self.init_scale}}
        

class WordEmbedding(Layer):
    def __init__(self, num_word, num_vector, init_scale=None, device='cpu'):
        self.num_word = num_word
        self.num_vector = num_vector
        self.device = device
        if init_scale:
            self.init_scale = init_scale
        else:
            self.init_scale = 0.01
            
    
    def _forward(self, x, param):
        W_embed = param['W_embed']
        
        y = W_embed[x]
        
        cache = (x, W_embed.shape)
        
        return y, cache
    
    def _forward_mx(self, x, param, device):
        return self._forward_train(x, param)
    
    def _backward(self, dy, cache):
        x, W_shape = cache
        
        dW_embed = np.zeros(W_shape)
        np.add.at(dW_embed, x, dy)
        
        dx = dy
        dparam = {'W_embed': dW_embed}
        
        return dx, dparam
    
    def _backward_mx(self, dy, cache, device):
        x, W_shape = cache
        
        N, T = x.shape
        M, V = W_shape
        
        dx = dy
        
#        dW_embed = nd.zeros(W_shape, device)
#        for n in range(N):
#            for t in range(T):
#                dW_embed[x[n, t]] += dy[n, t]
                
        x = x.asnumpy().astype(np.int32)
        dy = dy.asnumpy()
        
        dW_embed = np.zeros(W_shape)
        np.add.at(dW_embed, x, dy)
        
        dW_embed = nd.array(dW_embed, device)
        
        dparam = {'W_embed': dW_embed}
        
        return dx, dparam
            
    def _get_init_param(self):
        W_embed = np.random.randn(self.num_word, self.num_vector) * self.init_scale
    
        return {'W_embed': W_embed}
#                'info': {'num_word': self.num_word, 
#                         'num_vector': self.num_vector,
#                         'init_scale': self.init_scale}}
        
        
            
            