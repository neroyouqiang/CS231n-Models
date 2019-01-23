#from utils import check_accuracy, show_pc_memory
import mxnet.ndarray as nd

                
class OptimerSGD:
    def __init__(self, hyperparams={}, 
                 print_every=100, record_every=10, silence=False,
                 check_val_acc=True, check_train_acc=True):
        
        self._base_init(hyperparams, print_every, record_every, 
                        silence, check_val_acc, check_train_acc)
    
        
    def train(self, model, dataloader):
        self._base_train(model, dataloader)
        
    
    def _base_init(self, hyperparams={}, 
                   print_every=100, record_every=10, silence=False,
                   check_val_acc=True, check_train_acc=True):
        
        self.silence = silence
        self.print_every = print_every
        self.record_every = record_every
        self.check_val_acc = check_val_acc
        self.check_train_acc = check_train_acc
        
        self.loss_history = []
        self.acc_train_history = []
        self.acc_val_history = []
        
        hyperparams.setdefault('learn_rate', 1e-5)
        hyperparams.setdefault('learn_rate_decay', 0.95)
        hyperparams.setdefault('num_iters', 100)
        hyperparams.setdefault('batch_size', 200)
        
        self.learn_rate = hyperparams['learn_rate']
        self.learn_rate_decay = hyperparams['learn_rate_decay']
        self.num_iters = hyperparams['num_iters']
        self.batch_size = hyperparams['batch_size']
        
        
    def _base_train(self, model, dataloader):
        # init the records
        self.loss_history = []
        self.acc_train_history = []
        self.acc_val_history = []
        self.param_trace = []
        
        iter_per_epoch = max(dataloader.num_train / self.batch_size, 1)
        
        # for every iteration
        for i in range(self.num_iters):
            x, y = dataloader.get_batch(self.batch_size)
            loss = model.backward(x, y)
            
            for layer in range(len(model.dparams)):
                for key in model.dparams[layer].keys():
                    param = model.params[layer][key]
                    dparam = model.dparams[layer][key]
                    
                    # to numpy
                    if type(param) == nd.ndarray.NDArray: 
                        param = param.asnumpy()
                    if type(dparam) == nd.ndarray.NDArray: 
                        dparam = dparam.asnumpy()
                    
                    # modify
                    param = self._step(param, dparam, layer, key)
                     
                    # to ndarray
                    if type(model.params[layer][key]) == nd.ndarray.NDArray:
                        param = nd.array(param, model.params[layer][key].context)
                    
                    model.params[layer][key] = param
            
            # record loss history
            if i % self.record_every == 0:
                self.loss_history.append(loss)
#                self.param_trace.append(model.params[0]['W'][0, 0: 2])
            
            # print training info
            if i % self.print_every == 0:
                if not self.silence: print(i, '/', self.num_iters, 'loss is', loss)
                
                if self.check_train_acc:
                    acc_train = check_accuracy(model.predict(x), y)
                    self.acc_train_history.append(acc_train)
                    
                if self.check_val_acc:
                    acc_val = check_accuracy(model.predict(dataloader.x_val), dataloader.y_val)
                    self.acc_val_history.append(acc_val)
                
#                # show memory
#                show_pc_memory(style='easy')
                
            # check accuracy and decay learning rate.
            if i % iter_per_epoch == 0:
                self.learn_rate *= self.learn_rate_decay
    
    
    def _step(self, param, dparam, layer, key):
        return param - self.learn_rate * dparam
    
    def _step_mx(self, param, dparam, layer, key, device):
        return self._step(param, dparam, layer, key)
                    