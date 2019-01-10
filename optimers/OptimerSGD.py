from utils import check_accuracy

                
class OptimerSGD:
    def __init__(self, hyperparams={}, 
                 print_every=100, silence=False,
                 check_val_acc=True, check_train_acc=True):
        
        self._base_init(hyperparams, print_every, silence, 
                        check_val_acc, check_train_acc)
    
        
    def train(self, model, dataloader):
        self._base_train(model, dataloader)
        
    
    def _base_init(self, hyperparams={}, 
                   print_every=100, silence=False,
                   check_val_acc=True, check_train_acc=True):
        
        self.silence = silence
        self.print_every = print_every
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
                    model.params[layer][key] = self._step(model.params[layer][key], 
                                                          model.dparams[layer][key],
                                                          layer, key)
            
            # record loss history
            if i % min(10, self.print_every) == 0:
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
                
            # check accuracy and decay learning rate.
            if i % iter_per_epoch == 0:
                self.learn_rate *= self.learn_rate_decay
    
    
    def _step(self, param, dparam, layer, key):
        return param - self.learn_rate * dparam
                    