import numpy as np

from utils import check_accuracy

class ParamTuner:
    def __init__(self, modelclass, optimerclass, dataloader, silence=False):
        self.modelclass = modelclass
        self.optimerclass = optimerclass
        self.dataloader = dataloader
        self.silence = silence
    
    
    def tune(self, param_def, params_list, epoch=1):
        # init
        model_bst = None
        acc_bst = 0
        param_bst = dict(param_def)
        
        # begin to tune
        for _ in range(epoch):
            for key in params_list.keys():
                tune_name = key
                
                if isinstance(params_list[key], list):
                    # the values are given by list
                    tune_list = params_list[key]
                else:
                    # the values are given by min, max and num
                    num = params_list[key]['num']
                    minval = params_list[key]['min']
                    maxval = params_list[key]['max']
                    dtype = params_list[key]['dtype']
                    tune_list = (np.random.rand(num) * (maxval - minval) + minval).astype(dtype)
                    tune_list.sort()
                
                # tune for one parameter
                model_bst, param_bst, acc_bst = self.tune_one(model_bst, param_bst, acc_bst, tune_name, tune_list)
        
        # return
        return model_bst, param_bst, acc_bst
    
    
    def tune_one(self, model_def, param_def, acc_def, tune_name, tune_list):
        # init
        model_bst = model_def
        acc_bst = acc_def
        param_bst = dict(param_def)
        
        if not self.silence:
            print('Tune', tune_name, 'in', tune_list)
        
        # begin to tune
        for tune_value in tune_list:
            param_def[tune_name] = tune_value
            
            # train model
            model = self.modelclass(param_def)
            optimer = self.optimerclass(param_def, silence=True)
            
            optimer.train(model, self.dataloader)
            
            # test model
            scores = model.predict(self.dataloader.x_val)
            acc = check_accuracy(scores, self.dataloader.y_val)
            
            # choose best parameter
            if acc > acc_bst:
                acc_bst = acc
                param_bst = dict(param_def)
                model_bst = model
                print('With', param_def, 'accuracy:', acc, ' - Best!')
            else:
                print('With', param_def, 'accuracy:', acc)
            
        # return
        return model_bst, param_bst, acc_bst