import numpy as np
import matplotlib.pyplot as plt
import psutil

def check_accuracy(scores, y):
    """
    Check the accuracy of results
    """
    num_correct = np.sum(np.argmax(scores, axis=1) == y)
    return float(num_correct) / len(y)


def check_rel_error(m1, m2):
    return np.max(np.abs((m1 - m2) / (m1 + m2) * 2.))


def show_pc_memory(style='full'):
    pc_mem = psutil.virtual_memory()
    
    gb_factor = 1024.0 ** 3
    
    if style == 'full':
        print("total memory: %fGB" % float(pc_mem.total / gb_factor))
        print("available memory: %fGB" % float(pc_mem.available / gb_factor))
        print("used memory: %fGB" % float(pc_mem.used / gb_factor))
        print("percent of used memory: %f" % float(pc_mem.percent))
        print("free memory: %fGB" % float(pc_mem.free / gb_factor))
    else:
        print("free memory: %fGB / %fGB" % (float(pc_mem.free / gb_factor), float(pc_mem.available / gb_factor)))


def show_weight_images(w, size, label_names=None):
    # show weight image
    w = w.reshape(size, size, 3, -1)
    w_min, w_max = np.min(w), np.max(w)
    
    for i in range(w.shape[3]):
          
        # Rescale the weights
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        
        # plot
        if i % 5 == 0:
            plt.figure()
        plt.subplot(1, 5, i % 5 + 1)
        plt.imshow(wimg.astype('uint8'))
        if label_names is not None:
            plt.title(label_names[i])
        plt.axis('off')
    
    plt.show()
   
    
def show_training_info(optimer):
    # loss curve
    plt.figure()
    plt.plot(optimer.loss_history)
    plt.title('Loss')
    plt.show()
    
    # accuracy
    if len(optimer.acc_train_history) > 0 or len(optimer.acc_val_history) > 0:
        plt.figure()
        plt.plot(optimer.acc_train_history)
        plt.plot(optimer.acc_val_history)
        plt.legend(['Training accuracy', 'Validation accuracy'])
        plt.title('Accuracy')
        plt.show()
    
    
def show_training_infos(optimers, legends=None, contains=['loss', 'train', 'val']):
    # loss curve
    if 'loss' in contains:
        plt.figure()
        for optimer in optimers:
            plt.plot(optimer.loss_history)
        if legends:
            plt.legend(legends)
        plt.title('Loss')
        plt.show()
    
    # training accuracy
    if 'train' in contains:
        plt.figure()
        for optimer in optimers:
            plt.plot(optimer.acc_train_history)
        if legends:
            plt.legend(legends)
        plt.title('Training accuracy')
        plt.show()
    
    # validation accuracy
    if 'val' in contains:
        plt.figure()
        for optimer in optimers:
            plt.plot(optimer.acc_val_history)
        if legends:
            plt.legend(legends)
        plt.title('Validation accuracy')
        plt.show()
    

def show_weight_trace(optimer):
    if optimer.param_trace is not None:
        data = np.array(optimer.param_trace)
        plt.figure()
        plt.plot(data[:, 0], data[:, 1])
        plt.show()
        
    
def show_weight_traces(optimers, legends=None):
    plt.figure()
    for optimer in optimers:
        if optimer.param_trace is not None:
            data = np.array(optimer.param_trace)
            plt.scatter(data[:, 0], data[:, 1], marker='.')
    if legends:
        plt.legend(legends)
    plt.show()
        
        
def check_gradient(model, x, y, h=0.00001):
    print('Layer | Key | Numerical gradient | Calculated gradient | Relative error')
    for layer in range(len(model.params)):
        for key in model.params[layer]:
            if key not in ['cache', 'info']:
                idx = []
                for d in model.params[layer][key].shape:
                    idx.append(np.random.randint(d))
                idx = tuple(idx)
                
                # fix the seed
                seed = np.random.randint(10000)
                
                # calculated gradient
                model.backward(x, y, mode='train', seed=seed)
                grad_cal = model.dparams[layer][key][idx]
                
                # loss +
                model.params[layer][key][idx] += h
                loss1 = model.backward(x, y, mode='train', seed=seed)
                
                # loss -
                model.params[layer][key][idx] -= 2 * h
                loss2 = model.backward(x, y, mode='train', seed=seed)
                # recover
                model.params[layer][key][idx] += h
                
                # numerical gradient 
                grad_num = (loss1 - loss2) / (2 * h)
                
                # print result
                print(type(model.layers[layer]), key, grad_num, grad_cal, 
                      check_rel_error(grad_num, grad_cal))
    