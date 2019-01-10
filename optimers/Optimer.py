class Optimer:
    def __init__(self, hyperparams=None, silence=False):
        self.silence = silence
        
        
    def train(self, model, dataloader):
        if hasattr(model, 'train'):
            model.train(dataloader.x_train, dataloader.y_train)