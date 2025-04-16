from torch import nn

class TrainableModel(nn.Module):
    def __init__(self):
        super(TrainableModel, self).__init__()
    
    def train_step(self, X, y=None) -> dict:
        raise NotImplementedError
    
    def val_step(self, X, y=None) -> dict:
        raise NotImplementedError
    
    def is_better(self, current_losses, best_losses) -> bool:
        return NotImplementedError