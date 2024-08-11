import torch

class EarlyStopping:
    def __init__(self, patience, min_delta) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping = False
        self.best_score = None
        self.counter = 0

    def __call__(self, loss, model, epoch):
        score = - loss
        if self.best_score is None:
            self.best_score = score 
            self.save_checkpoint(model, epoch)
        elif self.best_score + self.min_delta <= score:
            self.best_score = score
            self.save_checkpoint(model, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stopping = True

    def save_checkpoint(self, model, epoch):
        torch.save(model.state_dict(), f"./dump/checkpoint/checkpoint_{epoch}.pt")