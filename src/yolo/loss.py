import torch.nn as nn
import torch 

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        # predictions and target are of shape (batch_size, S, S, B*5 + C)
        assert not predictions.isnan().any(),  "There is NaN in prediction"
        batch_size = predictions.size(0)
        coord_loss = 0
        object_loss = 0
        no_object_loss = 0
        class_loss = 0

        for b in range(batch_size):
            for i in range(self.S):
                for j in range(self.S):
                    target_cell = target[b, i, j]
                    pred_cell = predictions[b, i, j]

                    if target_cell[4] == 1:  # If object is present in cell
                        # Coordinates loss
                        coord_loss += self.lambda_coord * (
                            (pred_cell[0] - target_cell[0])**2 +
                            (pred_cell[1] - target_cell[1])**2 +
                            (pred_cell[2]**0.5 - target_cell[2]**0.5)**2 +
                            (pred_cell[3]**0.5 - target_cell[3]**0.5)**2
                        )

                        # Object presence confidence loss
                        object_loss += (pred_cell[4] - target_cell[4])**2

                        # Class loss
                        class_loss += torch.sum((pred_cell[5:] - target_cell[5:])**2)

                    else:  # If no object is present
                        no_object_loss += self.lambda_noobj * (pred_cell[4] - target_cell[4])**2

        total_loss = coord_loss + object_loss + no_object_loss + class_loss
        return total_loss / batch_size
