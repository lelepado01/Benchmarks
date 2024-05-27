

import torch

class ChannelReconstructionLoss(torch.nn.Module):
    def __init__(self, layer=0):
        super(ChannelReconstructionLoss, self).__init__()
        self.layer = layer
        
        self.mse_loss = torch.nn.MSELoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()
        
        p_layer = pred[self.layer]
        t_layer = target[self.layer]

        # remove the reconstructed channel
        pred = torch.cat([pred[:self.layer], pred[self.layer+1:]])
        target = torch.cat([target[:self.layer], target[self.layer+1:]])
        
        return self.mse_loss(p_layer, t_layer) + self.cross_entropy(pred, target)
        