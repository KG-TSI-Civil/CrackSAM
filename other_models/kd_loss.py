import torch.nn as nn
import torch.nn.functional as F


def calc_loss(outputs, label, ce_loss, dice_loss, dice_weight:float=0.8):
    #print(label_batch.shape) # bs, 112,112
    loss_ce = ce_loss(outputs, label[:].long())  
    loss_dice = dice_loss(outputs, label, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def calc_soft_loss(outputs, soft_label, temperature:float=3.0):

    # Channel-wise distilation

    # outputs bs,2,448,448
    # label bs,2,448,448

    bs,C,W,H = outputs.shape
    
    soft_label_softmax = F.softmax(soft_label.view(bs,C,W*H)/temperature,dim=-1)
    outputs_softmax = F.softmax(outputs.view(bs,C,W*H)/temperature,dim=-1)

    loss_ce = nn.KLDivLoss(reduction='sum')(outputs_softmax.log(),soft_label_softmax)

    return loss_ce*(temperature**2)/C/bs

