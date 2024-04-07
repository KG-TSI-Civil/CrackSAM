import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target) 
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    try: 
        pred.max() <=1 and pred.min()>=0 and gt.max() <=1 and gt.min()>=0
    except:
        print("Please Ensure max(pred) <=1 and min(pred)>=0 and max(gt) <=1 and min(gt)>=0 !!!")

    A = pred.sum() 
    B = gt.sum() 

    if A > 0 and B > 0:

        AinterB = pred&gt
        AunionB = (pred | gt) 
        AinterB = AinterB.sum()
        AunionB = AunionB.sum()
        dice = 2*AinterB/(A+B) 
        iou = AinterB /AunionB
        precision= AinterB/A
        recall = AinterB/B
        
        return precision, recall, dice, iou
    elif A >0 and B ==0.0: # For non-crack images
        return 0.0,0.0,0.0,0.0
    elif A == 0.0 and B == 0.0: # For non-crack images
        return 1.0,1.0,1.0,1.0
    elif A ==0.0 and B >0:
        return 0.0,0.0,0.0,0.0



def test_single_volume(image, label, net, classes, multimask_output, patch_size=[448, 448], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    image, label = image.cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() # image: 1,c,h,w label: h,w
    if len(image.shape) == 4:
        prediction = np.zeros_like(label)
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (1,1,patch_size[0] / x, patch_size[1] / y), order=3)  # ndarray   
        inputs = torch.from_numpy(image).float().cuda() 
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0]) # inputs 1,c,h,w
            output_masks = outputs['masks']  # 1,2,h,w
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0) # h,w
            prediction = out.cpu().detach().numpy()# h,w  
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0) 

    metric_test=calculate_metric_percase(prediction , label)  # ndarray

    if test_save_path is not None:
        # image: 1,c,h,w  ndarray
        image = image*255
        label = label*255
        prediction = prediction*255
        image = Image.fromarray(np.transpose(image.squeeze(0), (1, 2, 0)).astype(np.uint8)) # 1,c,h,w -> h,w,c
        image.save(test_save_path + '/img/' + case + "_img.jpg")
        # pred h,w   ndarray
        pred = Image.fromarray(prediction.astype(np.uint8)) # h,w 
        pred.save(test_save_path + '/pred/' + case + "_img.jpg")
        # label h,w  ndarray
        label = Image.fromarray(label.astype(np.uint8)) # h,w 
        label.save(test_save_path + '/gt/' + case + "_img.jpg")

    return metric_test
