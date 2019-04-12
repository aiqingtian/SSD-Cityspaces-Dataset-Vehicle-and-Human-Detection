import torch.nn as nn
import torch.nn.functional as F
import torch

def hard_negative_mining(predicted_prob, gt_label, neg_pos_ratio=3.0):
    pos_flag = gt_label > 0                                        # 0 = negative label
    # Sort the negative samples
    predicted_prob[pos_flag] = -1.0                                # temporarily remove positive by setting -1
    _, indices = predicted_prob.sort(dim=1, descending=True)       # sort by descend order, the positives are at the end
    _, orders = indices.sort(dim=1)                                # sort the negative samples by its original index

    # Remove the extra negative samples
    num_pos = pos_flag.sum(dim=1, keepdim=True)                     # compute the num. of positive examples
    num_neg = neg_pos_ratio * num_pos                               # determine of neg. examples, should < neg_pos_ratio
    neg_flag = orders < num_neg                                     # retain the first 'num_neg' negative samples index.

    return pos_flag, neg_flag

class MultiboxLoss(nn.Module):
    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3.0):
        super(MultiboxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_label_idx = 0

    def forward(self, confidence, pred_loc, gt_class_labels, gt_bbox_loc):
        with torch.no_grad():
            neg_class_prob = -F.log_softmax(confidence, dim=2)[:, :, self.neg_label_idx]      # select neg. class prob.
            pos_flag, neg_flag = hard_negative_mining(neg_class_prob, gt_class_labels, neg_pos_ratio=self.neg_pos_ratio)
            sel_flag = pos_flag | neg_flag
            num_pos = pos_flag.sum(dim=1, keepdim=True).float().sum()
        # Loss for the classification
        num_classes = confidence.shape[2]
        sel_conf = confidence[sel_flag]
        conf_loss = F.cross_entropy(sel_conf.view(-1, num_classes), gt_class_labels[sel_flag], reduction='elementwise_mean')
        # Implementation on bounding box regression
        pos_pred_loc = pred_loc[pos_flag, :]
        pos_gt_bbox_loc = gt_bbox_loc[pos_flag, :]
        loc_huber_loss = F.smooth_l1_loss(pos_pred_loc, pos_gt_bbox_loc, reduction='elementwise_mean')
        return conf_loss, loc_huber_loss