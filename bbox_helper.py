import torch
import numpy as np

def generate_prior_bboxes(prior_layer_cfg):
    priors_bboxes = []
    s_min = 0.2
    s_max = 0.9
    for feat_level_idx in range(0, len(prior_layer_cfg)):               # iterate each layers
        layer_cfg = prior_layer_cfg[feat_level_idx]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']
        # Compute S_{k}
        sk = s_min + (s_max - s_min) / (len(prior_layer_cfg) - 1) * feat_level_idx
        sk1 = s_min + (s_max - s_min) / (len(prior_layer_cfg) - 1) * (feat_level_idx + 1)
        fk = layer_feature_dim[0]
        for y in range(0, layer_feature_dim[0]):
            for x in range(0,layer_feature_dim[0]):
                # Compute bounding box center
                cx = (y + 0.5) / fk
                cy = (x + 0.5) / fk
                sk0 = np.sqrt(sk * sk1)
                h = sk0
                w = sk0
                priors_bboxes.append([cx, cy, w, h])
                # Generate prior bounding box with respect to the aspect ratio
                for aspect_ratio in layer_aspect_ratio[:-1]:
                    h = sk / np.sqrt(aspect_ratio)
                    w = sk * np.sqrt(aspect_ratio)
                    priors_bboxes.append([cx, cy, w, h])
    # Convert to Tensor
    priors_bboxes = torch.tensor(priors_bboxes)
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)
    assert priors_bboxes.dim() == 2
    assert priors_bboxes.shape[1] == 4
    return priors_bboxes

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    box_a = torch.tensor(box_a, dtype = torch.float64)
    box_b = torch.tensor(box_b, dtype = torch.float64)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]





def iou(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    area_a = torch.tensor(area_a, dtype = torch.float64)
    area_b = torch.tensor(area_b, dtype = torch.float64)
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match_priors(priors, truths, labels, threshold=0.3):
    overlaps = iou(center2corner(truths), center2corner(priors))
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # Refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    matches = torch.tensor(matches, dtype = torch.float32)
    priors = torch.tensor(priors, dtype = torch.float32)
    loc = bbox2loc(matches, priors)
    return loc, conf

def nms(boxes, scores, overlap=0.5, top_k=20, prob_threshold=0.34):
    boxes = center2corner(boxes).detach()
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        if scores[i] < prob_threshold:
            break
        keep[count] = i
        count += 1
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    assert priors.shape[0] == 1
    assert priors.dim() == 3
    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]
    # locations
    l_center = loc[..., :2]
    l_size = loc[..., 2:]
    # real bounding box
    return torch.cat([
        center_var * l_center * p_size + p_center,      # b_{center}
        p_size * torch.exp(size_var * l_size)           # b_{size}
    ], dim=-1)


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]
    # locations
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]
    return torch.cat([
        1 / center_var * ((b_center - p_center) / p_size),
        torch.log(b_size / p_size) / size_var
    ], dim=-1)


def center2corner(center):
    return torch.cat([center[..., :2] - center[..., 2:]/2.0,
                      center[..., :2] + center[..., 2:]/2.0], dim=1)


def corner2center(corner):
    return torch.cat([(corner[..., :2] + corner[..., 2:])/2.0,
                      corner[..., 2:] - corner[..., :2]], dim=1)