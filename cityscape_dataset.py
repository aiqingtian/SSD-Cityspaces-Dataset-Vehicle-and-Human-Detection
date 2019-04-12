import numpy as np
import torch.nn
from torch.utils.data import Dataset
import cv2
import json
from bbox_helper import generate_prior_bboxes, match_priors
from augmentation_helper import SSDAugmentation


def get_bbox_label(json_dir):
    gt_bboxes = []
    gt_labels = []
    human_label = ['person', 'persongroup', 'rider']
    vehicle_label = ['car', 'cargroup', 'truck', 'bus', 'bicycle']
    with open(json_dir) as f:
        frame_info = json.load(f)
        for selected_object in frame_info['objects']:
            polygons = np.asarray(selected_object['polygon'])
            left_top = np.min(polygons, axis=0)
            right_bottom = np.max(polygons, axis=0)
            wh = right_bottom - left_top
            cx = left_top[0] + wh[0] / 2
            cy = left_top[1] + wh[1] / 2
            gt_bboxes.append([cx, cy, wh[0], wh[1]])
            if selected_object['label'] in human_label:
                gt_labels.append(1)
            elif selected_object['label'] in vehicle_label:
                gt_labels.append(2)
            else:
                gt_labels.append(0)
    assert len(gt_labels) == len(gt_labels)
    return gt_bboxes, gt_labels


def resize(img, bbox, w, h):
    w_ratio = float(w / img.size[0])
    h_ratio = float(h / img.size[1])
    img = img.resize((w, h))
    bbox[:, [0, 2]] *= w_ratio
    bbox[:, [1, 3]] *= h_ratio
    bbox[:, [0, 2]] /= w
    bbox[:, [1, 3]] /= h
    return img, bbox

class CityScapeDataset(Dataset):
    def __init__(self, img_dir_list, json_dir_list, transform, mode = 'train' ,augmentation_ratio = 50):
        self.transform = transform
        self.mode = mode
        # Implement prior bounding box
        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg=
                                                  [{'layer_name': '1', 'feature_dim_hw': (19, 19), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '2', 'feature_dim_hw': (10, 10), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '3', 'feature_dim_hw': (5, 5),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '4', 'feature_dim_hw': (3, 3),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '5', 'feature_dim_hw': (3, 3),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '6', 'feature_dim_hw': (1, 1),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
                                                   ])

        # Pre-process parameters:
        # Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127)).reshape(3, 1, 1)
        self.img_dir_list = img_dir_list
        self.json_dir_list = json_dir_list
        self.original_len = len(self.img_dir_list)
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, idx):
        img_dir = self.img_dir_list[idx]
        json_dir = self.json_dir_list[idx]
        sample_img = cv2.imread(img_dir, cv2.COLOR_BGR2RGB)
        gt_bboxes, gt_labels = get_bbox_label(json_dir)
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.int32)
        # data augmentation
        data_augmentation = SSDAugmentation(mode= self.mode)
        sample_img = np.array(sample_img, dtype=np.float64)
        sample_img, gt_bboxes, gt_labels = data_augmentation(sample_img, gt_bboxes, gt_labels)
        # Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, gt_bboxes, gt_labels)
        output_prior_bboxes = self.prior_bboxes
        return sample_img, bbox_tensor, bbox_label_tensor.long(), output_prior_bboxes
