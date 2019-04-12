import torch
import numpy as np
from numpy import random
from bbox_helper import center2corner, corner2center
import cv2

class Corner2Centre(object):
    def __call__(self, image, gt_bboxes, gt_labels):
        gt_bboxes = torch.tensor(gt_bboxes, dtype = torch.float64)
        gt_bboxes = corner2center(gt_bboxes)
        gt_bboxes = gt_bboxes.numpy()
        return image, gt_bboxes, gt_labels


class Centre2Corner(object):
    # Centre to Corner
    def __call__(self, image, gt_bboxes, gt_labels):
        gt_bboxes = torch.tensor(gt_bboxes, dtype = torch.float64)
        gt_bboxes = center2corner(gt_bboxes)
        gt_bboxes = gt_bboxes.numpy()
        return image, gt_bboxes, gt_labels

class Zoomout(object):
    def __init__(self):
        self.means = (127, 127, 127)
        self.ratio = np.random.randint(1, 5)

    def __call__(self, image, gt_bboxes, gt_labels):
        paseudo = np.random.randint(0, 10)
        if paseudo <8:
            return image, gt_bboxes, gt_labels
        height, width, channels = image.shape
        left = random.uniform(0, width* self.ratio - width)
        top = random.uniform(0, height* self.ratio- height)
        outputs = np.zeros((int(height* self.ratio), int(width* self.ratio), channels))
        outputs[:, :, :] = self.means
        outputs[int(top): int(top+ height), int(left): int(left+ width), :] = image
        image = outputs
        bboxes = np.asarray(gt_bboxes.copy())
        bboxes[:, 0] += int(left)
        bboxes[:, 1] += int(top)
        bboxes[:, 2] += int(left)
        bboxes[:, 3] += int(top)
        gt_bboxes = bboxes
        return image, gt_bboxes, gt_labels

def _RandomCrop(option, image, gt_bboxes, gt_labels):
    height, width, channels = image.shape
    choices = [0, 1, 2]
    _image = image
    w = option* height
    h = option* height
    choice = random.choice(choices)
    left = random.uniform(width - w)
    top = random.uniform(height - h)
    if choice == 0:
        left = 0
        top = 0
    elif choice ==1:
        left = width -w
        top = height -h
    reac = np.array([int(left), int(top), int(left + w), int(top + h)])
    gt_bboxes = np.array(gt_bboxes)
    _image = _image[reac[1]: reac[3], reac[0]: reac[2], :]
    centres = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
    mask_1 = (np.greater(centres[:, 0], reac[0]) * np.greater(centres[:, 1], reac[1]))
    mask_2 = (np.greater(reac[2], centres[:, 0]) * np.greater(reac[3], centres[:, 1]))
    m = mask_1 * mask_2
    _gt_bboxes = gt_bboxes[m, :].copy()
    _gt_bboxes = np.array(_gt_bboxes)
    gt_labels = np.array(gt_labels)
    _gt_labels = gt_labels[m]
    _gt_bboxes[:, :2] = np.maximum(_gt_bboxes[:, :2], reac[:2])
    _gt_bboxes[:, :2] -= reac[:2]
    _gt_bboxes[:, 2:] = np.minimum(_gt_bboxes[:, 2:], reac[2:])
    _gt_bboxes[:, 2:] -= reac[:2]
    return m, _image, _gt_bboxes, _gt_labels

class RandomCrop(object):
    def __init__(self):
        self.options = 1.0

    def __call__(self, image, gt_bboxes, gt_labels):
        _gt_bboxes = np.asarray(gt_bboxes)
        _gt_labels = np.asarray(gt_labels)
        _image = image
        for i in range(60):
            _sample = 1.0
            m, _image, _gt_bboxes, _gt_labels = _RandomCrop(_sample, image, gt_bboxes, gt_labels)
            if not m.any():
                continue
            if (np.any(_gt_bboxes) and np.any(_gt_labels)) == False:
                continue
            else:
                return _image, _gt_bboxes, _gt_labels
        return image, gt_bboxes, gt_labels

class RandomCrop_(object):
    def __init__(self):
        self.options = [0.1, 0.3, 0.5, 0.7, 0.9]

    def __call__(self, image, gt_bboxes, gt_labels):
        _gt_bboxes = np.asarray(gt_bboxes)
        _gt_labels = np.asarray(gt_labels)
        _image = image
        for i in range(100):
            _sample = random.choice(self.options)
            m, _image, _gt_bboxes,_gt_labels = _RandomCrop(_sample, image,gt_bboxes, gt_labels)
            if not m.any():
                continue
            if (np.any(_gt_labels)) == False:
                continue
            else:
                return _image, _gt_bboxes,_gt_labels
        return image, gt_bboxes, gt_labels

class RandomHorizontalFlip(object):
    def __call__(self, image, gt_bboxes, gt_labels):
        height, width, channels = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            gt_bboxes = np.asarray(gt_bboxes.copy())
            gt_bboxes[:, 0::2] = width - gt_bboxes[:, 2::-2]
        return image, gt_bboxes, gt_labels

class Resize(object):
    def __init__(self, w=300, h=300):
        self.w = w
        self.h = h

    def __call__(self, image, gt_bboxes, gt_labels):
        w_ratio = float(self.w / image.shape[1])
        h_ratio = float(self.h / image.shape[0])
        _image = cv2.resize(image, (self.h, self.w))
        _gt_bboxes = np.array(gt_bboxes.copy())
        _gt_bboxes[:, [0, 2]] *= w_ratio
        _gt_bboxes[:, [1, 3]] *= h_ratio
        _gt_bboxes[:, [0, 2]] /= self.w
        _gt_bboxes[:, [1, 3]] /= self.h
        return _image, _gt_bboxes, gt_labels


class Normalization(object):
    def __init__(self):
        self.mean = np.asarray((127, 127, 127)).reshape(1, 1, 3)
        self.std = 128.0

    def __call__(self, image, gt_bboxes, gt_labels):
        image = (image - self.mean) / self.std
        return image, gt_bboxes, gt_labels


class ToTensor(object):
    def __call__(self, image, gt_bboxes, gt_labels):
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image)
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float64)
        gt_labels = torch.tensor(gt_labels)
        return image, gt_bboxes, gt_labels

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, gt_bboxes= None, gt_labels= None):
        for i in self.transforms:
            image, gt_bboxes, gt_labels = i(image, gt_bboxes, gt_labels)
        return image, gt_bboxes, gt_labels


class SSDAugmentation(object):
    def __init__(self, mode):
        self.mode = mode
        self.augmentation = Compose(
            [Centre2Corner(),
             RandomCrop(),
             Zoomout(),
             RandomHorizontalFlip(),
             Resize(),
             Normalization(),
             Corner2Centre(),
             ToTensor(),
             ])
        self.test_augmentation = Compose(
            [Centre2Corner(),
             RandomCrop(),
             Resize(),
             Normalization(),
             Corner2Centre(),
             ToTensor(),
            ]
            )

    def __call__(self, image, gt_bboxes, gt_labels):
        if self.mode =="train":
            return self.augmentation(image, gt_bboxes, gt_labels)
        else:
            return self.test_augmentation(image, gt_bboxes, gt_labels)




