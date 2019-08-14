
import os
import numpy as np
import random
from utils.kitti_dataset import KittiDataset
import utils.kitti_aug_utils as augUtils
import utils.kitti_bev_utils as bev_utils
import utils.config as cnf

import torch
import torch.nn.functional as F

import cv2

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class KittiYOLODataset(KittiDataset):

    def __init__(self, root_dir, split='train', mode ='TRAIN', folder=None, data_aug=True, multiscale=False):
        super().__init__(root_dir=root_dir, split=split, folder=folder)

        self.split = split
        self.multiscale = multiscale
        self.data_aug = data_aug
        self.img_size = cnf.BEV_WIDTH
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        self.sample_id_list = []

        if mode == 'TRAIN':
            self.preprocess_yolo_training_data()
        else:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

        print('Load %s samples from %s' % (mode, self.imageset_dir))
        print('Done: total %s samples %d' % (mode, len(self.sample_id_list)))

    def preprocess_yolo_training_data(self):
        """
        Discard samples which don't have current training class objects, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        for idx in range(0, self.num_samples):
            sample_id = int(self.image_idx_list[idx])
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)
            if not noObjectLabels:
                labels[:, 1:] = augUtils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_pc_range(labels[i, 1:4]) is True:
                        valid_list.append(labels[i,0])

            if len(valid_list):
                self.sample_id_list.append(sample_id)

    def check_pc_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def __getitem__(self, index):
        
        sample_id = int(self.sample_id_list[index])

        if self.mode in ['TRAIN', 'EVAL']:
            lidarData = self.get_lidar(sample_id)    
            objects = self.get_label(sample_id)   
            calib = self.get_calib(sample_id)

            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)
    
            if not noObjectLabels:
                labels[:, 1:] = augUtils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P)  # convert rect cam to velo cord

            if self.data_aug and self.mode == 'TRAIN':
                lidarData, labels[:, 1:] = augUtils.complex_yolo_pc_augmentation(lidarData, labels[:, 1:], True)

            b = bev_utils.removePoints(lidarData, cnf.boundary)
            rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            target = bev_utils.build_yolo_target(labels)
            img_file = os.path.join(self.image_path, '%06d.png' % sample_id)

            ntargets = 0
            for i, t in enumerate(target):
                if t.sum(0):
                    ntargets += 1            
            targets = torch.zeros((ntargets, 8))
            for i, t in enumerate(target):
                if t.sum(0):
                    targets[i, 1:] = torch.from_numpy(t)
            
            img = torch.from_numpy(rgb_map).type(torch.FloatTensor)

            if self.data_aug:
                if np.random.random() < 0.5:
                    img, targets = self.horisontal_flip(img, targets)

            return img_file, img, targets

        else:
            lidarData = self.get_lidar(sample_id)
            b = bev_utils.removePoints(lidarData, cnf.boundary)
            rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            img_file = os.path.join(self.image_path, '%06d.png' % sample_id)
            return img_file, rgb_map

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def horisontal_flip(self, images, targets):
        images = torch.flip(images, [-1])
        targets[:, 2] = 1 - targets[:, 2] # horizontal flip
        targets[:, 6] = - targets[:, 6] # yaw angle flip

        return images, targets

    def __len__(self):
        return len(self.sample_id_list)

class KittiYOLO2WayDataset(KittiDataset):
    def __init__(self, root_dir, split='sample', folder='sampledata'):
        super().__init__(root_dir=root_dir, split=split, folder=folder)

        self.sample_id_list = []

        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

        print('Load TESTING samples from %s' % self.imageset_dir)
        print('Done: total TESTING samples %d' % len(self.sample_id_list))

    def __getitem__(self, index):
        sample_id = int(self.sample_id_list[index])

        lidarData = self.get_lidar(sample_id)
        front_lidar = bev_utils.removePoints(lidarData, cnf.boundary)
        front_bev = bev_utils.makeBVFeature(front_lidar, cnf.DISCRETIZATION, cnf.boundary)

        back_lidar = bev_utils.removePoints(lidarData, cnf.boundary_back)
        back_bev = bev_utils.makeBVFeature(back_lidar, cnf.DISCRETIZATION, cnf.boundary_back)

        img_file = os.path.join(self.image_path, '%06d.png' % sample_id)

        return img_file, front_bev, back_bev
    
    def __len__(self):
        return len(self.sample_id_list)