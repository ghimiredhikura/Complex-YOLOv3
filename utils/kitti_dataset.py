from __future__ import division
import os
import glob
import numpy as np
import cv2
import torch.utils.data as torch_data
import utils.kitti_utils as kitti_utils

class KittiDataset(torch_data.Dataset):

    def __init__(self, root_dir, split='train', folder='testing'):
        self.split = split

        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', folder)
        self.lidar_path = os.path.join(self.imageset_dir, "velodyne")

        self.image_path = os.path.join(self.imageset_dir, "image_2")
        self.calib_path = os.path.join(self.imageset_dir, "calib")
        self.label_path = os.path.join(self.imageset_dir, "label_2")

        if not is_test:
            split_dir = os.path.join('data', 'KITTI', 'ImageSets', split+'.txt')
            self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        else:
            self.files = sorted(glob.glob("%s/*.bin" % self.lidar_path))
            self.image_idx_list = [os.path.split(x)[1].split(".")[0].strip() for x in self.files]
            print(self.image_idx_list[0])

        self.num_samples = self.image_idx_list.__len__()

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file) # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_path, '%06d.png' % idx)
        assert  os.path.exists(img_file)
        img = cv2.imread(img_file)
        width, height, channel = img.shape
        return width, height, channel

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return kitti_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_path, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.read_label(label_file)

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented