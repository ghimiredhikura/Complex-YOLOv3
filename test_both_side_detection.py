import numpy as np
import math, os, argparse, time
import cv2
import torch

import utils.utils as utils
from models import *

import torch.utils.data as torch_data
from utils.kitti_yolo_dataset import KittiYOLO2WayDataset
import utils.kitti_bev_utils as bev_utils
import utils.kitti_utils as kitti_utils
import utils.mayavi_viewer as mview
import utils.config as cnf
from test_detection import predictions_to_kitti_format

def detect_and_draw(model, bev_maps, Tensor, is_front=True):

    # If back side bev, flip around vertical axis
    if not is_front:
        bev_maps = torch.flip(bev_maps, [2, 3])
    imgs = Variable(bev_maps.type(Tensor))

    # Get Detections
    img_detections = []
    with torch.no_grad():
        detections = model(imgs)
        detections = utils.non_max_suppression_rotated_bbox(detections, opt.conf_thres, opt.nms_thres)

    img_detections.extend(detections)

    # Only supports single batch
    display_bev = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
    
    bev_map = bev_maps[0].numpy()
    display_bev[:, :, 2] = bev_map[0, :, :]  # r_map
    display_bev[:, :, 1] = bev_map[1, :, :]  # g_map
    display_bev[:, :, 0] = bev_map[2, :, :]  # b_map

    display_bev *= 255
    display_bev = display_bev.astype(np.uint8)

    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        detections = utils.rescale_boxes(detections, opt.img_size, display_bev.shape[:2])
        for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(im, re)
            # Draw rotated box
            bev_utils.drawRotatedBox(display_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

    return display_bev, img_detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_epoch-298.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--save_video", type=bool, default=False, help="Set this flag to True if you want to record video")
    parser.add_argument("--split", type=str, default="test", help="text file having image lists in dataset")
    parser.add_argument("--folder", type=str, default="training", help="directory name that you downloaded all dataset")
    opt = parser.parse_args()
    print(opt)

    classes = utils.load_classes(opt.class_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))
    # Eval mode
    model.eval()

    dataset = KittiYOLO2WayDataset(cnf.root_dir, split=opt.split, folder=opt.folder)
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if opt.save_video:
        out = cv2.VideoWriter('bev_detection_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (opt.img_size*2, opt.img_size+375))

    start_time = time.time()
    for index, (img_paths, front_bevs, back_bevs) in enumerate(data_loader):

        front_bev_result, img_detections = detect_and_draw(model, front_bevs, Tensor, True)
        back_bev_result, _ = detect_and_draw(model, back_bevs, Tensor, False)

        end_time = time.time()
        print(f"FPS: {1.0/(end_time-start_time):.2f}")
        start_time = end_time

        front_bev_result = cv2.rotate(front_bev_result, cv2.ROTATE_90_CLOCKWISE)
        back_bev_result = cv2.rotate(back_bev_result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        vis = np.concatenate((front_bev_result, back_bev_result), axis=1)

        img2d = cv2.imread(img_paths[0])
        calib = kitti_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img2d.shape, opt.img_size)  
        img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)

        img2d = cv2.resize(img2d, (opt.img_size*2, 375))
        vis = np.concatenate((img2d, vis), axis=0)

        cv2.imshow('BEV_DETECTION_RESULT', vis)
        if opt.save_video:
            out.write(vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if opt.save_video:
        out.release()