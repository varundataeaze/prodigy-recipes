import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys

import yaml
from pprint import pprint


def get_prediction(input_image):
    results =[]
    sys.path.append("/home/labeling/code/prodigy-recipes/image/image_object_detection")    
    sys.path.append("/home/labeling/code/prodigy-recipes/image/image_object_detection/library/yolov7")
    
    # pprint(sys.path)

    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
    from utils.datasets import letterbox
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)

    source, weights, view_img, save_txt, imgsz, trace = cfg["source"],cfg["weights"], cfg["view_img"],cfg["save_txt"], cfg["imgsz"] ,cfg["trace"]

    # Initialize
    device = select_device(cfg["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA
    conf_thres= cfg["conf_thres"]
    iou_thres= cfg["iou_thres"]
    classes= cfg["classes"]
    agnostic_nms= cfg["agnostic_nms"]
    augment= cfg["augment"]
    save_conf= cfg["save_conf"]

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16
    # Set Dataloader

    # dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    # t0 = time.time()
    # for path, img, im0s, vid_cap in dataset:
    img = letterbox(input_image, imgsz, stride=cfg["stride"])[0]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.permute(2, 0, 1)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=augment)[0]

    # #     # Inference
    # #     t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    # #     t2 = time_synchronized()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    # #     t3 = time_synchronized()

    for i, det in enumerate(pred):  # detections per image
        
        s, im0 ='', input_image
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                label = f'{names[int(cls)]} {conf:.2f}'
                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                results+=[{"xyxy":xyxy,"cls":cls,"conf":conf}]
    return results

        