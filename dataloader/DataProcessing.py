import copy

import torch
import numpy as np
import cv2 as cv
import pycocotools
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils


def mapper(record, img_path):
    """
    不做数据增强处理
    Args:
        record:
        img_path:

    Returns:

    """
    record = copy.deepcopy(record)
    image = record["image"]
    image = torch.from_numpy(image.transpose(2, 0, 1))
    return {
        "image": image,
        "id": img_path,
        "height": record["image"].shape[0],
        "width": record["image"].shape[1],
        "per_class": record["per_class"],
        "instances": utils.annotations_to_instances(record["annotations"], image.shape[1:], mask_format="bitmask")
    }


def transformToTrainRawFormat(rgb, semantic_instance, semantic_classes):
    """
    将传入的图像转换为程序训练用API可用的训练格式，方便后续转换为自由训练可用格式
    Args:
        rgb: 投影后的RGB图像
        semantic_instance: 每个通道保存一个实例的蒙板
        semantic_classes: 每个元素对应semantic_instance通道的实例的类别

    Returns:

    """

    record = {"image": rgb}

    objs = []
    num = semantic_instance.shape[0]
    indices = []
    for i in range(num):
        sem_mask = semantic_instance[i]
        sem_mask = sem_mask.astype(np.uint8)
        sem_class = semantic_classes[i]

        if sem_mask.sum() <= 200:
            continue

        contours, _ = cv.findContours(sem_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        merged_contours = np.concatenate(contours)
        x_all = []
        y_all = []
        for point in merged_contours.squeeze(axis=1):
            x_all.append(point[0])
            y_all.append(point[1])

        obj = {
            "bbox": [np.min(x_all), np.min(y_all), np.max(x_all), np.max(y_all)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": pycocotools.mask.encode(np.asarray(sem_mask, order="F")),
            "category_id": sem_class,
        }
        objs.append(obj)

        indices.append(i)
    record["annotations"] = objs
    record["per_class"] = [semantic_classes[k] for k in indices]
    return record


def labeledForTrain(labeled_data: list[dict]):
    dataset_dict_raw = []

    for labeled in labeled_data:
        semantic_classes = labeled["per_class"]
        record_raw = transformToTrainRawFormat(labeled["image"], labeled["instance_mask"], semantic_classes)
        dataset_dict_raw.append(mapper(record_raw, labeled["id"]))

    return dataset_dict_raw


def instancesForTrain(instances: list[list], unlabeled_data: list[dict]):
    dataset_dict_raw = []
    for unlabeled, img_instances in zip(unlabeled_data, instances):
        num = len(img_instances)
        semantic_classes = []
        semantic_instance = np.zeros((num, unlabeled["image"].shape[0], unlabeled["image"].shape[1]))

        for i in range(num):
            semantic_instance[i] = img_instances[i].mask * 1.
            semantic_classes.append(img_instances[i].class_id)
        record_raw = transformToTrainRawFormat(unlabeled["image"], semantic_instance, semantic_classes)
        dataset_dict_raw.append(mapper(record_raw, unlabeled["id"]))

    return dataset_dict_raw


def unlabeledForPseudoLabel(data: list[dict]):
    """
    将原始图标签的图像转换为可以用于模型推理的tensor数据格式
    Args:
        data: 列表中的元素是字典，每个字典保存原始的BGR图像及其对应的深度图

    Returns: list[dict]: 列表中元素是字典，每个字典的key为“image”，对应的值是尺寸为(3,480,640)的图像

    """
    data_batch = []
    for img_dict in data:
        image = img_dict["image"]
        data_dict = {
            "image": torch.from_numpy(image.astype("float32").transpose(2, 0, 1)),
            "height": img_dict["image"].shape[0], "width": img_dict["image"].shape[1]}
        data_batch.append(data_dict)
    return data_batch
