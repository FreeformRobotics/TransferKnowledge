import numpy as np
import cv2 as cv
import torch
from detectron2.structures import BoxMode
import pycocotools


def whetherMerge(box1: np.ndarray, box2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou_box = intersection / float(box1_area + box2_area - intersection)

    mask_intersection = np.sum(mask1 * mask2)
    mask1_area = np.sum(mask1)
    mask2_area = np.sum(mask2)

    iou_mask = 0.0
    if float(mask1_area + mask2_area - mask_intersection) > 0:
        iou_mask = mask_intersection / float(mask1_area + mask2_area - mask_intersection)

    flag = False
    if (iou_box > 0.70) or (iou_mask > 0.70):
        flag = True
    elif (mask_intersection / mask1_area > 0.9) or (mask_intersection / mask2_area > 0.9):
        flag = True
    return flag


def iou(box1: np.ndarray, box2: np.ndarray):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou_box = intersection / float(box1_area + box2_area - intersection)
    if iou_box >= 0.75:
        return True
    else:
        return False


def mergeInstance(instances, args):
    merge_instances = []
    for instance in instances:
        merge_instance = []
        scores = instance["instances"].scores.cpu().detach().numpy()
        classes = instance["instances"].pred_classes.cpu().detach().numpy()
        masks = instance["instances"].pred_masks.cpu().detach().numpy()
        boxs = instance["instances"].pred_boxes.tensor.cpu().detach().numpy()

        valid_indices = []
        num = len(scores)
        for i in range(num):
            m = masks[i].astype(np.float64)
            b = boxs[i]
            s = scores[i]
            if s >= args.pseudo_threshold and m.sum() > 200 and (
                    m.sum() / (np.abs(b[2] - b[0]) * np.abs(b[3] - b[1]))) > 0.15 and (
                    min(np.abs(b[2] - b[0]), np.abs(b[3] - b[1])) / max(np.abs(b[2] - b[0]),
                                                                        np.abs(b[3] - b[1]))) > 0.15:
                valid_indices.append(i)

        valid_indices = np.array(valid_indices)
        if len(valid_indices) <= 0:
            merge_instances.append(merge_instance)
            continue
        scores = scores[valid_indices]
        classes = classes[valid_indices]
        masks = masks[valid_indices]
        boxs = boxs[valid_indices]

        num = len(scores)
        flag = np.zeros(num, dtype=bool)

        i = 0
        instance_id = 0
        while i < num:
            if flag[i]:
                i += 1
            else:
                j = i + 1
                m1 = masks[i]
                b1 = boxs[i]
                merge_index = [i]
                flag[i] = True
                while j < num:
                    if flag[j]:
                        j += 1
                    else:
                        m2 = masks[j]
                        b2 = boxs[j]
                        if whetherMerge(b1, b2, m1, m2):
                            merge_index.append(j)
                            flag[j] = True
                        j += 1

                merge_index = np.array(merge_index)
                if np.sum(masks[merge_index]) > 0:
                    merge_instance.append(
                        Instance(instance_id, scores[merge_index], classes[merge_index], masks[merge_index],
                                 args))
                    instance_id += 1
                    i += 1

        merge_instances.append(merge_instance)

    return merge_instances


class Instance:
    def __init__(self, instance_id, scores: np.ndarray, classes: np.ndarray, masks: np.ndarray, args):
        self.class_all_id = classes  # 保存当前候选区域可能的类别      store current class
        self.class_score = scores  # 保存每个类别对应的得分（小数形式） store class score
        self.instance_id = instance_id  # 该实例在所属图像中的编号    set the instance id
        self.class_len = len(scores)  # 保存该实例被判定为几个类别  save the length of class number

        # 选择该实例的类别
        index = np.argmax(self.class_score)
        self.class_id = self.class_all_id[index]  # 将得分最高的作为这个实例的类别  save the highest name of class score
        self.class_max_score = self.class_score[index]  # 保存最高得分   save the highest score

        # 合并mask
        self.mask = np.any(masks, axis=0).astype(np.uint8)

        sem_mask = self.mask
        contours, _ = cv.findContours(sem_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # filter mask
        areas = [cv.contourArea(contour) for contour in contours]
        max_area = max(areas)
        filter_contours = []
        for i, contour in enumerate(contours):
            if len(contours) == 1 or areas[i] >= args.area_threshold * max_area or (
                    self.class_id == 1 and areas[i] >= 0.1 * max_area):
                filter_contours.append(contour)
            else:
                cv.drawContours(self.mask, [contour], 0, 0, -1)

        merged_contours = np.concatenate(filter_contours)

        x_all = []
        y_all = []
        for point in merged_contours.squeeze(axis=1):
            x_all.append(point[0])
            y_all.append(point[1])

        # 计算新的候选框位置
        self.box = np.array([np.min(x_all), np.min(y_all), np.max(x_all), np.max(y_all)])

        normalization_score = (self.class_score - args.pseudo_threshold) / (1.0 - args.pseudo_threshold)
        self.weight_value = np.min(np.exp(normalization_score) / np.sum(np.exp(normalization_score)))

    def boxAreaInfo(self):
        return np.abs(self.box[2] - self.box[0]) * np.abs(self.box[3] - self.box[1])

    def maskAreaInfo(self):
        return np.sum(self.mask.astype(np.uint8))


def weightAllocation(candidate_instance, proposal_info, epoch, args, device="cuda"):
    # 计算权重的调整函数（x：0-1）  map weight
    min_weight = args.min_weight + (0.85 - args.min_weight) * (epoch / (args.end_epoch - 1))
    k = (1.0 - min_weight) / 1.0
    b = 1.0 - k

    # 配置权重  weight allocation
    batch_size = args.train_batch_size
    boxs = proposal_info['proposal_boxs'].detach().to(torch.device(device))
    boxs = [boxs[i * args.BATCH_SIZE_PER_IMAGE:(i + 1) * args.BATCH_SIZE_PER_IMAGE] for i in range(batch_size)]
    labels = proposal_info['proposal_classes'].detach().to(torch.device(device))
    labels = [labels[i * args.BATCH_SIZE_PER_IMAGE:(i + 1) * args.BATCH_SIZE_PER_IMAGE] for i in range(batch_size)]
    loss_weight = []

    for instance_per, box_per, labels_per in zip(candidate_instance, boxs, labels):
        sample_size = len(labels_per)
        instance_size = len(instance_per)
        for i in range(sample_size):
            if labels_per[i] == args.NUM_CLASSES:
                loss_weight.append(1.0)
                sample_label = [0.0] * (args.NUM_CLASSES + 1)
                sample_label[-1] = 1.0
            else:
                proposal_box = box_per[i].cpu().numpy()
                for j in range(instance_size):
                    box = instance_per[j].box
                    if iou(proposal_box, box):
                        loss_weight.append(k * instance_per[j].weight_value + b)
                        break
    loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
    return loss_weight


# ###############################将同图像的实例转换为可视化格式#########################################

def imageInstancesToDict(img_raw_data, instances):
    dataset_dict_raw = []
    for img_data, img_instances in zip(img_raw_data, instances):
        record = {"id": img_data["id"],
                  "height": img_data["image"].shape[0],
                  "width": img_data["image"].shape[1]
                  }

        objs = []
        for instance in img_instances:
            obj = {
                "bbox": instance.box.tolist(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": pycocotools.mask.encode(np.asarray(instance.mask.astype(np.uint8), order="F")),
                "category_id": instance.class_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dict_raw.append(record)
    return dataset_dict_raw
