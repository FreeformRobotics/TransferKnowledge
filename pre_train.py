from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os
import cv2 as cv
import numpy as np

from detectron2.structures import BoxMode
import pycocotools
import imageio.v2 as imageio


def get_dataset_dict(data_dir, num_class=22):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!若要训练不同数量的类别，这条函数共需要修改两处地方！！！！！！！！！！！！！！！！！
    Args:
        data_dir:
        num_class: 类别数量包括了背景

    Returns:

    """
    img_dir = []
    label_dir = []
    instance_dir = []

    dir_list = os.listdir(data_dir)
    for i_dir in dir_list:
        rgb_path = os.path.join(data_dir, i_dir, 'rgb')
        semantic_path = os.path.join(data_dir, i_dir, 'semantic')
        instance_path = os.path.join(data_dir, i_dir, 'instance')

        rgb_name = os.listdir(rgb_path)
        semantic_name = os.listdir(semantic_path)
        instance_name = os.listdir(instance_path)

        rgb_list = [os.path.join(rgb_path, r) for r in rgb_name]
        semantic_list = [os.path.join(semantic_path, s) for s in semantic_name]
        instance_list = [os.path.join(instance_path, i) for i in instance_name]

        img_dir.extend(rgb_list)
        label_dir.extend(semantic_list)
        instance_dir.extend(instance_list)

    img_dir.sort()
    label_dir.sort()
    instance_dir.sort()

    dataset_dicts = []
    for rgb_p, label_p, instance_p in zip(img_dir, label_dir, instance_dir):
        record = {}

        rgb = cv.imread(rgb_p)
        label = imageio.imread(label_p)
        instance = imageio.imread(instance_p)

        record["file_name"] = rgb_p

        img_name = os.path.basename(rgb_p)
        img_name = os.path.splitext(img_name)[0]
        record["image_id"] = img_name

        record["height"] = rgb.shape[0]
        record["width"] = rgb.shape[1]

        objs = []
        object_id = np.unique(instance) - 0
        object_id = object_id.tolist()
        for o_id in object_id:
            if o_id == 0:
                continue
            obj_mask = instance == o_id
            label_obj = np.zeros_like(label)
            instance_obj = np.zeros(instance.shape, dtype=np.uint8)

            label_obj[obj_mask] = label[obj_mask]

            sem_id = np.unique(label_obj[label_obj != 0])[0]
            if sem_id == num_class:
                continue
            else:
                label_obj[label_obj != 0] = 1
                if label_obj.sum() <= 200:
                    print(label_obj.sum())
                    continue
            sem_encode = pycocotools.mask.encode(np.asarray(label_obj, order="F"))

            instance_obj[obj_mask] = 1

            contours, _ = cv.findContours(instance_obj, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            merged_contours = np.concatenate(contours)

            x_all = []
            y_all = []
            for point in merged_contours.squeeze(axis=1):
                x_all.append(point[0])
                y_all.append(point[1])

            obj = {
                "bbox": [np.min(x_all), np.min(y_all), np.max(x_all), np.max(y_all)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": sem_encode,
                "category_id": sem_id - 1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def train():
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("labeled",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.OUTPUT_DIR = "./output/teacher"
    cfg.SOLVER.IMS_PER_BATCH = 7
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 28500  # 当使用两个高度时，迭代28500步(7 mini-batch)，一个高度时，迭代27000步(7 mini-batch)
    cfg.SOLVER.STEPS = []
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = 480
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TEST = 480
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    for d in ["labeled"]:
        DatasetCatalog.register(d, lambda x=d: get_dataset_dict(
            os.path.join("./Semantic_Dataset", x)))
        MetadataCatalog.get(d).set(
            thing_classes=["chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
                           "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
                           "tv_monitor", "shower", "bathtub", "counter", "fireplace", "gym_equipment",
                           "seating", "clothes"])
    train()


if __name__ == "__main__":
    main()
