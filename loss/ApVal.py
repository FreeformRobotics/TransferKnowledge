import os
import numpy as np
import cv2 as cv
import pycocotools
import imageio.v2 as imageio

from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from config.Arguments import get_args


def get_dataset_dict(data_dir, num_class=22):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!若要训练不同数量的类别，这条函数共需要修改两处地方！！！！！！！！！！！！！！！！！
    Args:
        data_dir: 直接用到特定高度的路径
        num_class: 类别数量包括了背景

    Returns:

    """
    # 获取各文件路径   file path
    rgb_path = os.path.join(data_dir, "rgb")
    semantic_path = os.path.join(data_dir, "semantic")
    instance_path = os.path.join(data_dir, "instance")
    rgb_name = os.listdir(rgb_path)
    semantic_name = os.listdir(semantic_path)
    instance_name = os.listdir(instance_path)

    img_dir = [os.path.join(rgb_path, r) for r in rgb_name]
    label_dir = [os.path.join(semantic_path, s) for s in semantic_name]
    instance_dir = [os.path.join(instance_path, i) for i in instance_name]

    img_dir.sort()
    label_dir.sort()
    instance_dir.sort()

    # 给每张图像注册标签  set the labels
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


def Ap_my():
    args, cfg, _ = get_args()
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, "../output/tmp")
    model_path = os.path.join("." + args.model_output_path, args.model_save_name + ".pth")
    checkpointer.load(model_path)
    model.eval()

    # 验证数据集  val dataset
    DatasetCatalog.register("val", lambda: get_dataset_dict("." + args.val_path, num_class=22))
    MetadataCatalog.get("val").set(
        thing_classes=["chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
                       "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
                       "tv_monitor", "shower", "bathtub", "counter", "fireplace", "gym_equipment",
                       "seating", "clothes"])
    save_path = "./output/tmp/AP_0.28"

    val_loader = build_detection_test_loader(cfg, 'val', batch_size=args.val_batch_size)
    print("val len: ", len(val_loader))

    evaluator = COCOEvaluator("val", cfg, False, output_dir=save_path)

    eval_results = inference_on_dataset(model=model, data_loader=val_loader, evaluator=evaluator)
    print("#############################################################")
    with open(os.path.join(save_path, 'AP.txt'), 'w') as f:
        for key, value in eval_results.items():
            f.write(f"{key}:{value}\n")
    print(eval_results)


if __name__ == "__main__":
    Ap_my()
