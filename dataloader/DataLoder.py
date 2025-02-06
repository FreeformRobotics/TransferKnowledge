import numpy as np
import os
import cv2 as cv

import imageio.v2 as imageio

from torch.utils.data import Dataset

"""
注意：在训练集中，所有的dataloader使用的类别数量均为22（包含背景）
"""


class UnlabeledDataset(Dataset):
    def __init__(self, unlabeled_datasets_root_path):
        """

        Args:
            unlabeled_datasets_root_path: 目标高度的文件夹路径
        """
        self.root_path = unlabeled_datasets_root_path
        rgb_path = os.path.join(self.root_path, "rgb")
        rgb_name = os.listdir(rgb_path)

        self.rgb_list = [os.path.join(rgb_path, r) for r in rgb_name]
        self.rgb_list.sort()

        self.len = len(self.rgb_list)

    def __getitem__(self, item):
        bgr = cv.imread(self.rgb_list[item])

        sample = {'image': bgr, 'id': self.rgb_list[item]}
        return sample

    def __len__(self):
        return self.len


class LabeledDataset(Dataset):
    def __init__(self, labeled_datasets_root_path, num_classes=22):
        """

        Args:
            labeled_datasets_root_path: 模型源高度的文件夹路径
        """
        self.root_path = labeled_datasets_root_path
        self.num_classes = num_classes  # 包括背景

        # 获取各文件路径
        rgb_path = os.path.join(self.root_path, "rgb")
        semantic_path = os.path.join(self.root_path, "semantic")
        instance_path = os.path.join(self.root_path, "instance")

        rgb_name = os.listdir(rgb_path)
        semantic_name = os.listdir(semantic_path)
        instance_name = os.listdir(instance_path)

        self.rgb_list = [os.path.join(rgb_path, r) for r in rgb_name]
        self.semantic_list = [os.path.join(semantic_path, s) for s in semantic_name]
        self.instance_list = [os.path.join(instance_path, i) for i in instance_name]
        self.rgb_list.sort()
        self.semantic_list.sort()
        self.instance_list.sort()

        self.len = len(self.rgb_list)

    def __getitem__(self, item):
        return self.transform2rawFormat(item)

    def __len__(self):
        return self.len

    def transform2rawFormat(self, item):
        rgb_p = self.rgb_list[item]
        semantic_p = self.semantic_list[item]
        instance_p = self.instance_list[item]

        record = {}

        # 读取所需信息
        rgb = cv.imread(rgb_p)
        label = imageio.imread(semantic_p)
        instance = imageio.imread(instance_p)

        # 该图像的尺寸
        record["image"] = rgb
        record["height"] = rgb.shape[0]
        record["width"] = rgb.shape[1]

        # 该图像的完整路径
        record["id"] = rgb_p

        # 记录每个实例的蒙板和语义
        per_class = []  # 每个实例对应的类别
        semantic_instance = np.zeros((0, 480, 640), dtype=np.uint8)  # 按照通道顺序记录每个实例的蒙板

        # 每个物体的关联信息
        object_id = np.unique(instance) - 0  # 得到每种物体的编号
        object_id = object_id.tolist()
        for o_id in object_id:
            if o_id == 0:
                continue
            obj_mask = instance == o_id
            label_obj = np.zeros_like(label)

            label_obj[obj_mask] = label[obj_mask]

            # 编码语义蒙版
            sem_id = np.unique(label_obj[label_obj != 0])[0]
            if sem_id == self.num_classes:
                continue
            else:
                label_obj[label_obj != 0] = 1
                if label_obj.sum() <= 200:
                    continue
            per_class.append(sem_id - 1)  # 注意：要记得减1  ！！！！！！
            semantic_instance = np.concatenate((semantic_instance, np.expand_dims(label_obj, axis=0)),
                                               axis=0)  # 将实例保存到对应通道中

        record["per_class"] = per_class
        record["instance_mask"] = semantic_instance
        return record


class ValDataset(Dataset):
    def __init__(self, val_datasets_root_path, num_classes=22):
        """
        读取验证集
        Args:
            val_datasets_root_path: 验证集（包含所有高度）所在位置的根目录
            num_classes: 类别总数
        """
        self.root_path = val_datasets_root_path
        self.height_list = os.listdir(self.root_path)
        self.num_classes = num_classes

        rgb_path = os.path.join(self.root_path, self.height_list[0], "rgb")
        semantic_path = os.path.join(self.root_path, self.height_list[0], "semantic")
        rgb_name = os.listdir(rgb_path)
        semantic_name = os.listdir(semantic_path)

        self.height_list.sort()
        rgb_name.sort()
        semantic_name.sort()

        self.rgb_list = [os.path.join(self.root_path, h, 'rgb', n) for n in rgb_name for h in self.height_list]
        self.semantic_list = [os.path.join(self.root_path, h, 'semantic', n) for n in semantic_name for h in
                              self.height_list]

        self.len = len(self.rgb_list)

    def __getitem__(self, item):
        return self.transform2channelMask(item)

    def __len__(self):
        return self.len

    def transform2channelMask(self, item):
        rgb_p = self.rgb_list[item]
        semantic_p = self.semantic_list[item]

        record = {}
        rgb = cv.imread(rgb_p)
        label = imageio.imread(semantic_p)

        record["image"] = rgb
        record["height"] = rgb.shape[0]
        record["width"] = rgb.shape[1]

        record["file_name"] = rgb_p

        semantic_channel = np.zeros((self.num_classes - 1, rgb.shape[0], rgb.shape[1]))
        for i in range(1, self.num_classes):
            semantic_channel[i - 1] = (label == i).astype(np.float32)

        record["instance_mask"] = semantic_channel
        return record

