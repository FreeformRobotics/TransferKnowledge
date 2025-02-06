import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloader.DataProcessing import labeledForTrain


class PrototypeBank:
    def __init__(self, args):
        self.device = args.device
        self.class_num = args.NUM_CLASSES + 1  # 包括背景类的数量
        self.momentum = args.prototype_momentum  # 原型更新动量
        self.feature_num = args.prototype_feature_num
        self.prototype_path = args.prototype_path
        self.resume = None  # 是否有初始化文件（True：已读取初始化文件；False：无初始化文件）

        # 判断是否有初始文件
        if os.path.isfile(self.prototype_path):  # 有文件
            checkpoint = torch.load(self.prototype_path, map_location=torch.device('cpu'))
            self.prototype = checkpoint['prototype_centre'].cuda(non_blocking=True)
            self.sample_amount = checkpoint['amount'].cuda(non_blocking=True)
            print("Prototype loaded")
            self.resume = True
        else:  # 无文件
            self.prototype = torch.zeros(self.class_num, self.feature_num).to(torch.device(self.device))
            self.sample_amount = torch.zeros(self.class_num).to(torch.device(self.device))
            self.resume = False

    def whetherResume(self) -> bool:
        return self.resume

    def initPrototype(self, model, dataloader: DataLoader):
        from detectron2.utils.events import EventStorage
        model.train()
        print("Prototype initializing ...")
        with EventStorage() as _:
            for data in dataloader:
                input_data = labeledForTrain(data)
                loss, proto_info = model(input_data)
                self.update(proto_info['box_features'], proto_info['proposal_classes'])
        model.eval()
        self.save()

    def update(self, features, labels):
        unique_labels, counts = torch.unique(labels, return_counts=True)
        temp_proto = torch.zeros(self.class_num, self.feature_num).to(torch.device(self.device))
        temp_amount = torch.zeros(self.class_num).to(torch.device(self.device))

        temp_proto[unique_labels] = torch.stack(
            [torch.mean(features[labels == label], dim=0) for label in unique_labels])
        temp_amount[unique_labels] = counts.to(temp_amount.dtype)

        mask = (temp_amount > 0)
        self.prototype[mask] = (self.prototype[mask] * self.sample_amount[mask].unsqueeze(1) + temp_proto[mask] *
                                temp_amount[mask].unsqueeze(1)) / (
                                       self.sample_amount[mask].unsqueeze(1) + temp_amount[mask].unsqueeze(1))
        self.sample_amount[mask] += temp_amount[mask]

    def updateEMA(self, labeled_info, unlabeled_info, loss_weight=None):
        # 有标签和无标签一起更新
        labels = torch.cat(
            [labeled_info['proposal_classes'], unlabeled_info['proposal_classes']], dim=0)
        protos = torch.cat([labeled_info['box_features'], unlabeled_info['box_features']], dim=0).detach()
        weight_tmp = torch.ones_like(loss_weight)
        weight = torch.cat([weight_tmp, loss_weight], dim=0).to(torch.device(self.device))

        valid_mask = (weight >= 0.85)

        temp_proto = torch.zeros(self.class_num, self.feature_num).to(torch.device(self.device))
        temp_amount = torch.zeros(self.class_num).to(torch.device(self.device))

        # 计算每个类的平均特征向量  compute the mean feature vector
        labels = labels[valid_mask]
        protos = protos[valid_mask]
        unique_labels, counts = torch.unique(labels, return_counts=True)
        temp_proto[unique_labels] = torch.stack(
            [torch.mean(protos[labels == label], dim=0) for label in unique_labels])
        temp_amount[unique_labels] = counts.to(temp_amount.dtype)

        mask = (temp_amount > 0)
        self.prototype[mask] = self.momentum * self.prototype[mask] + (1 - self.momentum) * temp_proto[mask]
        self.sample_amount[mask] += temp_amount[mask]

    def computeSimilarityClasses(self):
        cosine_sim = F.cosine_similarity(self.prototype.unsqueeze(1), self.prototype.unsqueeze(0), dim=2)
        cosine_sim_soft = F.softmax(cosine_sim, dim=1)
        return cosine_sim_soft

    def save(self):
        torch.save({
            'prototype_centre': self.prototype.cpu(),
            'amount': self.sample_amount.cpu()
        }, self.prototype_path)
