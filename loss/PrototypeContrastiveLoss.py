import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.PrototypeCompute import PrototypeBank


class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(PrototypeContrastiveLoss, self).__init__()
        self.device = args.device
        self.class_num = args.NUM_CLASSES + 1  # 包括背景类的数量  include the number of class name
        self.feature_num = args.prototype_feature_num
        self.temperature = args.contrast_temperature
        self.contrast_similarity_momentum = args.contrast_similarity_momentum
        self.contrast_score_momentum = args.contrast_score_momentum
        self.cross_criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, proposal_info, proto_bank: PrototypeBank, loss_weight=None):
        protos = proposal_info['box_features'].to(torch.device(self.device))
        labels = proposal_info['proposal_classes'].to(torch.device(self.device))
        scores = proposal_info['proposal_scores'].to(torch.device(self.device))

        similarity_matrix = F.cosine_similarity(
            protos.unsqueeze(1),
            proto_bank.prototype.unsqueeze(0),
            dim=2
        )

        numerator = similarity_matrix / self.temperature

        loss_weight = loss_weight.to(torch.device(self.device))
        sim_proto_centre = proto_bank.computeSimilarityClasses().to(
            torch.device(self.device))
        target = torch.zeros(labels.size(0), self.class_num).to(torch.device(self.device))
        target[torch.arange(labels.size(0)), labels] = 1.0

        target_proto = (self.contrast_similarity_momentum * target +
                        (1 - self.contrast_similarity_momentum) * torch.mm(target, sim_proto_centre))
        target_score = (self.contrast_score_momentum * target +
                        (1 - self.contrast_score_momentum) * torch.mm(target, sim_proto_centre))
        loss_proto = loss_weight * self.cross_criterion(numerator, target_proto)
        loss_score = loss_weight * self.cross_criterion(scores, target_score)
        return loss_proto.mean(), loss_score.mean()
