import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.PrototypeCompute import PrototypeBank


class PrototypeInfoNCE(nn.Module):
    def __init__(self, args):
        super(PrototypeInfoNCE, self).__init__()
        self.device = args.device
        self.class_num = args.NUM_CLASSES + 1  # 包括背景类的数量
        self.feature_num = args.prototype_feature_num
        self.contrast_similarity_momentum = args.contrast_similarity_momentum
        self.temperature = args.infoNCE_temperature

        self.cross_criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, proposal_info, proto_bank: PrototypeBank):
        protos = proposal_info['box_features'].to(torch.device(self.device))
        labels = proposal_info['proposal_classes'].to(torch.device(self.device))
        scores = proposal_info['proposal_scores'].to(torch.device(self.device))

        similarity_matrix = F.cosine_similarity(
            protos.unsqueeze(1),
            proto_bank.prototype.unsqueeze(0),
            dim=2
        )

        numerator = torch.exp(similarity_matrix[torch.arange(similarity_matrix.size(0)), labels] / self.temperature)
        denominator = torch.exp(similarity_matrix / self.temperature).sum(dim=1)

        loss_score = self.cross_criterion(scores, labels)

        loss = -torch.log(numerator / denominator).mean()

        return loss, loss_score
