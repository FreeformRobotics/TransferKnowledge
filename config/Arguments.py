import argparse
import os
from detectron2.config import get_cfg
from detectron2 import model_zoo


def get_args():
    # 设置基础参数
    parser = argparse.ArgumentParser(description='Semi-supervised')
    parser.add_argument('--device', type=str, default="cuda", help='运行设备')
    parser.add_argument('--start_epoch', type=int, default=0, help='开始的训练轮数')
    parser.add_argument('--end_epoch', type=int, default=10, help='结束的训练轮数')
    parser.add_argument('--unlabeled_train_size', type=int, default=5548, help='训练集无标签数据数量')
    parser.add_argument('--train_batch_size', type=int, default=4, help='训练集(有标签和无标签相同)mini-batch大小')
    parser.add_argument('--val_batch_size', type=int, default=3, help='各高度验证集mini-batch大小')
    parser.add_argument('--pseudo_threshold', default=0.60, help='筛选伪标签的分数初始阈值')
    parser.add_argument('--min_weight', type=float, default=0.3, help='权重分配时的最小权重')
    parser.add_argument('--area_threshold', type=float, default=0.15, help='用于过滤离群的伪标签区域')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='半监督初始学习率')  # 0.00001
    parser.add_argument('--gamma', type=float, default=0.95, help='每个epoch调整学习率的参数')
    parser.add_argument('--ema_decay', type=float, default=0.9996, help='Teacher的EMA更新动量')
    parser.add_argument('--ema_frequency', type=int, default=600, help='EMA的初始更新频率（迭代次数/次更新）')
    parser.add_argument('--ema_frequency_times', type=int, default=10, help='EMA频率变为1所需的更新次数')
    parser.add_argument('--infoNCE_temperature', type=float, default=0.5, help='infoNCE损失的温度系数')
    parser.add_argument('--contrast_temperature', type=float, default=0.8, help='对比损失的温度系数')
    parser.add_argument('--contrast_similarity_momentum', type=float, default=0.9, help='原型得分相似度调整参数')
    parser.add_argument('--contrast_score_momentum', type=float, default=0.95, help='无标签得分相似度调整参数')
    parser.add_argument('--prototype_similarity_threshold', type=float, default=0.6, help='原型更新相似度阈值')
    parser.add_argument('--prototype_momentum', type=float, default=0.995, help='原型中心的更新动量')
    parser.add_argument('--prototype_feature_num', type=int, default=1024, help='原型的特征长度')

    parser.add_argument('--IOUVal_threshold', type=float, default=0.85, help='验证集获取置信度阈值')

    # 路径
    parser.add_argument('--pretrained_model', type=str,
                        default='./output/teacher/model_final.pth',
                        help='预训练模型所在路径')
    parser.add_argument('--model_output_path', type=str, default='./output/student', help='训练后模型输出位置')
    parser.add_argument('--model_save_name', type=str, default='semi_model_5k')
    parser.add_argument('--prototype_path', type=str, default='./prototype/prototype_centre.pth')
    parser.add_argument('--labeled_path', type=str,
                        default='./Semantic_Dataset/labeled/0.88->0.28_projection',
                        help='有标签数据位置 ')
    parser.add_argument('--unlabeled_path', type=str, default='./Semantic_Dataset/unlabeled/0.28',
                        help='无标签数据位置')
    parser.add_argument('--val_path', type=str, default='./Semantic_Dataset/val/0.28', help='验证集位置')

    # 数据集参数
    parser.add_argument('--labeled_high', type=float, default=0.88, help='有标签数据高度（原高度）')
    parser.add_argument('--unlabeled_high', type=float, default=0.28, help='无标签数据高度（目标高度）')

    # 模型参数
    parser.add_argument('--BATCH_SIZE_PER_IMAGE', type=int, default=128, help='模型ROI头输出的预测框数量')
    parser.add_argument('--NUM_CLASSES', type=int, default=21, help='模型预测的类别数量（不包括背景）')

    args = parser.parse_args()

    # 设置模型参数
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # 设置student net配置
    cfg.OUTPUT_DIR = args.model_output_path

    cfg.MODEL.WEIGHTS = os.path.join(args.pretrained_model)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    cfg.SOLVER.BASE_LR = args.base_lr * pow(args.gamma, args.start_epoch)
    cfg.SOLVER.MAX_ITER = int(((args.unlabeled_train_size / args.train_batch_size) + 1) * (args.end_epoch + 1))
    cfg.SOLVER.WARMUP_FACTOR = 1.0
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.STEPS = [i for i in range(args.start_epoch + 1, args.end_epoch, 2)]
    cfg.SOLVER.GAMMA = args.gamma

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = 480
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TEST = 480

    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    # 设置teacher net配置
    cfg_teacher = cfg.clone()
    cfg_teacher.defrost()
    cfg_teacher.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg_teacher.MODEL.DEVICE = args.device
    cfg_teacher.freeze()
    return args, cfg, cfg_teacher
