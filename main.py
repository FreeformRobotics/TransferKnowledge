import detectron2.utils.comm as comm
from detectron2.modeling import build_model
from detectron2.engine import default_writers
from detectron2.utils.events import EventStorage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

from config.Arguments import get_args
from dataloader.DataLoder import UnlabeledDataset, LabeledDataset
from prototype.PrototypeCompute import PrototypeBank
from dataloader.DataProcessing import labeledForTrain, unlabeledForPseudoLabel, instancesForTrain
from dataloader.Instance import mergeInstance, weightAllocation
from loss.PrototypeInfoNceLoss import PrototypeInfoNCE
from loss.PrototypeContrastiveLoss import PrototypeContrastiveLoss

from loss.ApVal import get_dataset_dict
from detectron2.data import build_detection_test_loader


class EmaFrequencyUpdate:
    def __init__(self, args):
        self.init_frequency = args.ema_frequency
        self.ema_frequency_times = args.ema_frequency_times

        self.frequency = [int(self.init_frequency - i * (self.init_frequency - 1) / (self.ema_frequency_times - 1)) for
                          i in range(self.ema_frequency_times)]

        self.now_frequency = self.frequency.pop(0)  # 当前的更新频率  Current update frequency
        self.iter_num = 0

    def step(self):
        self.iter_num += 1
        if self.iter_num == self.now_frequency:
            if self.frequency:
                self.now_frequency = self.frequency.pop(0)
            else:
                self.now_frequency = 1
            self.iter_num = 0
            return True
        else:
            return False


def do_train():
    args, cfg, cfg_teacher = get_args()
    model = build_model(cfg)
    model_teacher = build_model(cfg_teacher)

    # 加载student net模型参数  load student model
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.train()

    # 加载teacher net模型参数   load teacher model
    checkpointer_teacher = DetectionCheckpointer(model_teacher, cfg_teacher.OUTPUT_DIR)
    checkpointer_teacher.load(cfg_teacher.MODEL.WEIGHTS)
    for param in model_teacher.parameters():
        param.detach_()
    model_teacher.requires_grad_(False)
    model_teacher.eval()

    # 有标签训练数据集   labeled dataset
    train_data_labeled = LabeledDataset(labeled_datasets_root_path=args.labeled_path)
    train_labeled_loader = DataLoader(train_data_labeled, batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.train_batch_size, pin_memory=False, collate_fn=lambda x: x)

    # 无标签训练数据集   unlabeled dataset
    train_data_unlabeled = UnlabeledDataset(unlabeled_datasets_root_path=args.unlabeled_path)
    train_unlabeled_loader = DataLoader(train_data_unlabeled, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=args.train_batch_size, pin_memory=False, collate_fn=lambda x: x)

    # 自动验证数据集    val dataset
    DatasetCatalog.register("val", lambda: get_dataset_dict(args.val_path, num_class=22))
    MetadataCatalog.get("val").set(
        thing_classes=["chair", "table", "picture", "cabinet", "cushion", "sofa", "bed",
                       "chest_of_drawers", "plant", "sink", "toilet", "stool", "towel",
                       "tv_monitor", "shower", "bathtub", "counter", "fireplace", "gym_equipment",
                       "seating", "clothes"])
    metadata = MetadataCatalog.get("val")

    val_loader = build_detection_test_loader(cfg, 'val', batch_size=args.val_batch_size)
    save_path = "./output/student/AP_test"
    evaluator = COCOEvaluator("val", cfg, False, output_dir=save_path)

    print("labeled len: ", len(train_labeled_loader))
    print("unlabeled len: ", len(train_unlabeled_loader))
    print("val len: ", len(val_loader))

    # 初始化原型中心
    prototype_bank = PrototypeBank(args)
    if not prototype_bank.whetherResume():  # 没有初始文件   without prototype center file
        prototype_bank.initPrototype(model_teacher, train_labeled_loader)

    # 初始化损失函数  initialize loss
    infoNCE_loss = PrototypeInfoNCE(args)  # 用于有标签数据的损失   for labeled dataset
    contrast_loss = PrototypeContrastiveLoss(args)  # 用于无标签数据损失   for unlabeled dataset

    # EMA更新频率调整   EMA update frequency
    frequency_adjust = EmaFrequencyUpdate(args)

    # 创建训练过程保存文件  training information file
    writers = default_writers(args.model_output_path, cfg.SOLVER.MAX_ITER) if comm.is_main_process() else []

    max_AP50 = 0
    AP_sem = 0
    save_epoch = None

    with EventStorage() as storage:
        for epoch in tqdm(range(args.start_epoch, args.end_epoch)):
            model.train()
            for now_num, (labeled_data, unlabeled_data) in enumerate(
                    zip(train_labeled_loader, train_unlabeled_loader)):
                storage.iter = epoch * len(train_unlabeled_loader) + now_num
                # 生成伪标签  generate pseudo-labels
                input_data = unlabeledForPseudoLabel(unlabeled_data)
                candidate_label = model_teacher(input_data)
                candidate_instance = mergeInstance(candidate_label, args)

                # 伪标签训练  trained by pseudo-labels
                unlabeled_train = instancesForTrain(candidate_instance, unlabeled_data)
                loss_unlabeled, proposal_info_unlabeled = model(unlabeled_train)
                loss_weight = weightAllocation(candidate_instance, proposal_info_unlabeled, epoch, args)

                # 有标签训练  trained by gt-label
                labeled_train = labeledForTrain(labeled_data)
                loss_labeled, proposal_info_labeled = model(labeled_train)

                # 更新原型中心  update prototype center
                prototype_bank.updateEMA(proposal_info_labeled, proposal_info_unlabeled, loss_weight)

                # 计算损失   loss
                infoNCE_loss_labeled, loss_score_labeled = infoNCE_loss(proposal_info_labeled,
                                                                        prototype_bank)
                cont_loss_unlabeled, loss_score_unlabeled = contrast_loss(proposal_info_unlabeled, prototype_bank,
                                                                          loss_weight)
                losses = 0.5 * (
                        sum(loss_labeled.values()) + loss_score_labeled + 0.25 * infoNCE_loss_labeled) + 0.5 * (
                                 sum(loss_unlabeled.values()) + loss_score_unlabeled + 0.25 * cont_loss_unlabeled)  #

                # 反向传播   backpropagation
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # 记录训练过程  record
                loss_dict_reduced = {k: v.item() for k, v in
                                     comm.reduce_dict(loss_labeled).items()}
                print("epoch ", epoch, ": [", now_num, "/", int(args.unlabeled_train_size / args.train_batch_size),
                      "]*******************")
                print("最大AP50：", max_AP50, "   当前AP50：", AP_sem)
                print("总损失：", losses.item())
                print("各项损失：", loss_dict_reduced)
                print("lr: ", optimizer.param_groups[0]["lr"])

                storage.put_scalars(total_loss=losses.item(), **loss_dict_reduced)
                storage.put_scalar("infoNCE_loss", infoNCE_loss_labeled.item(), smoothing_hint=False)
                storage.put_scalar("cont_loss", cont_loss_unlabeled.item(), smoothing_hint=False)
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                for writer in writers:
                    writer.write()

                # 更新teacher net     update teacher net
                if frequency_adjust.step():
                    UpdateEMA(model, model_teacher, ema_decay=args.ema_decay)

                # 保存临时文件    store tmp file
                if now_num % 99 == 0:
                    checkpointer.save("tmp_model")

            # AP
            eval_results = inference_on_dataset(model=model, data_loader=val_loader, evaluator=evaluator)
            AP_sem = eval_results['segm']['AP50']
            if AP_sem > max_AP50:  # 保存模型   save model
                checkpointer.save(args.model_save_name)
                save_epoch = epoch
                max_AP50 = AP_sem
            else:
                break

            print("平均AP50: ", AP_sem, "最大AP50：", max_AP50)
            # 保存新原型中心（每个epoch保存一次）   save prototype center
            prototype_bank.save()
            scheduler.step()

        print("最终保存的模型的epoch为：", save_epoch)
        checkpointer.save("tmp_model")
        print("平均AP50: ", AP_sem, "最大AP50：", max_AP50)
        print("max AP50: ", max_AP50)


def UpdateEMA(module, ema_module, ema_decay):
    # UpdateEMA function is largely borrowed from
    # https://github.com/facebookresearch/GuidedDistillation/blob/main/modules/train_loop.py
    # Update parameters.
    module_params = OrderedDict(module.named_parameters())
    ema_module_params = OrderedDict(ema_module.named_parameters())

    assert module_params.keys() == ema_module_params.keys()

    for name, param in module_params.items():
        ema_module_params[name].sub_((1. - ema_decay) * (ema_module_params[name] - param))

    # Update buffers.
    module_buffers = OrderedDict(module.named_buffers())
    ema_module_buffers = OrderedDict(ema_module.named_buffers())

    assert module_buffers.keys() == ema_module_buffers.keys()

    for name, buffer in module_buffers.items():
        if buffer.dtype == torch.float32:
            ema_module_buffers[name].sub_((1. - ema_decay) * (ema_module_buffers[name] - buffer))
        else:
            ema_module_buffers[name] = buffer.clone()


if __name__ == "__main__":
    do_train()
