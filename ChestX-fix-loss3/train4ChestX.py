r""" CuMNet training (validation) code """
import sys

sys.path.insert(0, "../")

import argparse

import torch.optim as optim
import torch.nn as nn
import torch

from model.cumnet import CuMNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from assistDataSet.assistDS4ChestX import cutmix4support_segmentation, randomSelect4Lung, cutmix4query_segmentation


def train(epoch, model, dataloader, optimizer, training):
    r""" Train CuMNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    # if training:
    for idx, batch in enumerate(dataloader):
        # 1. Networks forward pass

        # get assistDataSet from ChestX
        if training:
            img2, mask2 = randomSelect4Lung()
            batch['support_imgs'], batch['support_masks'] = cutmix4support_segmentation(
                batch['support_imgs'],
                batch['support_masks'],
                img2,mask2,
                beta_factor=0.4)
            batch['query_img'], batch['query_mask'] ,partMask= cutmix4query_segmentation(
                batch['query_img'],
                batch['query_mask'],
                img2,mask2,
                beta_factor=0.4)

        batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss1 = model.module.compute_objective(logit_mask, batch['query_mask'])

        # 对cutmix的部分进行loss计算    ---     start
        bsz,_, H, W  = logit_mask.shape
        H = int(0.4*H)
        lossi = 0

        if training:
            loss_f =  nn.CrossEntropyLoss()
            for i in range(bsz):
                a = logit_mask[i,:, 100:100 + H, 100:100 + H].reshape(1,2,-1).cuda()
                b = partMask.reshape(1, -1).long().cuda()
                lossi += loss_f(a, b)
            # 对cutmix的部分进行loss计算    ---     end

        loss = loss1 + lossi * 0.1

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Cross-Domain Few-Shot Semantic Segmentation Pytorch Implementation')
    # modify-WHT
    parser.add_argument('--datapath', type=str,
                        default=r'F:/WHT/CDFSSDataSet/Source domain/PASCAL VOC2012/VOCtrainval_11-May-2012/VOCdevkit')
    parser.add_argument('--benchmark', type=str, default='pascal')
    parser.add_argument('--logpath', type=str, default='test_case_AC')
    parser.add_argument('--bsz', type=int, default=4)
    # Deeplobe+ISIC=1e-3 ChestX+FSS=5e-5
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr2', type=float, default=5e-3)
    parser.add_argument('--niter', type=int, default=40)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50'])
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = CuMNetwork(args.backbone)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    custom_learning_rates = [{"params": model.module.backbone.parameters(), "lr": args.lr},
                             {"params": model.module.reference_layer3.parameters(), "lr": args.lr},
                             {"params": model.module.reference_layer2.parameters(), "lr": args.lr},
                             {"params": model.module.reference_layer1.parameters(), "lr": args.lr},
                             {"params": model.module.hpn_learner.parameters(), "lr": args.lr},
                             {"params": model.module.cross_entropy_loss.parameters(), "lr": args.lr},
                             # {"params": model.module.RA_weight1, "lr": args.lr2},
                             # {"params": model.module.RA_weight2, "lr": args.lr2},
                             {"params": model.module.beta, "lr": args.lr2},
                             # {"params": model.module.RANet.parameters(), "lr": args.lr},
                             ]
    optimizer = optim.Adam(custom_learning_rates)

    # optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    FSSDataset.initialize(img_size=400, datapath=r'F:\WHT\CDFSSDataSet\Target domains')
    dataloader_val = FSSDataset.build_dataloader('lung', args.bsz, args.nworker, args.fold, 'val')

    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)

        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
