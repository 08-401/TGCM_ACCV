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
from assistDataSet.assistDS4DeepGlobe import cutmix4support_segmentation, randomSelect4DeepGlobe, cutmix4query_segmentation


def train( model, dataloader, optimizer, training):
    r""" Train CuMNet """
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode()
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        loss = model.module.finetune_reference(batch, batch['query_mask'], 1)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Cross-Domain Few-Shot Semantic Segmentation Pytorch Implementation')
    parser.add_argument('--datapath', type=str,
                        default=r'F:/WHT/CDFSSDataSet/Source domain/PASCAL VOC2012/VOCtrainval_11-May-2012/VOCdevkit')
    parser.add_argument('--benchmark', type=str, default='pascal')
    parser.add_argument('--logpath', type=str, default='test_case_AC')
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--niter', type=int, default=10)
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

    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()

    FSSDataset.initialize(img_size=400, datapath=r'F:\WHT\CDFSSDataSet\Target domains')
    dataloader_val = FSSDataset.build_dataloader('deepglobe', args.bsz, args.nworker, args.fold, 'val')

    for epoch in range(args.niter):
        loss = train(model, dataloader_val, optimizer, training=False)
        Logger.save_TFI_backbone(model)

