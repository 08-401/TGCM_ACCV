r""" CuMNetwork """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
# from .base.ReverseAttention import RA
# from .base.ReverseAttention_2 import RA
from .base.BatchChannelPreturbeAttention import  BatchChannelPreturbeAttention
import torchvision.models as models
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
import matplotlib.pyplot as plt

class CuMNetwork(nn.Module):
    def __init__(self, backbone):
        super(CuMNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.reference_layer3 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer3.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer3.bias, 0)
            self.reference_layer2 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer2.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer2.bias, 0)
            self.reference_layer1 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer1.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer1.bias, 0)
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            # Anchor Layer l-m-h
            self.reference_layer3 = nn.Linear(2048, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer3.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer3.bias, 0)
            self.reference_layer2 = nn.Linear(1024, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer2.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer2.bias, 0)
            self.reference_layer1 = nn.Linear(512, 2, bias=True)
            nn.init.kaiming_normal_(self.reference_layer1.weight, a=0, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.reference_layer1.bias, 0)
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.fuse_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.5)

        # modify-WHT
        self.BCPA = BatchChannelPreturbeAttention()
        # self.RANet = RA()
        # self.RANet= RA(2048, hidden_channel=64)
        # self.RA_weight1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.RA_weight1.data.fill_(0.5)
        # self.RA_weight2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.RA_weight2.data.fill_(0.4)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.4)
        # self.AANet = AA()
    def forward(self, query_img, support_img, support_mask):
        # feature_extraction特征提取器是不需要训练的
        with torch.no_grad():
            # 将query_set 以及 support_set进行特征提取
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            # 保留预处理后的矩阵
            support_feats_old = support_feats.copy()
            query_feats_old = query_feats.copy()
            # mask_features:前景特征金字塔 prototypes_f:前景原型表示 prototypes_b:背景原型表示 masks:不同size的mask
            support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats, support_mask)
        query_feats, support_feats = self.Transformation_Feature(query_feats, support_feats, prototypes_f, prototypes_b,None,None)

        query_feats_1, support_feats_1 = query_feats, support_feats
        # 这个在第一层上面效果非常差
        # query_feats, support_feats = self.Modifylast12feature(query_feats, support_feats)
        corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

        logit_mask = self.hpn_learner(corr)
        logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)


        # 获取第一次获取到的query-mask并做一次mask_feature
        support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats_old, support_mask)
        query_feats, prototypes_qf, prototypes_qb = self.mask_feature(query_feats_old,logit_mask)
        query_feats, support_feats = self.Transformation_Feature(query_feats, support_feats, prototypes_f, prototypes_b,
                                                                 prototypes_qf, prototypes_qb)

        # 下次实验试试把这条注释去了
        query_feats, support_feats = self.Modifylast12feature(query_feats, support_feats)
        for i in range(1,len(support_feats_1)):
            query_feats[i] = torch.add(self.beta * query_feats_1[i], query_feats[i])
            support_feats[i] = torch.add(self.beta * support_feats_1[i], support_feats[i])

        corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
        logit_mask = self.hpn_learner(corr)
        logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)
        return logit_mask

    def mask_feature(self, features, support_mask):
        # mask_feature - 遮蔽特征
        # features - 特征金字塔 - ids:feature
        # support_mask - support_set的掩膜
        eps = 1e-6
        epx = 5e-3
        prototypes_f = []
        prototypes_b = []
        bg_features = []
        mask_features = []
        for idx, feature in enumerate(features): ## [layernum, batchsize, C, H, W]
            # 使用线性插值的方式对feature进行mask操作
            if support_mask.size(1) == 2:
                # 4*2*400*400 -- 4*2*400*400
                support_mask_old = support_mask.clone()
                support_mask_old = support_mask_old.argmax(dim=1,keepdim=True)
                # 即 support_mask[:, 1, :, :]的意义 --
                pre_mask = F.interpolate(support_mask_old.float(), feature.size()[2:], mode='bilinear', align_corners=True)
                # 即 support_mask[:, 0, :, :]的意义
                bgmask_0 = 1-pre_mask
                pre_mask = pre_mask.bool()
                bgmask_0 = bgmask_0.bool()

                # 通过 pre_mask * support_mask[:, 0, :, :] 以及 bgmask_0 * support_mask[:, 0, :, :]得到两次矩阵
                support_mask = support_mask.softmax(dim=1)
                mask = torch.masked_fill(
                    F.interpolate(support_mask[:, 1, :, :].unsqueeze(1).float(), feature.size()[2:], mode='bilinear',
                                  align_corners=True),
                    mask=pre_mask, value=0) + epx

                bg_mask = torch.masked_fill(
                    F.interpolate(support_mask[:, 0, :, :].unsqueeze(1).float(), feature.size()[2:], mode='bilinear',
                                  align_corners=True),
                    mask=bgmask_0, value=0) + epx

            else:
                mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear',align_corners=True)
                bg_mask = 1 - mask
            # 分别获取 前景 和 背景并加入至 背景特征金字塔-前景特征金字塔 中
            bg_features.append(features[idx] * bg_mask)
            mask_features.append(features[idx] * mask)
            features[idx] = features[idx] * mask
            ### prototype - [分层特征原型表示] 及 [背景原型表示]
            proto_f = features[idx].sum((2, 3))
            label_sum = mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            prototypes_f.append(proto_f)

            proto_b = bg_features[idx].sum((2, 3))
            label_sum = bg_mask.sum((2, 3))
            proto_b = proto_b / (label_sum + eps)
            prototypes_b.append(proto_b)
        # return - mask_features:前景特征金字塔 prototypes_f:前景原型表示 prototypes_b:背景原型表示
        return mask_features, prototypes_f, prototypes_b

    def Modifylast12feature(self,query_feats, support_feats):
        for i in range(1,len(support_feats)):
            if i <=3 :
                pass
            elif i<=6:
                support_feats[i] = torch.add(0.5 * support_feats[i], self.BCPA(support_feats[i]))
                query_feats[i] = torch.add(0.5 * query_feats[i], self.BCPA(query_feats[i]))
            else:
                support_feats[i] = torch.add(0.5 * support_feats[i],self.BCPA(support_feats[i]))
                query_feats[i] = torch.add(0.5 * query_feats[i],self.BCPA(query_feats[i]))


        return query_feats, support_feats

    def Transformation_Feature(self, query_feats, support_feats, prototypes_f, prototypes_b,prototypes_qf=None, prototypes_qb=None):
        # 转换特征模块
        # query_feats:      query_set   的特征金字塔
        # support_feats:    support_set 的前景特征金字塔
        # prototypes_f:     support_set 的前景原型表示
        # prototypes_b:     support_set 的背景原型表示
        # prototypes_qf:    query_set 的前景原型表示
        # prototypes_qb:    query_set 的前景原型表示
        transformed_query_feats = []
        transformed_support_feats = []
        bsz = query_feats[0].shape[0]
        for idx, feature in enumerate(support_feats):
            # Cat(前景原型表示,背景原型表示) C.shape = BatchSize,Channel*2,H,W
            if prototypes_qf == None:
                C = torch.cat((prototypes_b[idx].unsqueeze(1), prototypes_f[idx].unsqueeze(1)), dim=1)
            else:
                C = self.fuse_weight * torch.cat((prototypes_b[idx].unsqueeze(1), prototypes_f[idx].unsqueeze(1)),
                                                 dim=1) + (1. - self.fuse_weight) * torch.cat(
                    (prototypes_qb[idx].unsqueeze(1), prototypes_qf[idx].unsqueeze(1)), dim=1)
            eps = 1e-6
            # 逐像素分类器 - pixel-wise classification
            # 将reference_layer1的参数进行扩展
            if idx <= 3:
                R = self.reference_layer1.weight.expand(C.shape)
            elif idx <= 9:
                R = self.reference_layer2.weight.expand(C.shape)
            elif idx <= 12:
                R = self.reference_layer3.weight.expand(C.shape)

            # 对 R-C 进行正则化
            power_R = ((R * R).sum(dim=2, keepdim=True)).sqrt()
            R = R / (power_R + eps)
            power_C = ((C * C).sum(dim=2, keepdim=True)).sqrt()
            C = C / (power_C + eps)

            # torch.pinverse=奇异值分解
            P = torch.matmul(torch.pinverse(C), R)
            P = P.permute(0, 2, 1)

            # idx   query_feat    support_mask    feature
            # 3     bsz*512*2500  bzs*1*50*50     bsz*512*2500
            # 9     bsz*1024*625  bzs*1*25*25     bsz*1024*625
            # 12    bsz*2048*169  bzs*1*13*13     bsz*2048*169
            init_size = feature.shape
            feature = feature.view(bsz, C.size(2), -1)
            transformed_support_feats.append(torch.matmul(P, feature).view(init_size))

            init_size = query_feats[idx].shape
            query_feats[idx] = query_feats[idx].view(bsz, C.size(2), -1)
            transformed_query_feats.append(torch.matmul(P, query_feats[idx]).view(init_size))
        return transformed_query_feats, transformed_support_feats

    def calDist(self, feature, prototype, scaler=20):
        dist = F.cosine_similarity(feature, prototype[..., None, None], dim=1) * scaler
        return dist # dist:[1,53,53]

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
