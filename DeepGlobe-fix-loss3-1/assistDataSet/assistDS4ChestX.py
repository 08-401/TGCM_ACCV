import os
import torch
import random
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


def cutmixFeature4support_segmentation(support_feats, support_mask, assist_feats, mask2, beta_factor=0.4):
    bsz = support_feats[0].shape[0]
    H = 400
    W = 400
    cut_width = int(W * beta_factor)
    cut_height = int(H * beta_factor)
    cut_x1 = 128
    # 为了保持向下decoder我找到了一个地板除方案，使得每次的cut_x2都是8的倍数
    cut_x2 = np.random.randint(0, W - cut_width) // 8 * 8
    support_mask[:, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
                                                                             cut_x2:cut_x2 + cut_width]
    for idx, feature in enumerate(support_feats):
        # idx   query_feat    support_mask    feature
        # 3     bsz*512*2500  bzs*1*50*50     bsz*512*2500
        # 9     bsz*1024*625  bzs*1*25*25     bsz*1024*625
        # 12    bsz*2048*169  bzs*1*13*13     bsz*2048*169
        if idx <= 3:  # bsz*512*2500
            H_1 = 50
            W_1 = 50
            cut_width = int(W_1 * beta_factor)
            cut_height = int(H_1 * beta_factor)
            cut_x1_1 = 16
            cut_x2_1 = cut_x2 // 8
            support_feats[idx][:, :, cut_x1_1:cut_x1_1 + cut_height, cut_x1_1:cut_x1_1 + cut_width] = assist_feats[idx][0, :,
                                                                                           cut_x2_1:cut_x2_1 + cut_height,
                                                                                           cut_x2_1:cut_x2_1 + cut_width]
        elif 3<idx<=9:
            H_2 = 25
            W_2 = 25
            cut_width = int(W_2 * beta_factor)
            cut_height = int(H_2 * beta_factor)
            cut_x1_2 = 8
            cut_x2_2 = cut_x2 // 16
            support_feats[idx][:, :, cut_x1_2:cut_x1_2 + cut_height, cut_x1_2:cut_x1_2 + cut_width] = assist_feats[idx][
                                                                                                      0, :,
                                                                                                      cut_x2_2:cut_x2_2 + cut_height,
                                                                                                      cut_x2_2:cut_x2_2 + cut_width]
    return support_feats, support_mask


def cutmixFeature4query_segmentation(queryImgs, queryMasks, img2, mask2, beta_factor=0.4):
    # BatchSize Channel Height Weight
    bsz, C, H, W = queryImgs.shape
    cut_width = int(W * beta_factor)
    cut_height = int(H * beta_factor)
    partMaskList = []
    # queryImgs[:, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = img2[:, cut_x2:cut_x2 + cut_height,
    #                                                                          cut_x2:cut_x2 + cut_width]
    # queryMasks[:, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
    #                                                                        cut_x2:cut_x2 + cut_width]
    # partMask = mask2[cut_x2:cut_x2 + cut_height, cut_x2:cut_x2 + cut_width]
    for i in range(bsz):
        cut_x1 = 128
        cut_x2 = np.random.randint(0, W - cut_width)
        queryImgs[i, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = img2[:, cut_x2:cut_x2 + cut_height,
                                                                                 cut_x2:cut_x2 + cut_width]
        queryMasks[i, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
                                                                               cut_x2:cut_x2 + cut_width]
        # 找到目标域嵌入源域的部分mask 并 返回
        partMaskList.append(mask2[cut_x2:cut_x2 + cut_height, cut_x2:cut_x2 + cut_width])
    return queryImgs, queryMasks, partMaskList


def cutmix4support_segmentation(supportImgs, supportMasks, img2, mask2, beta_factor=0.4):
    # BatchSize shot Channel Height Weight
    bsz, shot, C, H, W = supportImgs.shape
    cut_width = int(W * beta_factor)
    cut_height = int(H * beta_factor)
    for i in range(bsz):
        cut_x1 = 128
        cut_x2 = np.random.randint(0, W - cut_width)
        supportImgs[i, :, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = img2[:,
                                                                                      cut_x2:cut_x2 + cut_height,
                                                                                      cut_x2:cut_x2 + cut_width]
        supportMasks[i, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
                                                                                    cut_x2:cut_x2 + cut_width]
    return supportImgs, supportMasks


def cutmix4query_segmentation(queryImgs, queryMasks, img2, mask2, beta_factor=0.4):
    # BatchSize Channel Height Weight
    bsz, C, H, W = queryImgs.shape
    cut_width = int(W * beta_factor)
    cut_height = int(H * beta_factor)
    partMaskList = []
    # queryImgs[:, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = img2[:, cut_x2:cut_x2 + cut_height,
    #                                                                          cut_x2:cut_x2 + cut_width]
    # queryMasks[:, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
    #                                                                        cut_x2:cut_x2 + cut_width]
    # partMask = mask2[cut_x2:cut_x2 + cut_height, cut_x2:cut_x2 + cut_width]
    for i in range(bsz):
        cut_x1 = 128
        cut_x2 = np.random.randint(0, W - cut_width)
        queryImgs[i, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = img2[:, cut_x2:cut_x2 + cut_height,
                                                                                 cut_x2:cut_x2 + cut_width]
        queryMasks[i, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
                                                                               cut_x2:cut_x2 + cut_width]
        # 找到目标域嵌入源域的部分mask 并 返回
        partMaskList.append(mask2[cut_x2:cut_x2 + cut_height, cut_x2:cut_x2 + cut_width])
    return queryImgs, queryMasks, partMaskList


imgPath = r"F:\WHT\CDFSSDataSet\AssistDataSet\ChestX\Img"
maskPath = r"F:\WHT\CDFSSDataSet\AssistDataSet\ChestX\Mask"


def randomSelect4Lung():
    imgDir = os.listdir(imgPath)
    random_number = int(random.random() * 1000)
    idx = random_number % len(imgDir)
    # print(idx)

    img = imgPath + "\\" + imgDir[idx]
    mask = maskPath + "\\" + imgDir[idx]

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.529, 0.424, 0.425]
    H, W = 400, 400
    img = Image.open(img).convert('RGB')
    mask = read_mask(mask)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((H, W)),
        transforms.Normalize(img_mean, img_std)
    ])
    img_tensor = transform(img)
    mask_tensor = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (H, W), mode='nearest').squeeze()
    return img_tensor, mask_tensor


def read_mask(img_name):
    mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
    mask[mask < 128] = 0
    mask[mask >= 128] = 1
    return mask

# a, b = randomSelect4Lung()
#
# print(a.shape)
# print(b.shape)
#
#
# a = a.permute(1, 2, 0).numpy()
# plt.imshow(a)
# plt.show()
#
# plt.imshow(b)
# plt.show()
