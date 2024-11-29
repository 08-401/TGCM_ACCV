import os
import torch
import random
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms


def cutmix4support_segmentation(supportImgs, supportMasks, img2, mask2, beta_factor=0.25):
    # BatchSize shot Channel Height Weight
    bsz, shot, C, H, W = supportImgs.shape
    cut_width = int(W * beta_factor)
    cut_height = int(H * beta_factor)

    for i in range(bsz):
        # cut_x = np.random.randint(0, W - cut_width)
        # supportImgs[i, :, :, cut_x:cut_x + cut_height, cut_x:cut_x + cut_width] = img2[:, cut_x:cut_x + cut_height,
        #                                                                           cut_x:cut_x + cut_width]
        # supportMasks[i, :, cut_x:cut_x + cut_height, cut_x:cut_x + cut_width] = mask2[ cut_x:cut_x + cut_height,
        #                                                                          cut_x:cut_x + cut_width]
        cut_x1 = 100
        cut_x2 = np.random.randint(0, W - cut_width)
        supportImgs[i, :, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = img2[:, cut_x2:cut_x2 + cut_height,
                                                                                  cut_x2:cut_x2 + cut_width]
        supportMasks[i, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
                                                                                cut_x2:cut_x2 + cut_width]
    return supportImgs, supportMasks


def cutmix4query_segmentation(queryImgs, queryMasks, img2, mask2, beta_factor=0.4):
    # BatchSize Channel Height Weight
    bsz, C, H, W = queryImgs.shape
    cut_width = int(W * beta_factor)
    cut_height = int(H * beta_factor)

    for i in range(bsz):
        # cut_x = np.random.randint(0, W - cut_width)
        # queryImgs[i, :, cut_x:cut_x + cut_height, cut_x:cut_x + cut_width] = img2[:, cut_x:cut_x + cut_height,
        #                                                                           cut_x:cut_x + cut_width]
        # queryMasks[i, cut_x:cut_x + cut_height, cut_x:cut_x + cut_width] = mask2[ cut_x:cut_x + cut_height,
        #                                                                          cut_x:cut_x + cut_width]
        cut_x1 = 100
        cut_x2 = np.random.randint(0, W - cut_width)
        queryImgs[i, :, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = img2[:, cut_x2:cut_x2 + cut_height,
                                                                             cut_x2:cut_x2 + cut_width]
        queryMasks[i, cut_x1:cut_x1 + cut_height, cut_x1:cut_x1 + cut_width] = mask2[cut_x2:cut_x2 + cut_height,
                                                                           cut_x2:cut_x2 + cut_width]
        # 找到目标域嵌入源域的部分mask 并 返回
        partMask = mask2[cut_x2:cut_x2 + cut_height, cut_x2:cut_x2 + cut_width]
    return queryImgs, queryMasks, partMask


imgPath = r"F:\WHT\CDFSSDataSet\AssistDataSet\ISIC\Img"
maskPath = r"F:\WHT\CDFSSDataSet\AssistDataSet\ISIC\Mask"


def randomSelect4ISIC():
    imgDir = os.listdir(imgPath)
    random_number = int(random.random() * 1000)
    idx = random_number % len(imgDir)
    # print(idx)

    img = imgPath + "\\" + imgDir[idx]
    mask = maskPath + "\\" + imgDir[idx].split(".")[0]+".png"
    # print(img)
    # print(mask)
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
    return img_tensor,mask_tensor


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
