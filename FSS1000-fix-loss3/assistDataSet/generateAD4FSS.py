import os.path
from glob import glob
import shutil
from PIL import Image
origialDir = r"F:\WHT\CDFSSDataSet\Target domains\FSS-1000\\*"
targetDir = r"F:\WHT\CDFSSDataSet\AssistDataSet\FSS-1000\\"
paths = glob(origialDir)
for i in range(100):
    img = paths[i] + "\\1.jpg"
    mask = paths[i] + "\\1.png"
    img = Image.open(img).convert('RGB')
    mask = Image.open(mask).convert('RGB')
    targetImgDir = targetDir+"Img\\"+str(i)+".png"
    targetMaskDir = targetDir + "Mask\\"+str(i)+".png"

    img.save(targetImgDir)
    mask.save(targetMaskDir)
    # print(path)
# print(len(path))






# ImgsName = ['ISIC_0015233.jpg','ISIC_0015260.jpg','ISIC_0015295.jpg']
# MasksName = ['ISIC_0015233.png','ISIC_0015260.png','ISIC_0015295.png']

# for i in range(len(ImgsName)):
#
#     ImgName = ImgsName[i]
#     MaskName = MasksName[i]
#
#     img = Image.open(ImgDir+ImgName).convert('RGB')
#     mask = Image.open(MaskDir+MaskName).convert('RGB')
#
#     for i in range(1,4):
#         img1 = img.rotate(90*i,expand=True)
#         saveName = ImgDir+ImgName.split(".")[0]+"_"+str(90*i)+".jpg"
#         img1.save(saveName)
#
#         mask1 = mask.rotate(90 * i,expand=True)
#         saveName = MaskDir + MaskName.split(".")[0] + "_" + str(90 * i) + ".png"
#         mask1.save(saveName)
#
#     mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     mirrored_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
#     for i in range(0,4):
#         img1 = mirrored_img.rotate(90*i,expand=True)
#         saveName = ImgDir+ImgName.split(".")[0]+"_m"+str(90*i)+".jpg"
#         img1.save(saveName)
#
#         mask1 = mask.rotate(90 * i,expand=True)
#         saveName = MaskDir + MaskName.split(".")[0] + "_m" + str(90 * i) + ".png"
#         mask1.save(saveName)




