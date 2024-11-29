import os.path
from glob import glob
import shutil
from PIL import Image
ImgDir = r"F:\WHT\CDFSSDataSet\AssistDataSet\ISIC\Img\\"
MaskDir = r"F:\WHT\CDFSSDataSet\AssistDataSet\ISIC\Mask\\"

ImgsName = ['ISIC_0000000.jpg','ISIC_0000031.jpg','ISIC_0012086.jpg']
MasksName = ['ISIC_0000000.png','ISIC_0000031.png','ISIC_0012086.png']

for i in range(len(ImgsName)):

    ImgName = ImgsName[i]
    MaskName = MasksName[i]

    img = Image.open(ImgDir+ImgName).convert('RGB')
    mask = Image.open(MaskDir+MaskName).convert('RGB')

    for i in range(1,4):
        img1 = img.rotate(90*i,expand=True)
        saveName = ImgDir+ImgName.split(".")[0]+"_"+str(90*i)+".jpg"
        img1.save(saveName)

        mask1 = mask.rotate(90 * i,expand=True)
        saveName = MaskDir + MaskName.split(".")[0] + "_" + str(90 * i) + ".png"
        mask1.save(saveName)

    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    for i in range(0,4):
        img1 = mirrored_img.rotate(90*i,expand=True)
        saveName = ImgDir+ImgName.split(".")[0]+"_m"+str(90*i)+".jpg"
        img1.save(saveName)

        mask1 = mask.rotate(90 * i,expand=True)
        saveName = MaskDir + MaskName.split(".")[0] + "_m" + str(90 * i) + ".png"
        mask1.save(saveName)




