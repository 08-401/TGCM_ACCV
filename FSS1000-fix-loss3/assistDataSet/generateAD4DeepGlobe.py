import os.path
from glob import glob
import shutil
from PIL import Image
ImgDir = r"F:\WHT\CDFSSDataSet\AssistDataSet\DeepGlobe\Img\\"
MaskDir = r"F:\WHT\CDFSSDataSet\AssistDataSet\DeepGlobe\Mask\\"

ImgsName = ['119_00.jpg','119_02.jpg','606_04.jpg','855_01.jpg','2334_31.jpg','855_30.jpg']
MasksName = ['119_00.png','119_02.png','606_04.png','855_01.png','2334_31.png','855_30.png']

for i in range(len(ImgsName)):

    ImgName = ImgsName[i]
    MaskName = MasksName[i]

    img = Image.open(ImgDir+ImgName).convert('RGB')
    mask = Image.open(MaskDir+MaskName).convert('RGB')

    for i in range(1,4):
        img1 = img.rotate(90*i)
        saveName = ImgDir+ImgName.split(".")[0]+"_"+str(90*i)+".jpg"
        img1.save(saveName)

        mask1 = mask.rotate(90 * i)
        saveName = MaskDir + MaskName.split(".")[0] + "_" + str(90 * i) + ".png"
        mask1.save(saveName)

    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    for i in range(0,4):
        img1 = mirrored_img.rotate(90*i)
        saveName = ImgDir+ImgName.split(".")[0]+"_m"+str(90*i)+".jpg"
        img1.save(saveName)

        mask1 = mask.rotate(90 * i)
        saveName = MaskDir + MaskName.split(".")[0] + "_m" + str(90 * i) + ".png"
        mask1.save(saveName)




