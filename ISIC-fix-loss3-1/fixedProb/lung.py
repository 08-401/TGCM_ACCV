from glob import glob
import os
rpc = "_mask.png"
# input the Chest-X dir
# fix the problem:
# D:\Datasets\WHT\CDFSSDataSet\TargetDomains\Chest-X\Lung Segmentation\masks\MCUCXR_0399_1.png --> MCUCXR_0399_1_mask.png
base_dir = "F:\WHT\CDFSSDataSet\Target domains\Lung Segmentation\masks\\*.png"
fileNames = glob(base_dir)
for fileName in fileNames:
    if "MCUCXR" in fileName:
        n2 = fileName.replace(".png",rpc)
        os.rename(fileName,n2)