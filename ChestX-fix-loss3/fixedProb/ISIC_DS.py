import pandas as pd
import shutil
data = pd.read_csv("../data/isic/class_id.csv",sep=",",header="infer")
origialPath = r"F:\WHT\CDFSSDataSet\Target domains\ISIC\origial\\"
targetPath = r"F:\WHT\CDFSSDataSet\Target domains\ISIC\ISIC2018_Task1-2_Training_Input\\"
# for i in range(len(data)):

for i in range(len(data)):

    id = data['ID'][i]
    Class = data['Class'][i]
    if Class == "nevus":
        # 这是1类
        sourceImgDir = origialPath + id + ".jpg"
        targetImgDir = targetPath + "1\\" + id + ".jpg"
    if Class=="melanoma":
        # 这是2类
        sourceImgDir = origialPath+id+".jpg"
        targetImgDir = targetPath + "2\\"+ id+".jpg"
    if Class=="seborrheic_keratosis":
        # 这是3类
        sourceImgDir = origialPath+id+".jpg"
        targetImgDir = targetPath + "3\\"+ id+".jpg"
    shutil.copy(sourceImgDir, targetImgDir)

