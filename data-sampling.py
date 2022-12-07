# step 2 in Documentation
import pandas as pd 
from PIL import Image
import os

# Do a random sample of 10.000 annotations. 
# 
# Make sure the coral types are balanced. 
# 
# Use the random annotations to create patches and save those to a new folder in the drive. 
# 
# This will be your dataset.

def create_patch(im, x, y): 
    #calculate bounding box for patch given coordinates 
    left = x-50
    upper = y-50
    right = x+50
    lower = y+50
    box = (left, upper, right, lower)

    #create a patch
    subim = im.crop(box)
    print(f'I have just created a patch for {x}, {y}')
    return(subim)


#create standard dirs 
dir_input = "data\input"
dir_output = "data\clean"
cwd = os.getcwd()

#open annotations file --> here you should put the directory to your sample annotations file 
annotations = pd.read_csv(r"C:\Users\20191678\OneDrive - TU Eindhoven\AI\FruitPunch Bootcamp 2022\Capstone Project\fruitpunch_coral_classification\data\input\Annotations.csv")
occur = annotations.groupby(['func_group']).size()
print(occur)
sample_annotations = annotations.groupby('func_group', group_keys=False).apply(lambda x: x.sample(2000))
occur = sample_annotations.groupby(['func_group']).size()
shape = sample_annotations.shape

for i, r in sample_annotations.iterrows(): 
    #get the correct image
    region = r['region']
    qid = r['quadratid']
    #file_path = f"{cwd}\{dir_input}\{region}\{qid}.jpg"
    file_path = os.path.join(cwd,dir_input,region,str(qid)+".jpg")
    if os.path.exists(file_path):
        im = Image.open(file_path)
        x = r['x']
        y = r['y']
        #get the coordinates of the sample 
        subim = create_patch(im, x, y)
        
        #save the sample
        subim.save(f"data\clean\{qid}_{x}_{y}.jpg", "JPEG")
        print(f'I have just saved the patch {x}, {y} for image {qid} from {region}')
    else:
        print("Path does not exist")
    

