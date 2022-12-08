# step 1 in Documentation

# import libraries 
import pandas as pd 
from PIL import Image

#create standard dirs 
dir_input = 'data/input/'
dir_output = 'data/clean/'

#open annotations file --> here you should put the directory to your sample annotations file 
annotations = pd.read_csv(f'{dir_output}/sampled_annotations.csv')

#check full size to be able to track progress 
total_size = len(annotations.index)

# !improvement would be to not loop through all annotations but instead all images (computationally more efficient)!

#function that takes an image and point locations and returns a 100x100 pixels patch for each point with the point as the center
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

# a loop that uses the function 
for i, r in annotations.iterrows(): 
    #get the correct image
    region = r['region']
    qid = r['quadratid']
    im = Image.open(f'{dir_input}{region}/{qid}.jpg')
    
    x = r['x']
    y = r['y']
    #get the coordinates of the sample 
    subim = create_patch(im, x, y)
    
    #save the sample
    subim.save(f"{dir_output}{region}/{qid}_{x}_{y}.jpg")
    print(f'I have just saved the patch {x}, {y} for image {qid} from {region}')
    
    #print progress
    progress = (i / total_size) * 100
    print(f'I am now at {progress} % of the annotations')