# step 2 in Documentation
import pandas as pd 
from PIL import Image
import os

#create standard dirs 
dir_input = "data/input"
dir_output = "data/clean"
cwd = os.getcwd()

annotations = pd.read_csv(f'{dir_input}/Annotations.csv')

existing_region_list = []
existing_qid_list = []

#check which annotations actually exist 
for i, r in annotations.iterrows(): 
    region = r['region']
    qid = r['quadratid']

    file_path = f"{dir_input}/{region}/{qid}.jpg"
    isExist = os.path.exists(file_path)

    if isExist == True: 
        existing_qid_list.append(qid)
        print(f'The {qid} from {region} does exist in the dataset')
    else: 
        print(f'The {qid} from {region} does not exist in the dataset')

#removing all non-existing qid's from the dataset
existing_annotations = annotations.loc[annotations['quadratid'].isin(existing_qid_list)] 

#formatting and saving 
existing_annotations = existing_annotations.reset_index()
existing_annotations = existing_annotations.drop(columns=['Unnamed: 0', 'index'])
existing_annotations.to_csv(f'{dir_output}/existing_annotations.csv')

#doing random sample of 10.000 annotations. 
occur = existing_annotations.groupby(['func_group']).size()
sample_annotations = existing_annotations.groupby('func_group', group_keys=False).apply(lambda x: x.sample(2000))

#formatting and saving 
sample_annotations = sample_annotations.reset_index()
sample_annotations = sample_annotations.drop(columns=['index'])
sample_annotations.to_csv(f'{dir_output}/sampled_annotations.csv')