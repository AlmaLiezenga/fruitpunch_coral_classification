from sklearn.model_selection import train_test_split
import pandas as pd 
import shutil 

#create standard dirs 
dir_input = "data/clean"
dir_output = "data/clean/split"

#read annotations file 
annotations = pd.read_csv(f'{dir_input}/sampled_annotations.csv')

X = annotations[['quadratid', 'y', 'x', 'label_name', 'label', 'region']]
y = annotations['func_group']

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, stratify=y)

X_test, X_valid, y_test, y_valid = train_test_split(X_test_val, y_test_val, test_size=0.15, stratify=y_test_val)


train = pd.concat([X_train, y_train], axis=1)
train['group'] = 'train'
print(train)
test = pd.concat([X_test, y_test], axis=1)
test['group'] = 'test'
valid = pd.concat([X_valid, y_valid], axis=1)
valid['group'] = 'valid'

split_annotations = pd.concat([train, test, valid])
print(split_annotations)

split_annotations.to_csv(f'{dir_input}/split_annotations.csv')

#copying the images to new folder, per group
for i, r in split_annotations.iterrows():
    region = r['region']
    qid = r['quadratid']
    x = r['x']
    y = r['y']
    group = r['group']
    func_group = r['func_group']

    jpg_file = f"{dir_input}/per_region/{region}/{qid}_{x}_{y}.jpg"
    dest_dir1 = f"{dir_output}/on-func/{group}/{func_group}/{region}_{qid}_{x}_{y}.jpg"
    shutil.copy(jpg_file, dest_dir1)

    dest_dir2 = f"{dir_output}/on-region/{group}/{region}"
    shutil.copy(jpg_file, dest_dir2)