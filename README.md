# Bootcamp Fruitpunch AI 2022: Toy Challenge Coral Classification
This is the code for the toy challenge Coral Classification for the Fruitpunch AI bootcamp 2022.

This project uses [DVC](dvc.org) for data versioning. The data itself are stored on a Google Drive remote.

## How to run

To run this project you should do the following:

1. `git clone` the repository
1. Create a new development branch and check out to it with `git checkout -B <my_branch_name>`
1. Create a virtual environment by navigating to the project directory and running: `python3 -m venv .venv`
1. Activate the virtual environment with `source .venv/bin/activate`
1. Install the requirements with `pip install -r requirements.txt`
1. Use `dvc pull` to pull the data from the remote storage. You may need to authenticate in your browser. There's a lot of data, so this may take a while.
1. Run the modules with `python data_processing.py` (or equivalent for other modules)

## What the scripts do

- `data-sampling.py` takes the annotations file and checks if the images associated with the annotations actually exist. Those that exist are added to a new dataframe `existing_annotations` and saved as a `.csv` in the `data/clean` directory. A random sample with 2000 samples from each of the 5 coral types is then created from these `existing_annotations` and stored as `sample_annotations` and saved as a `.csv` in the `data/clean` directory.
- `data-processing.py` takes the `sampled_annotations` and iterates over the rows, for each annotation it applies a function that takes as input an image and the point locations corresponding with that image. The function returns cropped patches for each point of 100x100 pixels with the point in the center. These images are written to the `data/clean` directory and stored in the appropriate folders per region.
- `data-split.py` takes the `sampled_annotations` and splits the data into training, testing and validation groups. It then iterates over all annotations and puts the subimage belonging to that annotation into the correct folder for training. 

## How to contribute
As described above, make sure you are working on a development branch. When you complete your changes, and in-between to keep track of versions, you should run the following code to store your changes:

- When making changes to datasets or files versioned by DVC: run `dvc commit` to update the DVC cache and `dvc push` to sync the cache with the remote storage. Then run `git add .`, `git commit -m "<description of changes>"`, `and git push`
- When making changes to code: run `git add .`, `git commit -m "<description of changes>"`, `and git push`.
- - The first time you push to your development branch, you'll need to run `git push --set-upstream origin`.

## Known issues
- Still need to create a DVC pipeline
- Access credentials for DVC remote need to be properly configured
