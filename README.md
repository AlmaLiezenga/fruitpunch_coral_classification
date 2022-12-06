# fruitpunch_coral_classification
This is the code for the toy challenge Coral Classification for the Fruitpunch AI bootcamp 2022. 

This project uses DVC to connect to a remote storage in the form of a shared Google drive folder. 

To run this project you should do the following: 
* create a vertual environment by navigating to the folder your project is in in your terminal and running: “`python3 -m venv .venv“`
* activate the virtual environment by running `source .venv/bin/activate`
* install the requiremnets using `pip3 install -r requirements.txt`
* to activate the Google drive remote storage for your device 
* edit the code and data!

When you complete your changes, and in-between to keep track of versions, you should run the following code to store your version 
* to sync with your DVC environment `dvc commit` and `dvc push` this will push the new data towards the remote storage in Google drive 
* to sync with Git `git add .`, `git commit -m "A description of the changes I made"` and finally `git push`
