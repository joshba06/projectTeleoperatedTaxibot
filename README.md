# Configuration and usage

## Information for first installation

### 1. Navigate to a new folder

### 2. Create new virtual environment
python3 -m venv 'name of virtual environment'

### 3. Activate virtual environment 
source 'name of virtual environment'/bin/activate

### 4. Install local python version and kernel in virtual environment
python -m pip install --upgrade pip
pip install ipykernel
python3 -m ipykernel install --user --name='name of virtual environment'

### 5. Clone repository from github
git clone https://github.com/joshba06/Object_Detection.git


## Workflow
### 1. Navigate to the folder containing name of virtual env and "Object_Detection"

### 2. Activate virtual environment
source 'name of virtual environment'/bin/activate

### 3. Work on files / edit etc. Then add, commit and push
git add .
git commit -m 'Some meaningful change'
git push

### 4. If you want to download the latest version of the repository from github
git fetch origin
git reset --hard "COMMIT NUMBER"

# Then close the file and reopen it and the changes will be restored to the online file






