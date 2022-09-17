# Setup information

1. Create a new folder

2. Create new virtual environment inside this folder
```python3 -m venv 'nameOfVirtualEnvironment'```

3. Activate virtual environment 
```source 'nameOfVirtualEnvironment'/bin/activate```

4. Upgrade pip and install jupyter kernel in virtual environment
```python -m pip install --upgrade pip```
```pip install ipykernel```
```python3 -m ipykernel install --user --name='nameOfVirtualEnvironment'```

5. Clone this repository from github
```git clone https://github.com/joshba06/projectTeleoperatedTaxibot.git```

6. Update paths in userSettings.py

7. Install wget
macOS: install homebrew, then wget

8. Install packages
```python3 main.py --initialInstallation True```

# Sources
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model
https://github.com/nicknochnack/TFODCourse



