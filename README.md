# Background
This project is a prototype for a computer vision application (Python and openCV) to be used with a two-camera setup installed on a pushback tractor (airport vehicle to push back the aircraft from its parking position). The goal was to analyse the application of object detection and depth computation (computer vision) in a driver assistance system that supports the docking process of a remotely controlled pushback tractor by indicating the distance to the aircraft depending on the tractors position to the operator.

The main challenge of the project was to create a calibration algorithm, which selects the best combination of images, from a range of images taken with each camera, to optimise the camera calibration for each individual camera as well as the stereo setup.
By doing so, the error of the computed disparity map and the depth map were significantly reduced. 
The software uses a deep learning model (using transfer learning) to identify aircraft engines.


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
https://github.com/oreillymedia/Learning-OpenCV-3_examples
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model
https://github.com/nicknochnack/TFODCourse



