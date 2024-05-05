# EyeTracker
A lightweight, robust Python eye tracker

This repository is an open-source eye tracking algorithm written in Python. Currently, it is an updated version of the pupil tracker from https://github.com/YutaItoh/3D-Eye-Tracker/blob/master/main/pupilFitter.h that has been optimized and simplified. 

To use the script, run "python .\OrloskyPupilDetector.py" from your shell. If the hardcoded file path in the select_video() function does not find a video at the specified path, it will open a browse window that allows you to select a video. The process_video() function handles the majority of the processing and can be easily modified to work with a camera capture or image. 

A test video is included for reference. When running the script on this test video, your results should look like https://youtu.be/B06cUMplDHw.  

Requirements:
- A Python environment

Packages
- numpy
- opencv
