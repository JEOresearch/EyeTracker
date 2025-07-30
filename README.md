# EyeTracker
A lightweight, robust Python eye tracker

This repository is an open-source eye tracking algorithm written in Python. Currently, it is an updated version of the pupil tracker from https://github.com/YutaItoh/3D-Eye-Tracker/blob/master/main/pupilFitter.h that has been optimized and simplified. 

To use the script, run "python .\OrloskyPupilDetector.py" from your shell. If the hardcoded file path in the select_video() function does not find a video at the specified path, it will open a browse window that allows you to select a video. The process_video() function handles the majority of the processing and can be easily modified to work with a camera capture or image. It returns a rotated_rect that represents the pupil ellipse. A lite version is also included that is more efficient, but less robust. Be sure to have an adequate light source for the lite version. 

A test video (eye_test.mp4) is included in the root directory for testing. Algorithm details are explained here: https://www.youtube.com/watch?v=bL92JUBG8xw

When running the script on this test video, your results should look like this: https://youtu.be/B06cUMplDHw.  

If you need an eye camera with custome LEDs, I have instructions for building your own IR camera for under $100 here: https://www.youtube.com/watch?v=8lZqCMRMtC8
Alternatively, there is a small $17 IR camera that works well with some modification: https://amzn.to/41x8p2W

To help support this software and other open-source projects, please consider subscribing to my YouTube channel: https://www.youtube.com/@jeoresearch, or joining for $1 per month: https://www.youtube.com/@jeoresearch/join. 

Other useful tools/supplies 
- Spinel Camera (~$100): https://amzn.to/3D8faQB (price may vary)
- LEDs for Spinel ($8): https://amzn.to/41rEwAS
- USB Extension cable x2 ($8): https://amzn.to/4knyf1N (for extending GC0308 cable)

Affiliate links on this page help support the channel at no extra cost to you. As an Amazon Associate, I earn from qualifying purchases. All earnings support the development of open-source software and projects like this! 

Requirements:
- A Python environment

Packages
- numpy ****There is a known issue with numpy 2.0.0. Downgrading to 1.26.0 or another version can solve this issue.
- opencv

Assumptions
- Works best with 640x480 videos. Images will be cropped to size equally horizontally/vertically if aspect ratio is not 4:3.
- The image must be that of the entire eye. Dark regions in the corners of the image (e.g. VR display lens borders) should be cropped. 
