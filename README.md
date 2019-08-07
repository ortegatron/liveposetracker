# Live tracker made with Open Pose and Deep SORT

This trackers detects bodies on the video captured by the main camera, assigns an id to each seen person and keep track of that person along the video.

<p align="center">
    <img src="demo.gif", width="480">
</p>

This is the same code used for the analysis of the video, but working directly on the camera input. I can't test the code now but some people asked for it so here it is. Should be enough to clarify ideas about the OpenPose and Deep SORT integration.

Updated to work with OpenPose v1.5.0! Make sure you installed the Python API.

See more at https://www.youtube.com/watch?v=GIJjyjeFmF8

* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [Deep SORT](https://github.com/nwojke/deep_sort)

## Set up
### Prerequisites
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - For body detection. Install the Python API!
* OpenCv - Used for image manipulation
* PyGame - Used to simplify the workflow of Input - Output.
* Numpy, ConfigParser, etc.


### Configuration

- **Constants.py**: Screen resolution and tracker parameters.
- **Constants.py**: Openpose parameters.

### Run
On folder src, just do python Twister.py

### System design
Most of the work is done on Input.py. There, the current frame is processed with OpenPose to get body parts detections, and the bounding boxes for those detections are feed into the Deep SORT tracker. These boxes and the given ids are shown on the screen using simple OpenCV.
