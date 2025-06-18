# FruitNinja Vision
Detect the position and velocity of an orange ping pong ball and predict where it will intersect with the ground plane.

This repository was written for Azure Kinect DK and Franka Emika Panda (the setup in the 16-662 Autonomy Lab at CMU).

This project was completed by Thomas Detlefsen, Yatharth Ahuja, Kasina Euchukanonchai, and Austin Windham. The project video can be viwed here https://drive.google.com/file/d/1x_GPk81O9758dPhHsjKNldnvr-XTszXD/view?usp=sharing.

## Setup
Navigate to the `src` folder of your workspace and clone this repository.

```
cd ~/catkin_ws/src
git clone git@github.com:t-detlefsen/fruit-ninja-vision.git
```

Next update [config/azure_easy.yaml](config/azure_easy.yaml) with the values obtained from eye-on-base camera calibration. The repository [easy_handeye](https://github.com/IFL-CAMP/easy_handeye) is an extremely useful tool that can be used for this.

The model (YOLO V8n) included in this repository was trained using the Azure Kinect DK with an exposure time of 2500 us and brightness of 145. The only object used in training was an orage ping pong ball. If desired, a new model can be trained using [the documentation from Ultralytics](https://github.com/ultralytics/ultralytics).


## Usage
In order to launch the node, run:
```
roslaunch fruit_ninja_vision track-fruit.launch visualize:=True
```

This will launch the camera driver, transforms, and object prediction nodes along with the visualization shown below. The orange sphere represents the ping pong ball and the red disk is the estimation of the impact position

<img src="https://github.com/t-detlefsen/fruit-ninja-vision/blob/main/figures/visualization.png" width="650">

Debugging arguments and other parameters for the node can be set in [config/default.yaml](config/default.yaml).

## Credits
Thanks to [auto_jenga](https://github.com/vib2810/auto_jenga) for easy_tf_publisher.py