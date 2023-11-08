<p align="center">

  <h1 align="center">🗿 Cavemens: A simple approach for the Robot Perception project</h1>
  
  <h4 align="center"><a href="https://github.com/gabriel-bronfman">Gabriel Bronfman</a>, <a href="https://github.com/iamshubhamgupto">Shubham Gupta</a></h4>
  
  <h5 align="center">&emsp; <a href="https://docs.google.com/presentation/d/1YwzE0sU7YGphtKLduPHkAlaf1JeGr86exQJ9t-L-AA8/edit?usp=sharing"> Slides</a> | <a href=""> Report [Coming Soon] </a> | <a href=""> Code [Coming Soon] </a></h5>

  <!-- Images container -->
  <div style="align-items: center;">
    <div style="display: flex; justify-content: center;">
        <!-- First image with title -->
        <div style="margin-right: 10px;">
        <img src="./assets/teaser.png" alt="First Image" style="width: auto; height: auto;"/>
        </div>
    </div>
    <b>(a)</b> The data structure overview of how mapping is done in the exploration phase. <b>(b)</b> The top 12 resulting targets and their corresponding (x,y,w) which is the displacement in the x, y axis and rotation. The bottom four images show the target front, right, back and left view respectively.
</div>
</p>


## Abstract
    
This repository contains the source code for submission made by the team `Shoob and Goob`. The goal of the project is to map and localize views in previously unexplored 3D worlds. We implement the Dead Reckoning algorithm to keep track of the movement from the given origin. We extract features of the 3D world using the ORB-SLAM3 feature extractor which is then used to train a Visual Bag of Words on all scenes seen by the robot during the exploration phase. We are able to localize the target scene before the navigation phase begins with the coordinates and rotation.

## Installation
**NOTE**: The instructions have been tested on a M1 Macbook pro.

### Environment
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd midterm_rp
conda env create -f environment.yml
conda activate game
```

### Additional 
Our codebase relies on [Python-orb-slam3](https://github.com/mnixry/python-orb-slam3) for the feature extractor. The codebase is compiled for `amd64` architecture but can be compiled for `arm64` using `poetry`.
```commandline
conda activate game
git clone https://github.com/mnixry/python-orb-slam3
poetry build --format wheel
python -m pip install dist/*.whl
python -vc "from orb_slam3 import ORBExtractor"
```

## Play
Play using the default keyboard player
```commandline
conda activate game
python player.py
```

### Key bindings
| <b> Key </b>    | <b> Bindings </b>                               |
| -------------           | -------------                                     |
| ↑                | move forward             |
| ↓             | move backward           |
| ←                   | rotate left              |
| →                | rotate right          |
| esc          | change game phase  |
| space bar        | check         |
| p  | process visual bag of words |
| r  | reset position value |
| t  | reset rotation value |

## References
The project is built from the [starter code](https://github.com/ai4ce/vis_nav_player) released by Professor Chen Feng and the TAs.
