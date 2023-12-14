<p align="center">

  <h1 align="center">🗿 Cavemen: A prehistoric approach for Mapless Navigation</h1>

  <h4 align="center"><a href="https://github.com/gabriel-bronfman">Gabriel Bronfman</a>, <a href="https://github.com/iamshubhamgupto">Shubham Gupta</a></h4>

  <h5 align="center">&emsp; <a href="https://docs.google.com/presentation/d/1VS014bRyqm4rN3s_tJweIRwDdKjDeuX9mA59doqHDfc/edit?usp=sharing"> Slides</a> | <a href=""> <a href="https://gabriel-bronfman.github.io/cavemen/">Project Page</a></h5>

  <!-- Images container -->
  <div style="align-items: center;">
    <div style="display: flex; justify-content: center;">
        <!-- First image with title -->
        <div style="margin-right: 10px;">
        <img src="./assets/gif/teaser.gif" alt="First Image" style="width: auto; height: auto;"/>
        </div>
    </div>
    <b>(a)</b> The data structure overview of how mapping is done in the exploration phase. <b>(b)</b> The top 12 resulting targets and their corresponding (x,y,w) which is the displacement in the x, y axis and rotation. The bottom four images show the target front, right, back and left view respectively.
</div>
</p>

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

## Play
Play using the default keyboard player
```commandline
conda activate game
python player.py
```
### Redis environment
We use Python environment files to store redis database details. A sample `.env` for the project is provided below
```
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=robot_interface
```

## Development
We use [pre-commit](https://pre-commit.com/index.html) to format and style our code. To contribute to this project, first clone the repository and activate the environment. Then run the following:
```
pre-commit install
```
Once the hooks are installed, continue committing to the repository as usual. The first commit will be slow.

### To update the environment
```
conda env export --no-builds | grep -v "prefix" > environment.yml
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

## Acknowledgments
The project is built from the [starter code](https://github.com/ai4ce/vis_nav_player) released by Professor Chen Feng and the TAs for the course Robot Perception.

## Citation
If you find this work useful, please cite us using (bibtex):
```
@software{Bronfman_Cavemen_A_2023,
author = {Bronfman, Gabriel and Gupta, Shubham},
month = dec,
title = {{🗿 Cavemen: A prehistoric approach for Mapless Navigation}},
url = {https://github.com/gabriel-bronfman/cavemen},
version = {1.0.0},
year = {2023}
}
```
