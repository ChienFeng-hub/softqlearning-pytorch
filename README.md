# softqlearning-pytorch

## Installation

Requirements:
* Anaconda >= 24.1.2
* Python >= 3.9
* PyTorch >= 2.2.1
* MuJoCo >= 3.1.3

```
conda env create -f environment.yml
```

## Usage

```
python train.py
```

## Results
### Multi-goal
<p align="center">
  <img src="https://github.com/ChienFeng-hub/softqlearning-pytorch/blob/main/figures/multigoal_sql.png">                                      
</p>

### MuJoCo
<p align="center">
  <img width=49% src="https://github.com/ChienFeng-hub/softqlearning-pytorch/blob/main/figures/halfcheetah_sql.png">                                      
  <img width=49% src="https://github.com/ChienFeng-hub/softqlearning-pytorch/blob/main/figures/hopper_sql.png">
</p>
<p align="center">
  <img width=32% src="https://github.com/ChienFeng-hub/softqlearning-pytorch/blob/main/figures/ant_sql.png">                                      
  <img width=32% src="https://github.com/ChienFeng-hub/softqlearning-pytorch/blob/main/figures/walker2d_sql.png">
  <img width=32% src="https://github.com/ChienFeng-hub/softqlearning-pytorch/blob/main/figures/humanoid_sql.png">
</p>


