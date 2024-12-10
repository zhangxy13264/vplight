# Introduction

This repository is for VPLight to address the traffic signal control problem in mixed traffic flows.

The code framework is based on [LibSignal](https://github.com/DaRL-LibSignal/LibSignal). It extends the environment to include pedestrian traffic flows. It supports the SUMO simulation environment.

# Install

## SUMO Environment

The traffic simulator uses SUMO. For guidelines, please refer to [SUMO Doc](https://epics-sumo.sourceforge.io/sumo-install.html#).

```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig

git clone --recursive https://github.com/eclipse/sumo

export SUMO_HOME="$PWD/sumo"

mkdir sumo/build/cmake-build && cd sumo/build/cmake-build
cmake ../..
make -j$(nproc)
```

To verify the installation:

```
cd $PWD/sumo/bin
./sumo
```

Add SUMO into the system PATH and test the configuration:

```
export SUMO_HOME=~/$PWD/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

import libsumo
import traci
```

## Requirement

Our code is based on Python version 3.9. Please install Pytorch according to the instructions on [PyTorch](https://pytorch.org/get-started/locally/). The following is an example with CUDA version 11.3.

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

For Colight, the following libraries need to be installed additionally:

```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

# Run Model

Based on LibSignal, we modified world_sumo.py, added tscp_trainer.py and some generators to implement the traffic signal control task for mixed traffic flows. In run.py, you can modify the task to tsc_p to select the mixed traffic flow, choose the agent method, change the road network, and adjust different pedestrian flow configurations through arrival_rate.

```
python run.py
```