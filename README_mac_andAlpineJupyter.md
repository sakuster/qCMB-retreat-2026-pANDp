# Running the Prion Pipeline — Mac

## Requirements
- Python 3.11 — download from https://www.python.org/downloads/ if not installed

---

## Setup (run once) DOUBLE CHECK the name of the folder in downloads matches below

Open **Terminal** or **Alpine Jupyter Terminal** and navigate to this folder:
```
cd ~/Downloads/qCMBRetreat26/qCMB-retreat-2026-pANDp
```

Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Fix macOS security quarantine (required):
```
xattr -r -d com.apple.quarantine venv/
```

Install dependencies:
```
pip install -r requirements.txt
```

---


##IMPORTANT: Make sure to change the paths and directory names as well as user names in the following files:
- run_job.sh (user name)
- config_alpine.yaml (data and output directories, usernames in both directories)

  

## Run the pipeline

```
python run.py
```

##If not enough space/cache

```
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python run.py --config config_alpine.yaml
```

Results are saved to the `results/` folder.

---

## Notes
- Training on CPU takes several hours. To speed it up, open `config.yaml` and reduce `epochs` (e.g. to `10`) and `image_size` (e.g. to `384`).
- The pipeline automatically uses a GPU if one is available.

##To use pytorch with the gpu

```
import torch 
import torch.nn as nn 
net = nn.Sequential(nn.Conv2d(3,16,3),nn.Conv2d(16,3,3)) 
dev = torch.device('cuda') 
net.to(dev)
```

##Check if 'TRUE'

```
torch.cuda.is_available()
```

##Custom parameters for setting up a Jupyter notebook to run this code
- Anaconda Version: 2020.11
- Conda environment: base
- Configureation Type: Custom configuration
- Cluter: Alpine
- Account: csu-general
- Partition: atesting_a100
- QoS: testing
- Time: 1
- Number of cores: 1
- Reservation: None
- gres: gpu
- nodelist: none
- Constraint: none
