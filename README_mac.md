# Running the Prion Pipeline — Mac

## Requirements
- Python 3.11 — download from https://www.python.org/downloads/ if not installed

---

## Setup (run once)

Open **Terminal** and navigate to this folder:
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
