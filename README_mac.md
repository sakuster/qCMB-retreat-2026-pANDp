# Running the Prion Pipeline — Mac

## Requirements
- Python 3.11 — download from https://www.python.org/downloads/ if not installed

---

## Setup (run once)

Open **Terminal** and navigate to this folder:
```
cd ~/Downloads/qCMB_hackathon_prions
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

Results are saved to the `results/` folder.

---

## Notes
- Training on CPU takes several hours. To speed it up, open `config.yaml` and reduce `epochs` (e.g. to `10`) and `image_size` (e.g. to `384`).
- The pipeline automatically uses a GPU if one is available.
