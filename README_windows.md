# Running the Prion Pipeline — Windows

## Requirements
- Python 3.11 — download from https://www.python.org/downloads/
  - During installation, check **"Add Python to PATH"**

---

## Setup (run once)

Open **Command Prompt** (search for `cmd` in the Start menu) and navigate to this folder:
```
cd %USERPROFILE%\Downloads\qCMB_hackathon_prions
```

Create and activate a virtual environment:
```
python -m venv venv
venv\Scripts\activate
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

Results are saved to the `results\` folder.

---

## Notes
- Training on CPU takes several hours. To speed it up, open `config.yaml` and reduce `epochs` (e.g. to `10`) and `image_size` (e.g. to `384`).
- The pipeline automatically uses a GPU if one is available.
- If you see a permissions error on `venv\Scripts\activate`, open PowerShell as Administrator and run: `Set-ExecutionPolicy RemoteSigned`
