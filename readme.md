# smartforceps-AI-models

## About this repo

AI modeling framework for SmartForceps, a sensorized surgical bipolar forceps capable of quantifying the forces of tool-tissue interaction in microsurgery.

## How to run the codes locally

To run this app locally, clone this repository and open the pipeline folder for each model in your terminal/Command Prompt. We suggest you to create a virtual environment for installation of required packages for this pipeline.

```
Example:
cd smartforceps_segmentation_model

python3 -m venv .venvs/dash
```
In Unix System:
```
source .venvs/dash/bin/activate
```

In Windows: 
```
.venvs/dash\Scripts\activate
```

Install all required packages by running:
```
pip install -r requirements.txt
```

Run this app locally by:
```
python -W ignore run.py
```