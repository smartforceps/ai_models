# SmartForceps-AI-models

## About this repo

AI modeling framework for SmartForceps, a sensorized surgical bipolar forceps capable of quantifying the forces of tool-tissue interaction in microsurgery.

## How to run the codes locally

To run this app locally, clone this repository and open the pipeline folder in your terminal/Command Prompt. We suggest you to create a virtual environment for installation of required packages for this pipeline.

```
Example:

cd ai_models

python3 -m venv .venvs/sf_ai_models
```
In Unix System:
```
source .venvs/sf_ai_models/bin/activate
```

In Windows: 
```
.venvs/sf_ai_models\Scripts\activate
```

Install all required packages by running:
```
pip install -r requirements.txt
```

Run each modeling codes locally by navigating to the folder:
```
cd smartforceps-segmentation-model

python -W ignore run.py
```