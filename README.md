# PCFedH Code Guide

This guide provides detailed instructions on how to install and run experiments using the PCFedH codebase for the PPG and Distracted datasets.

## Installation

To set up the environment for running the experiments, install all the necessary packages with the following command:

```bash
pip install -r requirements.txt
```

## Running Experiments

### PPG Dataset


#### Federated PCL

Start by pretraining the encoder:

```bash
python3 main.py ppg pretrained
```
#### Two-phase Soft Clustered FL

Once the encoder is pretrained, predictions can be made using:

```bash
python3 main.py ppg predict
```

### Distracted Driver Dataset


#### Federated PCL

Start by pretraining the encoder:

```bash
python3 main.py distracted pretrained
```
#### Two-phase Soft Clustered FL

Once the encoder is pretrained, predictions can be made using:
```bash
python3 main.py distracted predict
```

## Results

The results from these experiments will be stored in the `log/exp_id/validation.json` file.
