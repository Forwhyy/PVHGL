# Patient-Visit-Spanned Hypergraph Learning for EHR-based Diagnosis Prediction

This repository contains the implementation of **Patient-Visit-Spanned Hypergraph Learning for EHR-based Diagnosis Prediction**, a method designed for Electronic Health Record (EHR)-based diagnosis prediction. The model leverages a hypergraph structure that spans both patient and visit levels, capturing complex dependencies in clinical data.

---


## 📁 Project Structure

```

├── process_data.py                 # Preprocess raw EHR datasets 
├── main.py                         # Entry point for training and evaluation
├── convert_datasets_to_pygDataset.py # Convert datasets into hypergraph format
├── data_utils.py                   # Utility functions for data handling
├── dataset.py                      # Dataset loading and transformation
├── eval.py                         # Evaluation metrics 
├── load_other_datasets.py          # Optional support for additional datasets
├── logger.py                       # Logging utility
├── pvhgl.py                        # PvHGL model definition
├── utils.py                        # Miscellaneous utilities
├── requirements.txt                # List of required packages
└── README.md                       # Project documentation
```

---

## ⚙️ Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have `torch` and `torch-geometric` compatible with your system and CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for help.

---

## 🚀 Getting Started

### 1️⃣ Step 1: Preprocess the Data

Before training the model, preprocess the dataset (e.g., MIMIC-III, MIMIC-IV) into the required format:

```bash
python process_data.py
```

This script reads the raw data and converts it into graph structures used by the PvHGL model.

---

### 2️⃣ Step 2: Train the Model

After preprocessing is complete, run the main training script:

```bash
python main.py
```

You can modify hyperparameters and configuration inside `main.py` or add argument parsing as needed.

---

## 📊 Evaluation

The model is evaluated using standard metrics for multi-label classification, including:

- weighted F1-score
- Recall

Results are printed during training and logged using the `logger.py` module.

---

## 📌 Notes

- Datasets are not included in this repository. Please download and prepare them according to instructions in `process_data.py` 
- Ensure correct formatting of EHR records before processing.

```

