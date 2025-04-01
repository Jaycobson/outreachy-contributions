# 🔬 Blood-Brain Barrier (BBB) Permeability Prediction 

## 🎯 Project Overview
In this project, a machine learning model is trained to predict whether drug compounds can cross the blood-brain barrier, a critical factor in developing effective treatments for neurological disorders. Using the BBB_Martins dataset from Therapeutics Data Commons (TDC), I've built a comprehensive pipeline for data acquisition, visualization, molecular feature extraction and model development.

The blood-brain barrier is a selective, semi-permeable membrane that protects the brain and central nervous system (CNS) from harmful materials in the bloodstream. BBB permeability of molecules is crucial in drug discovery and development for several reasons:

- **CNS Drug Development**: Drugs targeting neurological disorders must cross the BBB to be effective
- **Neurotoxicity Screening**: Identifying compounds that may cause CNS side effects
- **Drug Repurposing**: Finding existing drugs that might treat brain disorders
- **Early-Stage Drug Design**: Guiding medicinal chemists in developing BBB-permeable compounds

This binary classification model accepts SMILES notations of compounds as input and predicts BBB permeability (1 for permeable, 0 for non-permeable) based on molecular properties and structural features.

## 📊 Dataset Significance
The BBB_Martins dataset is vital for neuropharmaceutical research:

| Feature | Description |
|---------|-------------|
| Dataset Source | [Therapeutics Data Commons (TDC)](https://tdcommons.ai/single_pred_tasks/adme#bbb-blood-brain-barrier-martins-et-al) |
| Research Papers | [1]. [Martins, Ines Filipa, et al. *Journal of Chemical Information and Modeling*, 52.6 (2012): 1686-1697](https://pubmed.ncbi.nlm.nih.gov/22612593/). <br> [2].  [Wu, Zhenqin, et al. “MoleculeNet: a benchmark for molecular machine learning.” Chemical science 9.2 (2018): 513-530](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a).|
| Biological Barrier | Blood-Brain Barrier - critical for CNS drug development |
| Measurement | Binary permeability outcome (can/cannot cross BBB) |
| Data Quality | Well-curated but imbalanced dataset with multiple splits for robust validation |
| Applications | Essential for developing treatments for Alzheimer's, Parkinson's and brain cancer<br> [Abbott NJ, Patabendige AA, Dolman DE, Yusof SR, Begley DJ. “Structure and function of the blood–brain barrier.” Neurobiology of Disease, 37(1): 13–25, 2010. doi: 10.1016/j.nbd.2009.07.030](https://www.sciencedirect.com/science/article/abs/pii/S0969996109002083)|

## 🔍 Understanding the Data
### Experimental Background
- **Task Type**: Binary classification
- **Measurement**: Ability to penetrate the blood-brain barrier
- **Biological Relevance**: Essential for CNS drug development
- **Prediction Goal**: Classify compounds as BBB-permeable (1) or non-permeable (0)

### 📈 Dataset Composition
| Category | Details |
|----------|---------|
| Total Samples | 2030 compounds |
| BBB-Permeable (Y=1) | 1551 drugs (76.41%) |
| Non-Permeable (Y=0) | 479 drugs (23.59%) |
| Data Format | CSV with Drug_ID, Drug (SMILES) and Y (binary outcome) |
| Class Imbalance | Majority class (permeable) represents 76.41% of data |
| Data Splits | Train, validation, and test sets provided for model development |

## 📂 Project Structure
```
bbb-prediction/
├── data/                    # Stores all data
│   ├── download_data/       # Stores downloaded data from tdc       
│   ├── featurized_data/     # Processed molecular features data for each split
│   ├── results/             # Model evaluation results
│   └── plots/               # Performance visualizations
├── models/                  # Saved model files
│   ├── xgboost/             # XGBoost models
├── notebooks/               # Stores notebooks
│   ├── analysis.ipynb       # Main notebook for visualization and analysis
│   ├── exploration.ipynb    # Initial data exploration
│   └── evaluation.ipynb     # Detailed model evaluation
├── scripts/
│   ├── download_data.py     # Dataset acquisition script
│   ├── featurize_data.py    # Molecular featurization script
│   ├── modelling.py         # Model training and evaluation
├── environment.yml          # Conda environment specification
├── .gitignore
├── LICENSE
└── README.md                # Project documentation
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.12.9 or later
- Miniconda/Anaconda
- Ubuntu OS or WSL (if system OS is Windows)
- Docker (for Ersilia models)
- Git

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/jaycobson/bbb-prediction.git
cd bbb-prediction
```

2. **Set up the conda environment**
```bash
# Create and activate the environment using the yml file
conda env create --file env.yml
conda activate bbb
```

The `env.yml` file includes all necessary dependencies:
- ersilia (for molecular featurization)
- rdkit
- scikit-learn
- xgboost
- lightgbm
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

## 📈 **Dataset Acquisition**
```bash
# To download it, run:
python scripts/download_data.py --dataset BBB_Martins
```
You need to specify the dataset you want to download, the default on the CLI is BBB_Martins. If you need to download another kind of data, kindly specify the dataset name.

This script:
- Downloads the BBB_Martins dataset from TDC
- Creates train/validation/test splits 
- Saves data to `data/download_data/`

## 🧪 Data Exploration and Visualization

To explore the dataset and understand the molecular properties:

```bash
Navigate notebooks -> analysis.ipynb 
```

The notebook includes:
- Visualizing drug distributions
- Molecular structure representation using RDKit
- Checking for duplicates and data integrity
- Class distribution analysis
- Chemical property visualization (molecular weight, LogP, etc.)

## 🚀 Molecular Featurization
This project leverages the Uni-Mol molecular representation model from the [Ersilia Model Hub](https://www.ersilia.io/model-hub) for advanced molecular featurization. The model is also available in this [github repository](https://github.com/ersilia-os/eos39co).

Uni-Mol employs an SE(3) equivariant transformer architecture, designed to capture intricate 3D molecular structures. Trained on over 200 million conformations, it generates high-quality molecular embeddings, enhancing predictive performance in cheminformatics and drug discovery applications. Its model id is eos39co in the Ersilia Model Hub

### Setup Ersilia Model
```bash
# Install Ersilia if not already installed
pip install ersilia

# Fetch and serve the model
ersilia fetch eos39co
ersilia serve eos39co
```

### Generate Features
```bash
# Process SMILES strings into numerical features, the model id needs to be set to the one you want, the default in this repo is eos39co.
python scripts/featurize_data.py --model_id eos39co
```

This script:
- Serves the Uni-Mol (eos39co) model for molecular featurization.
- Reads molecular data from CSV files in data/download_data.
- Extracts 3D molecular representations using the model.
- Handles errors and missing files with logging.
- Supports command-line arguments to specify the model ID.
- Ensures output directories exist before processing.
- Runs the featurization process for all available input files.
- Saves featurized outputs to `data/featurized_data/`

### Feature Output Structure
```
data/
├── featurized_data/
│   ├── BBB_Martins_train_features.csv
│   ├── BBB_Martins_valid_features.csv
│   ├── BBB_Martins_test_features.csv
```

## 🧠 Model Development

### Available Models
The project implements several machine learning approaches:

1. **XGBoost** (Default): Gradient boosting framework known for performance on tabular data
2. **LightGBM**: Light Gradient Boosting Machine, optimized for efficiency
3. **Logistic Regression**: Baseline linear model for comparison

The default model for this project is XGBoost but you can use lightgbm or logistic regression by using the indentifier lightgbm and logistic respectively.

### Training the Model
```bash
python scripts/modelling.py --model xgboost
```

Command line options:
- `--model [xgboost|lightgbm|logistic]`: Select model type

### Model Configuration
#### XGBoost Configuration  
| Parameter             | Value  | Purpose                         |  
|-----------------------|--------|---------------------------------|  
| **Learning Rate**     | 0.01   | Controls step size in updates for gradual learning |  
| **Max Depth**        | 6      | Limits tree complexity to prevent overfitting |  
| **Min Child Weight**  | 1      | Minimum sum of instance weight needed in a child node |  
| **Subsample**        | 0.8    | Randomly samples data to prevent overfitting |  
| **Col Sample by Tree** | 0.8   | Controls feature sampling for more robust trees |  
| **N Estimators**      | 500    | Number of boosting rounds (trees) |  
| **Early Stopping**    | 50 rounds | Stops training when no improvement is seen |  
| **Objective**        | binary:logistic | Optimized for binary classification |  
| **Eval Metric**      | auc    | Uses AUC (Area Under Curve) for evaluation |  
| **Random State**      | 42     | Ensures reproducibility of results |  
| **Use Label Encoder** | False  | Avoids using deprecated label encoding |

## 📊 Model Evaluation

### Performance Metrics
The model is evaluated using multiple metrics to ensure robust assessment:

| Metric | Description | Importance for BBB Prediction |
|--------|-------------|--------------------------|
| Accuracy | Overall prediction correctness | Baseline metric |
| Precision | Positive predictive value | Critical for identifying true BBB+ compounds |
| Recall | True positive rate | Important for not missing potential BBB+ drugs |
| F1-Score | Harmonic mean of precision and recall | Balanced measure for imbalanced data |
| ROC-AUC | Area under ROC curve | Overall discrimination ability |
| PR-AUC | Area under precision-recall curve | Better for imbalanced datasets |

### Performance Visualization
The evaluation script generates detailed visualizations in `data/plots/`:

- **ROC Curves**: True vs false positive rates
- **Confusion Matrices**: True vs predicted permeability
- **Precision-Recall Curves**: Better for imbalanced datasets
- **Learning Curves**: Training and validation metrics
- **Feature Importance Plots**: Most influential molecular properties


## 📋 Logging and Monitoring
- Dataset download logs: `~/dataset_download.log`
- Model training logs: `models/bbb_training.log`
- All processes include detailed logging for monitoring

## 🔮 Future Work
- Implement deep learning approaches (Graph Neural Networks)
- Incorporate additional datasets for transfer learning
- Add interpretability tools for medicinal chemists
- Develop a web interface for prediction
- Include structural alerts for BBB-impermeable compounds

## 🙏 Acknowledgments
- [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) for the BBB permeability dataset
- [Ersilia Model Hub](https://ersilia.io/) for molecular featurization tools
- [RDKit](https://www.rdkit.org/) for molecular visualization and processing
- Contributors to the open-source ML libraries used in this project


