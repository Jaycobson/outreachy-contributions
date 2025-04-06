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
- shap
- imblearn

## 📈 **Dataset Acquisition**
```bash
# To download it, run:
python scripts/download_data.py --dataset BBB_Martins
```
You need to specify the dataset you want to download, the default on the CLI is BBB_Martins. If you need to download another kind of data in ADME, kindly specify the dataset name.

This script:
- Downloads the BBB_Martins dataset from TDC
- Creates train/validation/test splits 
- Saves data to `data/download_data/`

## 🧪 Data Exploration and Visualization

To explore the dataset and understand the molecular properties, You can navigate to the notebooks dir and run the analysis.ipynb which shows you an analysis of the datasets.

```bash
Navigate notebooks -> analysis.ipynb 
```

The notebook includes:
- Visualizing drug distributions
- Molecular structure representation using RDKit
- Checking for duplicates and data integrity
- Class distribution analysis
- Chemical property visualization (molecular weight, LogP, etc.)

## 🧪 Molecular Featurization

This project utilizes two different molecular featurization approaches from the [Ersilia Model Hub](https://www.ersilia.io/model-hub):

### Default Featurizer: Uni-Mol (eos39co)

Uni-Mol is the primary molecular representation model used in this project. This advanced featurizer employs an SE(3) equivariant transformer architecture designed to capture intricate 3D molecular structures. Key characteristics include:

- **Architecture**: SE(3) equivariant transformer for 3D molecular representation
- **Training Data**: Over 200 million molecular conformations
- **Output**: High-dimensional embeddings capturing 3D structural information
- **Strength**: Excellent at representing spatial relationships within molecules
- **Model ID**: eos39co in the Ersilia Model Hub
- **GitHub**: [https://github.com/ersilia-os/eos39co](https://github.com/ersilia-os/eos39co)

Uni-Mol was selected as the default featurizer due to its ability to capture complex 3D structural information crucial for predicting BBB permeability, where spatial arrangements significantly impact barrier penetration.

### Alternative Featurizer: DrugTax (eos24ci)

DrugTax was explored as an alternative featurization approach. Unlike Uni-Mol's focus on 3D structure, DrugTax classifies molecules according to their chemical taxonomy:

- **Approach**: Taxonomy-based classification of molecular structures
- **Input**: SMILES notation
- **Output**: 163-dimensional binary feature vector
- **Features Include**:
  - Organic/inorganic kingdom classification
  - Chemical subclass categorization (0/1 binary classification for each class)
  - Molecular composition metrics (number of carbons, nitrogens, etc.)
- **Model ID**: eos24ci in the Ersilia Model Hub


While Uni-Mol remains the default and recommended featurizer for this project, DrugTax was tested to evaluate whether taxonomic classification could provide valuable alternative insights for BBB permeability prediction.

### Setup and Feature Generation

#### Setup Ersilia Models
```bash
# Install Ersilia if not already installed
pip install ersilia

# Fetch and serve the default Uni-Mol model
ersilia fetch eos39co
ersilia serve eos39co

# Or fetch and serve the alternative DrugTax model if you want to experiment with it
# ersilia fetch eos24ci
# ersilia serve eos24ci
```

#### Generate Features
```bash
# Process SMILES using the default Uni-Mol featurizer
python scripts/featurize_data.py --model_id eos39co

# Or use the alternative DrugTax featurizer
# python scripts/featurize_data.py --model_id eos24ci
```

### Feature Output Structure
```
data/
├── featurized_data/
│   ├── featurized_data_with_eos39co/   # Uni-Mol features (default)
│   │   ├── BBB_Martins_train_features.csv
│   │   ├── BBB_Martins_valid_features.csv
│   │   ├── BBB_Martins_test_features.csv
│   ├── featurized_data_with_eos24ci/   # DrugTax features (alternative)
│   │   ├── BBB_Martins_train_features.csv
│   │   ├── BBB_Martins_valid_features.csv
│   │   ├── BBB_Martins_test_features.csv
```

## 🧠 Model Development

The modeling script provides a comprehensive framework for training and evaluating different models with customizable options for feature processing, dimensionality reduction and handling class imbalance.

### 🛠️ Available Models
The project implements several machine learning approaches:

1. **XGBoost** (Default): Gradient boosting framework known for performance on tabular data
2. **LightGBM**: Light Gradient Boosting Machine, optimized for efficiency
3. **Logistic Regression**: Baseline linear model for comparison

### 🧬 Feature Engineering Options
The pipeline supports multiple molecular featurization methods:

- **eos39co** (Default): Uni-Mol 3D molecular representation from Ersilia Model Hub
- **eos24ci**: DrugTax: Drug taxonomy representation model

### 📊 Training Options

#### Basic Model Training
Train a basic XGBoost model using the default Uni-Mol featurizer. This is the simplest approach to start with.

```bash
python scripts/modelling.py --model xgboost --feat eos39co
```

#### Using Different Model Types
Try LightGBM or Logistic Regression instead of XGBoost to compare performance.

```bash
# Use LightGBM model
python scripts/modelling.py --model lightgbm --feat eos39co

# Use Logistic Regression model
python scripts/modelling.py --model logistic --feat eos39co
```

#### Changing Featurization Method
Switch to an alternative molecular featurization method if needed.

```bash
python scripts/modelling.py --model xgboost --feat eos24ci
```

#### Adding Dimensionality Reduction
Use PCA to reduce feature dimensions, which can help with computational efficiency and prevent overfitting.
You can specify either a specific number of components or a variance ratio to retain.

```bash
# Keep exactly 100 principal components
python scripts/modelling.py --model xgboost --feat eos39co --pca 100

# Keep enough components to retain 95% of variance
python scripts/modelling.py --model xgboost --feat eos39co --pca 0.95
```

#### Handling Class Imbalance
Add SMOTE oversampling to address the imbalance between BBB-permeable and non-permeable classes.

```bash
python scripts/modelling.py --model xgboost --feat eos39co --smote
```

#### Feature Importance Analysis
Enable SHAP analysis to understand which molecular features most influence the predictions.

```bash
python scripts/modelling.py --model xgboost --feat eos39co --shap
```

#### Complete Configuration Example
Combine all options for advanced model training with dimensionality reduction, class balancing, and feature analysis.

```bash
python scripts/modelling.py --model lightgbm --feat eos39co --pca 0.95 --smote --shap
```

### 📁 Model Artifacts

Each run saves trained models and preprocessing objects for later use:

```
models/{config_id}/             # Model artifacts
├── model.pkl                   # Serialized trained model
├── scaler.pkl                  # Fitted feature scaler
```

### 📊 Evaluation Results

Performance metrics and configuration details are saved in dedicated directories:

```
data/results/{config_id}/       # Performance metrics
├── metrics.json                # Detailed performance metrics
├── config.json                 # Configuration details
├── config.txt                  # Human-readable configuration
```

### 📈 Visualization Outputs

Visual representations of model performance are generated automatically:

```
data/plots/{config_id}/         # Visualizations
├── confusion_matrix_train.png  # Training confusion matrix
├── confusion_matrix_valid.png  # Validation confusion matrix
├── confusion_matrix_test.png   # Test confusion matrix
├── shap_summary_plot.png       # Feature importance (if --shap enabled)
```

### 📊 Model Evaluation

The evaluation includes detailed metrics for each data split:
- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Harmonic mean of precision and recall

### 🔍 Comparing Model Configurations

To easily compare different model setups, results are organized in separate directories with unique configuration IDs. This helps you determine which combination of model type, features, and processing techniques works best for BBB permeability prediction.

### 📋 Logging and Monitoring
- Model training logs: `models/bbb_training.log`
- All processes include detailed logging for monitoring progress and errors
