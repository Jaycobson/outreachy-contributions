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
| Research Papers | [1]. Martins, Ines Filipa, et al. *Journal of Chemical Information and Modeling*, 52.6 (2012): 1686-1697. <br> [2]. Wu, Zhenqin, et al. *Chemical Science*, 9.2 (2018): 513-530. |
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
├── data/
│   ├── BBB_Martins/       
│   │   ├── BBB_Martins.csv  # Complete dataset
│   │   ├── train.csv        # Training split
│   │   ├── test.csv         # Testing split
│   │   ├── valid.csv        # Validation split
│   ├── featurized_data/     # Processed molecular features
│   ├── results/             # Model evaluation metrics
│   └── plots/               # Performance visualizations
├── models/                  # Saved model files
│   ├── xgboost/             # XGBoost models
│   ├── lightgbm/            # LightGBM models (if implemented)
│   └── logistic/            # Logistic regression models (if implemented)
├── notebooks/
│   ├── analysis.ipynb       # Main notebook for visualization and analysis
│   ├── exploration.ipynb    # Initial data exploration
│   └── evaluation.ipynb     # Detailed model evaluation
├── scripts/
│   ├── download_data.py     # Dataset acquisition script
│   ├── featurize_data.py    # Molecular featurization script
│   ├── modelling.py         # Model training and evaluation
│   └── utils.py             # Helper functions
├── environment.yml          # Conda environment specification
├── .gitignore
├── LICENSE
└── README.md                # Project documentation
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.9 or later
- Miniconda/Anaconda
- Ubuntu OS or WSL (if system OS is Windows)
- Docker (for Ersilia models)
- Git

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/bbb-prediction.git
cd bbb-prediction
```

2. **Set up the conda environment**
```bash
# Create and activate the environment using the yml file
conda env create --file environment.yml
conda activate bbbp
```

The `environment.yml` file includes all necessary dependencies:
- rdkit
- scikit-learn
- xgboost
- lightgbm
- pandas
- numpy
- matplotlib
- seaborn
- jupyter
- ersilia (for molecular featurization)

3. **Dataset Acquisition**
```bash
# The dataset is already included in the data folder
# To manually download it, run:
python scripts/download_data.py
```

This script:
- Downloads the BBB_Martins dataset from TDC
- Creates train/validation/test splits if needed
- Saves data to `data/BBB_Martins/`

4. **Verify Installation**
```bash
# Check that everything is working
python -c "import rdkit; import ersilia; import xgboost; print('Setup successful!')"
```

## 🧪 Data Exploration and Visualization

To explore the dataset and understand the molecular properties:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook includes:
- Visualizing drug distributions
- Molecular structure representation using RDKit
- Checking for duplicates and data integrity
- Class distribution analysis
- Chemical property visualization (molecular weight, LogP, etc.)
- Structural similarity assessment

## 🚀 Molecular Featurization

This project uses the **RDKit Descriptor Model** from Ersilia Model Hub for comprehensive molecular featurization.

### Setup Ersilia Model
```bash
# Install Ersilia if not already installed
pip install ersilia

# Fetch and serve the model
ersilia fetch eos8a4x
ersilia serve eos8a4x
```

### Generate Features
```bash
# Process SMILES strings into numerical features
python scripts/featurize_data.py
```

This script:
- Converts SMILES representations to RDKit molecular objects
- Calculates molecular descriptors (over 200 features) including:
  - Physical properties (MW, LogP, TPSA)
  - Topological indices
  - Functional group counts
  - Connectivity information
  - Quantum mechanical properties (if available)
- Handles missing values and normalization
- Saves features to `data/featurized_data/`

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

### Training the Model
```bash
python scripts/train_model.py --model xgboost
```

Command line options:
- `--model [xgboost|lightgbm|logistic]`: Select model type
- `--hyperopt`: Enable hyperparameter optimization
- `--cross_val [N]`: Perform N-fold cross-validation
- `--output [filename]`: Specify output model filename

### Model Configuration
#### XGBoost Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 0.01 | Gradual, robust learning |
| Max Depth | 6 | Control tree complexity |
| Subsample | 0.8 | Prevent overfitting |
| Col Sample by Tree | 0.8 | Feature randomness |
| Early Stopping | 50 rounds | Optimal model selection |
| Objective | binary:logistic | Binary classification task |
| Scale Pos Weight | Calculated | Address class imbalance |

### Class Imbalance Handling
The dataset has a class imbalance (76.41% permeable vs. 23.59% non-permeable). The following strategies are implemented:

- **Scale Positive Weight**: Adjusts for class imbalance in tree-based models
- **SMOTE** (optional): Synthetic Minority Over-sampling Technique
- **Class Weighting**: Weighting samples inversely proportional to class frequencies
- **Balanced Evaluation Metrics**: Using precision, recall, F1 and AUC for proper assessment

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

### Interpretation
Special focus is given to model interpretability:
- **SHAP Values**: Explains individual predictions
- **Feature Importance Analysis**: Identifies key molecular characteristics
- **Decision Boundary Visualization**: Helps understand model behavior

## 🔗 Advanced Usage

### Custom Model Parameters
Modify model parameters directly in `train_model.py`:
```python
# Example: Adjusting XGBoost parameters
model = xgb.XGBClassifier(
    learning_rate=0.005,  # More conservative learning
    max_depth=8,          # Deeper trees
    subsample=0.7,        # Different subsampling
    colsample_bytree=0.7, # Different feature selection
    scale_pos_weight=class_weight, # Calculated class weight
    # Other parameters...
)
```

### Hyperparameter Optimization
For automated hyperparameter tuning:
```bash
python scripts/train_model.py --model xgboost --hyperopt
```

This uses Bayesian optimization to find optimal parameters based on validation performance.

### Cross-Validation
For more robust model assessment:
```bash
python scripts/train_model.py --model xgboost --cross_val 5
```

This performs 5-fold cross-validation and reports aggregated performance metrics.

### Ensemble Methods
To combine multiple models for improved performance:
```bash
python scripts/train_model.py --ensemble
```

This trains and combines XGBoost, LightGBM, and Logistic Regression models using a voting classifier.

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

## 📬 Contact
[Your Contact Information]

## 📜 License
[Your License Information]
