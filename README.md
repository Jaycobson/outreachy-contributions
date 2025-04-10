## Blood-Brain Barrier (BBB) Permeability Predictor

This Streamlit application predicts whether chemical compounds can cross the blood-brain barrier, a crucial property for CNS drug development. The app provides a user-friendly interface with the following key features:

### Key Features

- Dual Input Methods: Users can input either SMILES strings directly or search for compounds by name using the PubChem API
- Molecular Visualization: Interactive display of compound structures using RDKit
- Physicochemical Properties: Calculation and display of Lipinski's Rule of Five parameters (molecular weight, LogP, H-bond donors/acceptors)
- ML-Based Prediction: Employs a pre-trained XGBoost model to classify compounds as BBB-permeable or non-permeable
- Confidence Metrics: Visual progress bar showing prediction confidence levels
- Feature Importance: Analysis of which molecular features most strongly influence the prediction
- Detailed Data Exploration: Expandable sections to examine all extracted molecular features

### Technical Implementation

The application integrates several bioinformatics and machine learning libraries:

- RDKit for chemical structure handling and visualization
- PubchemPy for compound name resolution
- DrugTax for molecular feature extraction
- XGBoost (via joblib) for prediction
- Streamlit for the interactive web interface

The prediction workflow includes feature extraction from molecular structures, model inference, and comprehensive result visualization. The application handles various edge cases including invalid SMILES strings, compound name lookup failures, and model loading errors.
