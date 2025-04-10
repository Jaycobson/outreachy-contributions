import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
# from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import pubchempy as pcp
import drugtax
import base64
from io import BytesIO
import os

# --- Helper functions ---

def get_smiles_from_name(compound_name):
    """Convert compound name to SMILES using PubChem API"""
    try:
        compounds = pcp.get_compounds(compound_name, 'name')
        if compounds:
            return compounds[0].isomeric_smiles
        else:
            return None
    except Exception as e:
        st.error(f"Error finding compound in PubChem: {e}")
        return None

def kekulize_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def extract_features(smile):
    canonical_smile = kekulize_smiles(smile)
    drug = drugtax.DrugTax(canonical_smile)
    features = dict(zip(drug.features.keys(), drug.features.values()))
    return pd.DataFrame([features])


def calculate_lipinski(mol):
    """Calculate Lipinski's Rule of Five parameters"""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if h_donors > 5: violations += 1
    if h_acceptors > 10: violations += 1
    
    return {
        "Molecular Weight": f"{mw:.2f} Da",
        "LogP": f"{logp:.2f}",
        "H-Bond Donors": h_donors,
        "H-Bond Acceptors": h_acceptors,
        "Rule Violations": violations
    }

# --- Set model path ---
MODEL_PATH = os.environ.get("MODEL_PATH", "models/xgboost/xgboost_eos24ci/model.pkl")

# --- Streamlit App UI ---

st.set_page_config(page_title="BBB Permeability Predictor", layout="wide")

st.title("🧠 Blood-Brain Barrier (BBB) Permeability Classifier")
st.markdown("Predict if a compound can cross the blood-brain barrier by entering a SMILES string or compound name.")

# Create tabs for input methods
tab1, tab2 = st.tabs(["SMILES Input", "Compound Name Input"])

with tab1:
    smiles_input = st.text_input("Enter SMILES string:", key="smiles_tab", value = 'CC(=O)OC1=CC=CC=C1C(=O)O')
    st.markdown("Note: The SMILES string will be converted to a canonical form.")
    st.markdown("This may take a few seconds.")
    process_button1 = st.button("Analyze Compound", key="analyze1")

with tab2:
    name_input = st.text_input("Enter compound name:", key="name_tab", value = 'paracetamol')
    st.markdown("Note: The compound name will be converted to SMILES using PubChem.")
    st.markdown("This may take a few seconds.")
    process_button2 = st.button("Analyze Compound", key="analyze2")

# Process inputs
smiles_to_process = None

if process_button1 and smiles_input:
    smiles_to_process = smiles_input
elif process_button2 and name_input:
    with st.spinner("Looking up compound..."):
        smiles_to_process = get_smiles_from_name(name_input)
        if smiles_to_process:
            st.success(f"Found SMILES: {smiles_to_process}")
        else:
            st.error(f"Could not find '{name_input}' in PubChem database.")

# Process the SMILES if available
if smiles_to_process:
    try:
        # Create molecule object for visualization and calculations
        mol = Chem.MolFromSmiles(smiles_to_process)
        if mol is None:
            st.error("Invalid SMILES string. Please check and try again.")
        else:
            # Create columns for visualization and prediction
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # st.subheader("📊 Compound Visualization")
                # img_str = mol_to_img(mol)
                # st.image(f"data:image/png;base64,{img_str}", caption="Molecular Structure")
                
                # Display basic properties
                st.subheader("⚗️ Molecular Properties")
                lipinski = calculate_lipinski(mol)
                properties_df = pd.DataFrame(list(lipinski.items()), columns=["Property", "Value"])
                st.dataframe(properties_df, hide_index=True)
            
            with col2:
                # Extract features
                with st.spinner("Extracting features..."):
                    features_df = extract_features(smiles_to_process)
                
                # Load model
                with st.spinner("Loading classification model..."):
                    try:
                        model = joblib.load(MODEL_PATH)
                        st.success("Model loaded successfully")
                    except FileNotFoundError:
                        st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
                        st.stop()
                
                # Predict
                prediction = model.predict(features_df)[0]
                proba = model.predict_proba(features_df)[0]
                
                # Display result
                st.subheader("🧪 BBB Permeability Prediction")
                
                # Fix for float32 error: convert numpy float32 to Python float
                prediction_idx = int(prediction)
                probability_value = float(proba[prediction_idx] * 100)
                
                # Use a gauge chart for probability visualization
                if prediction == 1:
                    st.markdown(f"### ✅ **BBB-Permeable**")
                    st.progress(probability_value/100)
                else:
                    st.markdown(f"### ❌ **Not BBB-Permeable**")
                    st.progress(probability_value/100)
                
                st.metric("Confidence", f"{probability_value:.2f}%")
                
                # Add interpretation
                st.subheader("🔍 Interpretation")
                if prediction == 1:
                    st.info("This compound is predicted to cross the blood-brain barrier, making it potentially useful for CNS therapeutic applications.")
                else:
                    st.info("This compound is predicted to have low permeability across the blood-brain barrier.")
                
                # Feature importance
                with st.expander("Key Contributing Features"):
                    try:
                        # Only attempt if model supports feature_importances_
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            feature_names = features_df.columns
                            
                            # Convert numpy values to Python native types
                            importances = [float(x) for x in importances]
                            
                            feature_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            st.bar_chart(feature_importance.set_index('Feature'))
                        else:
                            st.write("Feature importance not available for this model type.")
                    except Exception as e:
                        st.write(f"Could not extract feature importance: {str(e)}")
            
            # # Show full feature vector in expandable section
            # with st.expander("🧬 All Extracted Features"):
            #     # Convert any numpy types to Python native types for display
            #     display_df = features_df.copy()
            #     for col in display_df.columns:
            #         if isinstance(display_df[col].iloc[0], (np.float32, np.float64, np.int32, np.int64)):
            #             display_df[col] = display_df[col].astype(float)
                
            #     st.dataframe(display_df.T.rename(columns={0: "Value"}))
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
