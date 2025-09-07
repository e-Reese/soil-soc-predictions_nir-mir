import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import tensorflow as tf
import os
import glob

# Set random seed for reproducibility
np.random.seed(42)

# Load the NIR and MIR data
print("Loading data...")
nir_data = pd.read_csv('data/neospectra_nir_v1.2.csv')
mir_data = pd.read_csv('data/neospectra_mir_v1.2.csv')

# Extract sample IDs from both datasets
nir_ids = nir_data['id.sample_local_c'].values
mir_ids = mir_data['id.sample_local_c'].values

# Print the number of unique sample IDs in each dataset
print(f"Unique NIR sample IDs: {len(np.unique(nir_ids))}")
print(f"Unique MIR sample IDs: {len(np.unique(mir_ids))}")

# Find common sample IDs
common_ids = np.intersect1d(nir_ids, mir_ids)
print(f"Found {len(common_ids)} common samples between NIR and MIR datasets")

# Filter data to only include common samples
nir_filtered = nir_data[nir_data['id.sample_local_c'].isin(common_ids)]
mir_filtered = mir_data[mir_data['id.sample_local_c'].isin(common_ids)]

# Check for duplicates in each dataset
nir_duplicates = nir_filtered['id.sample_local_c'].duplicated().sum()
mir_duplicates = mir_filtered['id.sample_local_c'].duplicated().sum()
print(f"Duplicated NIR samples: {nir_duplicates}")
print(f"Duplicated MIR samples: {mir_duplicates}")

# Remove duplicates if any
if nir_duplicates > 0:
    nir_filtered = nir_filtered.drop_duplicates(subset=['id.sample_local_c'])
if mir_duplicates > 0:
    mir_filtered = mir_filtered.drop_duplicates(subset=['id.sample_local_c'])

# Sort both datasets by sample ID to ensure alignment
nir_data = nir_filtered.sort_values(by='id.sample_local_c').reset_index(drop=True)
mir_data = mir_filtered.sort_values(by='id.sample_local_c').reset_index(drop=True)

# Verify that the datasets have the same number of samples
print(f"NIR data shape after filtering: {nir_data.shape}")
print(f"MIR data shape after filtering: {mir_data.shape}")
assert len(nir_data) == len(mir_data), "NIR and MIR datasets have different numbers of samples"

# Extract features (NIR spectral data) and target (MIR spectral data)
# Find columns containing spectral data
nir_feature_cols = [col for col in nir_data.columns if col.startswith('scan_nir.')]
mir_target_cols = [col for col in mir_data.columns if col.startswith('scan_mir.')]

print(f"NIR features: {len(nir_feature_cols)}")
print(f"MIR targets: {len(mir_target_cols)}")

# Extract features and targets
X = nir_data[nir_feature_cols].values
y = mir_data[mir_target_cols].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# Find all saved models
sklearn_models = glob.glob("*_model.pkl") + glob.glob("*_pipeline.pkl")
keras_models = glob.glob("*_model.h5")

print(f"\nFound {len(sklearn_models)} scikit-learn models and {len(keras_models)} Keras models")

# Function to evaluate sklearn models
def evaluate_sklearn_model(model_path, X_test, y_test):
    print(f"Evaluating {model_path}...")
    model = joblib.load(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_name = os.path.basename(model_path).replace('_model.pkl', '').replace('_pipeline.pkl', '')
    model_name = model_name.replace('_', ' ').title()
    
    return model_name, mse, mae, r2, y_pred

# Function to evaluate keras models
def evaluate_keras_model(model_path, X_test, y_test):
    print(f"Evaluating {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load scalers if they exist
    if os.path.exists('nir_scaler.pkl') and os.path.exists('mir_scaler.pkl'):
        scaler_X = joblib.load('nir_scaler.pkl')
        scaler_y = joblib.load('mir_scaler.pkl')
        
        # Scale input data
        X_test_scaled = scaler_X.transform(X_test)
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    else:
        # If no scalers, just predict directly (not recommended)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_name = os.path.basename(model_path).replace('_model.h5', '')
    model_name = model_name.replace('_', ' ').title()
    
    return model_name, mse, mae, r2, y_pred

# Evaluate all models
results = []

# Evaluate sklearn models
for model_path in sklearn_models:
    try:
        result = evaluate_sklearn_model(model_path, X_test, y_test)
        results.append(result)
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")

# Evaluate keras models
for model_path in keras_models:
    try:
        result = evaluate_keras_model(model_path, X_test, y_test)
        results.append(result)
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")

# Sort results by MSE (lower is better)
results.sort(key=lambda x: x[1])

# Display results
print("\n=== Model Performance Comparison (sorted by MSE) ===")
print(f"{'Model Name':<30} {'MSE':<15} {'MAE':<15} {'R²':<15}")
print("-" * 75)
for name, mse, mae, r2, _ in results:
    print(f"{name:<30} {mse:<15.6f} {mae:<15.6f} {r2:<15.6f}")

# Select the best model
best_name, best_mse, best_mae, best_r2, best_preds = results[0]
print(f"\nBest model: {best_name} with MSE={best_mse:.6f}, R²={best_r2:.6f}, and MAE={best_mae:.6f}")

# Visualize predictions vs. actual for the best model
def plot_predictions(y_true, y_pred, model_name, num_samples=5, num_wavelengths=100):
    # Select random samples to visualize
    sample_indices = np.random.choice(len(y_true), num_samples, replace=False)
    
    # Get MIR wavelengths (just for visualization)
    mir_wavelengths = [float(col.split('_')[0].split('.')[1]) for col in mir_target_cols[:num_wavelengths]]
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(mir_wavelengths, y_true[idx][:num_wavelengths], 'b-', label='Actual MIR')
        plt.plot(mir_wavelengths, y_pred[idx][:num_wavelengths], 'r-', label='Predicted MIR')
        plt.title(f'Sample {i+1}')
        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.suptitle(f'{model_name} - Actual vs Predicted MIR Spectra', y=1.02)
    plt.savefig(f'best_model_comparison.png')
    plt.close()

# Plot predictions for the best model
plot_predictions(y_test, best_preds, best_name)

# Create a bar chart comparing all models
def plot_model_comparison(results):
    model_names = [result[0] for result in results]
    mse_values = [result[1] for result in results]
    r2_values = [result[3] for result in results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # MSE plot (lower is better)
    ax1.barh(model_names, mse_values, color='skyblue')
    ax1.set_title('Model Comparison - Mean Squared Error (lower is better)')
    ax1.set_xlabel('MSE')
    
    # R² plot (higher is better)
    ax2.barh(model_names, r2_values, color='lightgreen')
    ax2.set_title('Model Comparison - R² Score (higher is better)')
    ax2.set_xlabel('R²')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

# Plot model comparison
plot_model_comparison(results)

print("\nDone! Comparison charts saved as 'model_comparison.png' and 'best_model_comparison.png'.")
