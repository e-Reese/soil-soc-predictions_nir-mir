import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from pca_offset_model import PCAOffsetRegressor

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

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)  # Fixed: use y_pred_train instead of y_pred_test
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{model_name} Results:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Testing MSE: {test_mse:.6f}")
    print(f"Training R²: {train_r2:.6f}")
    print(f"Testing R²: {test_r2:.6f}")
    
    return model, test_mse, test_r2

# Define models to evaluate
models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge(alpha=1.0)),
    # ("Lasso Regression", Lasso(alpha=0.001, max_iter=10000)),
    ("PCA + Linear Regression", Pipeline([
        ('pca', PCA(n_components=50)),
        ('lr', LinearRegression())
    ])),
    ("PCA + Linear Regression with Offset", PCAOffsetRegressor(n_components=50, offset_value=0.1)),
    # ("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42)),
    # ("Gradient Boosting", MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))),
    ("Neural Network", MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
]

# Evaluate each model
results = []
for name, model in models:
    print(f"\nTraining {name}...")
    model, mse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results.append((name, model, mse, r2))

# Sort results by test MSE (lower is better)
results.sort(key=lambda x: x[2])

print("\n=== Model Performance Summary (sorted by MSE) ===")
for name, _, mse, r2 in results:
    print(f"{name}: MSE={mse:.6f}, R²={r2:.6f}")

# Select the best model
best_name, best_model, best_mse, best_r2 = results[0]
print(f"\nBest model: {best_name} with MSE={best_mse:.6f} and R²={best_r2:.6f}")

# Visualize predictions vs. actual for the best model
def plot_predictions(model, X, y, model_name, num_samples=5, num_wavelengths=100):
    # Make predictions
    y_pred = model.predict(X)
    
    # Select random samples to visualize
    sample_indices = np.random.choice(len(X), num_samples, replace=False)
    
    # Get MIR wavelengths (just for visualization)
    # Extract wavelength numbers from column names
    mir_wavelengths = []
    for col in mir_target_cols[:num_wavelengths]:
        try:
            # Try to extract wavelength from format like 'scan_mir.600_abs'
            parts = col.split('.')
            if len(parts) >= 2:
                wavelength = parts[1].split('_')[0]
                mir_wavelengths.append(float(wavelength))
            else:
                # Fallback to using index if parsing fails
                mir_wavelengths.append(len(mir_wavelengths))
        except (IndexError, ValueError):
            # If parsing fails, use the index as a placeholder
            mir_wavelengths.append(len(mir_wavelengths))
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(mir_wavelengths, y[idx][:num_wavelengths], 'b-', label='Actual MIR')
        plt.plot(mir_wavelengths, y_pred[idx][:num_wavelengths], 'r-', label='Predicted MIR')
        plt.title(f'Sample {i+1}')
        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.suptitle(f'{model_name} - Actual vs Predicted MIR Spectra', y=1.02)
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_predictions.png')
    plt.close()

# Plot predictions for the best model
plot_predictions(best_model, X_test, y_test, best_name)

# If the best model is the offset model, visualize the offset effect
if best_name == "PCA + Linear Regression with Offset":
    # Get the offset model
    offset_model = best_model
    
    # Get base predictions without offset
    X_pca = offset_model.pca.transform(X_test)
    base_predictions = offset_model.regressor.predict(X_pca)
    
    # Get predictions with offset
    predictions_with_offset = offset_model.predict(X_test)
    
    # Get the offset directions
    offset_directions = offset_model.offset_classifier.predict(X_pca)
    
    # Visualize a few samples with and without offset
    num_samples = 3
    num_wavelengths = 100
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Get MIR wavelengths for visualization
    mir_wavelengths = []
    for col in mir_target_cols[:num_wavelengths]:
        try:
            parts = col.split('.')
            if len(parts) >= 2:
                wavelength = parts[1].split('_')[0]
                mir_wavelengths.append(float(wavelength))
            else:
                mir_wavelengths.append(len(mir_wavelengths))
        except (IndexError, ValueError):
            mir_wavelengths.append(len(mir_wavelengths))
    
    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(mir_wavelengths, y_test[idx][:num_wavelengths], 'b-', label='Actual MIR')
        plt.plot(mir_wavelengths, base_predictions[idx][:num_wavelengths], 'g-', label='Base Prediction')
        plt.plot(mir_wavelengths, predictions_with_offset[idx][:num_wavelengths], 'r-', label='Prediction with Offset')
        
        offset_direction = "positive" if offset_directions[idx] else "negative"
        plt.title(f'Sample {i+1} - Offset Direction: {offset_direction} ({offset_model.offset_value if offset_directions[idx] else -offset_model.offset_value})')
        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        plt.legend()
    
    plt.tight_layout()
    plt.suptitle('Effect of Offset on Predictions', y=1.02)
    plt.savefig('offset_effect_visualization.png')
    plt.close()
    
    # Print information about the offset
    positive_offsets = np.sum(offset_directions)
    negative_offsets = len(offset_directions) - positive_offsets
    print(f"\nOffset Analysis:")
    print(f"Samples with positive offset (+{offset_model.offset_value}): {positive_offsets} ({positive_offsets/len(offset_directions)*100:.1f}%)")
    print(f"Samples with negative offset (-{offset_model.offset_value}): {negative_offsets} ({negative_offsets/len(offset_directions)*100:.1f}%)")

# Save the best model
import joblib
joblib.dump(best_model, f'{best_name.replace(" ", "_").lower()}_model.pkl')
print(f"Best model saved as {best_name.replace(' ', '_').lower()}_model.pkl")

print("\nDone!")

