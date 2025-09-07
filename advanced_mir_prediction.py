import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

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

# Define a function to create PCA features
def create_pca_features(X_train, X_test, n_components=50):
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Calculate explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")
    
    return X_train_pca, X_test_pca, pca, scaler

# Create PCA features
X_train_pca, X_test_pca, pca, scaler = create_pca_features(X_train, X_test, n_components=50)

# Define a function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"{model_name} Results:")
    print(f"  Training MSE: {train_mse:.6f}")
    print(f"  Testing MSE: {test_mse:.6f}")
    print(f"  Training MAE: {train_mae:.6f}")
    print(f"  Testing MAE: {test_mae:.6f}")
    print(f"  Training R²: {train_r2:.6f}")
    print(f"  Testing R²: {test_r2:.6f}")
    
    return model, test_mse, test_r2, test_mae

# Create advanced models using PCA features
print("\n=== Training Advanced Models ===")

# 1. Ridge Regression with PCA
ridge_model = Ridge(alpha=1.0)
ridge_results = evaluate_model(ridge_model, X_train_pca, X_test_pca, y_train, y_test, "Ridge with PCA")

# 2. Gradient Boosting with PCA
gb_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
gb_results = evaluate_model(gb_model, X_train_pca, X_test_pca, y_train, y_test, "Gradient Boosting with PCA")

# 3. Random Forest with PCA
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
rf_results = evaluate_model(rf_model, X_train_pca, X_test_pca, y_train, y_test, "Random Forest with PCA")

# 4. SVR with PCA
svr_model = MultiOutputRegressor(SVR(kernel='rbf', C=10))
svr_results = evaluate_model(svr_model, X_train_pca, X_test_pca, y_train, y_test, "SVR with PCA")

# 5. Create a voting ensemble
print("\nTraining Voting Ensemble...")
voting_regressor = VotingRegressor([
    ('ridge', ridge_model),
    ('gb', gb_model),
    ('rf', rf_model)
])
voting_results = evaluate_model(voting_regressor, X_train_pca, X_test_pca, y_train, y_test, "Voting Ensemble")

# Collect all results
all_results = [
    ("Ridge with PCA", ridge_model, *ridge_results[1:]),
    ("Gradient Boosting with PCA", gb_model, *gb_results[1:]),
    ("Random Forest with PCA", rf_model, *rf_results[1:]),
    ("SVR with PCA", svr_model, *svr_results[1:]),
    ("Voting Ensemble", voting_regressor, *voting_results[1:])
]

# Sort results by test MSE (lower is better)
all_results.sort(key=lambda x: x[2])

print("\n=== Model Performance Summary (sorted by MSE) ===")
for name, _, mse, r2, mae in all_results:
    print(f"{name}: MSE={mse:.6f}, R²={r2:.6f}, MAE={mae:.6f}")

# Select the best model
best_name, best_model, best_mse, best_r2, best_mae = all_results[0]
print(f"\nBest model: {best_name} with MSE={best_mse:.6f}, R²={best_r2:.6f}, and MAE={best_mae:.6f}")

# Visualize predictions vs. actual for the best model
def plot_predictions(model, X, y, model_name, num_samples=5, num_wavelengths=100):
    # Make predictions
    y_pred = model.predict(X)
    
    # Select random samples to visualize
    sample_indices = np.random.choice(len(X), num_samples, replace=False)
    
    # Get MIR wavelengths (just for visualization)
    mir_wavelengths = [float(col.split('_')[0].split('.')[1]) for col in mir_target_cols[:num_wavelengths]]
    
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
plot_predictions(best_model, X_test_pca, y_test, best_name)

# Create a complete pipeline for the best model
best_pipeline = Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('model', best_model)
])

# Save the best model pipeline
import joblib
joblib.dump(best_pipeline, f'{best_name.replace(" ", "_").lower()}_pipeline.pkl')
print(f"Best model pipeline saved as {best_name.replace(' ', '_').lower()}_pipeline.pkl")

# Function to analyze wavelength importance
def analyze_wavelength_importance(pca_model, feature_names):
    # Get the PCA components
    components = pca_model.components_
    
    # Calculate the absolute importance of each wavelength across all components
    importance = np.sum(np.abs(components), axis=0)
    
    # Normalize importance
    importance = importance / np.sum(importance)
    
    # Create a DataFrame with wavelengths and their importance
    wavelengths = [float(name.split('.')[1].split('_')[0]) for name in feature_names]
    importance_df = pd.DataFrame({
        'Wavelength': wavelengths,
        'Importance': importance
    })
    
    # Plot importance
    plt.figure(figsize=(12, 6))
    plt.plot(importance_df['Wavelength'], importance_df['Importance'])
    plt.xlabel('NIR Wavelength (nm)')
    plt.ylabel('Relative Importance')
    plt.title('NIR Wavelength Importance for MIR Prediction')
    plt.grid(True)
    plt.savefig('nir_wavelength_importance.png')
    plt.close()
    
    # Return top 10 most important wavelengths
    top_wavelengths = importance_df.sort_values('Importance', ascending=False).head(10)
    return top_wavelengths

# Analyze wavelength importance
top_wavelengths = analyze_wavelength_importance(pca, nir_feature_cols)
print("\nTop 10 most important NIR wavelengths:")
print(top_wavelengths)

print("\nDone!")
