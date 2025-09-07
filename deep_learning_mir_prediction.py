import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# Standardize the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Define neural network architectures

# 1. Simple Feedforward Neural Network
def create_simple_nn(input_dim, output_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(output_dim)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 2. Deep Neural Network with Residual Connections
def create_deep_residual_nn(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    
    # First block
    x = Dense(512, kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Residual block 1
    residual = x
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, residual])
    x = Activation('relu')(x)
    
    # Reduction block
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Residual block 2
    residual = x
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, residual])
    x = Activation('relu')(x)
    
    # Output layer
    outputs = Dense(output_dim)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 3. Bottleneck Autoencoder-like Network
def create_bottleneck_nn(input_dim, output_dim):
    model = Sequential([
        # Encoder
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        
        # Bottleneck
        Dense(64, activation='relu'),
        BatchNormalization(),
        
        # Decoder (to MIR space)
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(output_dim)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Define callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Create and train the models
input_dim = X_train_scaled.shape[1]
output_dim = y_train_scaled.shape[1]

print("\n=== Training Simple Neural Network ===")
simple_nn = create_simple_nn(input_dim, output_dim)
simple_nn_history = simple_nn.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n=== Training Deep Residual Neural Network ===")
deep_nn = create_deep_residual_nn(input_dim, output_dim)
deep_nn_history = deep_nn.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n=== Training Bottleneck Neural Network ===")
bottleneck_nn = create_bottleneck_nn(input_dim, output_dim)
bottleneck_nn_history = bottleneck_nn.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the models
def evaluate_nn_model(model, X_train, X_test, y_train, y_test, model_name):
    # Make predictions
    y_pred_train_scaled = model.predict(X_train)
    y_pred_test_scaled = model.predict(X_test)
    
    # Inverse transform predictions back to original scale
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{model_name} Results:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Testing MSE: {test_mse:.6f}")
    print(f"Training MAE: {train_mae:.6f}")
    print(f"Testing MAE: {test_mae:.6f}")
    print(f"Training R²: {train_r2:.6f}")
    print(f"Testing R²: {test_r2:.6f}")
    
    return test_mse, test_r2, test_mae, y_pred_test

# Evaluate all models
simple_results = evaluate_nn_model(simple_nn, X_train_scaled, X_test_scaled, y_train, y_test, "Simple Neural Network")
deep_results = evaluate_nn_model(deep_nn, X_train_scaled, X_test_scaled, y_train, y_test, "Deep Residual Neural Network")
bottleneck_results = evaluate_nn_model(bottleneck_nn, X_train_scaled, X_test_scaled, y_train, y_test, "Bottleneck Neural Network")

# Collect all results
nn_results = [
    ("Simple Neural Network", simple_nn, *simple_results),
    ("Deep Residual Neural Network", deep_nn, *deep_results),
    ("Bottleneck Neural Network", bottleneck_nn, *bottleneck_results)
]

# Sort results by test MSE (lower is better)
nn_results.sort(key=lambda x: x[2])

print("\n=== Neural Network Performance Summary (sorted by MSE) ===")
for name, _, mse, r2, mae, _ in nn_results:
    print(f"{name}: MSE={mse:.6f}, R²={r2:.6f}, MAE={mae:.6f}")

# Select the best model
best_name, best_model, best_mse, best_r2, best_mae, best_preds = nn_results[0]
print(f"\nBest neural network: {best_name} with MSE={best_mse:.6f}, R²={best_r2:.6f}, and MAE={best_mae:.6f}")

# Plot training history for the best model
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['lr'])
    plt.title(f'{model_name} - Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('Epoch')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_history.png')
    plt.close()

# Plot history for all models
plot_training_history(simple_nn_history, "Simple Neural Network")
plot_training_history(deep_nn_history, "Deep Residual Neural Network")
plot_training_history(bottleneck_nn_history, "Bottleneck Neural Network")

# Visualize predictions vs. actual for the best model
def plot_nn_predictions(y_true, y_pred, model_name, num_samples=5, num_wavelengths=100):
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
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_predictions.png')
    plt.close()

# Plot predictions for the best model
plot_nn_predictions(y_test, best_preds, best_name)

# Save the best model
best_model.save(f'{best_name.replace(" ", "_").lower()}_model.h5')
print(f"Best neural network model saved as {best_name.replace(' ', '_').lower()}_model.h5")

# Also save the scalers for future use
import joblib
joblib.dump(scaler_X, 'nir_scaler.pkl')
joblib.dump(scaler_y, 'mir_scaler.pkl')
print("Scalers saved as nir_scaler.pkl and mir_scaler.pkl")

print("\nDone!")
