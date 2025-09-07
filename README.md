# MIR Prediction from NIR Data

This project aims to predict Mid-Infrared (MIR) spectral data from Near-Infrared (NIR) spectral data using various machine learning models. The goal is to enable the prediction of MIR spectra (which typically require more expensive equipment) from NIR spectra (which can be collected with less expensive, more portable devices).

## Dataset

The dataset consists of paired NIR and MIR spectral measurements from soil samples:

- `neospectra_nir_v1.2.csv`: Contains NIR spectral data (1350-2550 nm)
- `neospectra_mir_v1.2.csv`: Contains MIR spectral data (600-4000 cm⁻¹)

## Models Implemented

The project implements several models with varying complexity:

### Basic Models (`mir_prediction_models.py`)
- Linear Regression
- Ridge Regression
- Lasso Regression
- PCA + Linear Regression
- Random Forest
- Gradient Boosting
- Neural Network (MLPRegressor)

### Advanced Models (`advanced_mir_prediction.py`)
- Ridge with PCA
- Gradient Boosting with PCA
- Random Forest with PCA
- SVR with PCA
- Voting Ensemble

### Deep Learning Models (`deep_learning_mir_prediction.py`)
- Simple Feedforward Neural Network
- Deep Residual Neural Network
- Bottleneck Autoencoder-like Network

## Model Comparison

The `model_comparison.py` script evaluates and compares all trained models on a common test set, providing metrics such as MSE, MAE, and R² score.

## Results Summary

Our experiments have shown:

1. **Ridge Regression** performed the best among the basic models with:
   - MSE: 0.029811
   - R²: 0.106460

2. **Gradient Boosting** was the second-best performer:
   - MSE: 0.030956
   - R²: 0.084588

3. Simpler linear models with regularization outperformed more complex models on this dataset.

4. The relationship between NIR and MIR spectra appears to be better captured by linear models with regularization rather than complex nonlinear models.

## Usage

1. Install required dependencies:
```
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
```

2. Run the basic models:
```
python mir_prediction_models.py
```

3. Run the advanced models:
```
python advanced_mir_prediction.py
```

4. Run the deep learning models:
```
python deep_learning_mir_prediction.py
```

5. Compare all models:
```
python model_comparison.py
```

## Project Structure

```
LowToHighRez/
├── data/
│   ├── neospectra_mir_v1.2.csv
│   └── neospectra_nir_v1.2.csv
├── mir_prediction_models.py    # Basic ML models
├── advanced_mir_prediction.py  # Advanced models with dimensionality reduction
├── deep_learning_mir_prediction.py  # Deep learning models
├── model_comparison.py         # Compare all models
├── README.md
└── .gitignore
```

## Output Files

Each script generates:
- Model performance metrics (MSE, MAE, R²)
- Visualizations of actual vs. predicted MIR spectra (PNG files)
- Saved model files for the best-performing models (PKL files)

## Model Selection Criteria

The best model can be selected based on:
- Mean Squared Error (MSE): Lower is better
- R² Score: Higher is better
- Mean Absolute Error (MAE): Lower is better
- Prediction speed and model complexity

## Future Work

Potential improvements:
- Hyperparameter tuning for each model
- Feature selection to identify most important NIR wavelengths
- Transfer learning approaches
- More complex neural network architectures
- Ensemble methods combining multiple model types
- Domain-specific preprocessing techniques for spectral data
- Exploring more advanced dimensionality reduction techniques
- Investigating the physical meaning of the spectral relationships

## License

This project is provided as-is with no warranties. Please respect the original data licenses.

## Acknowledgements

This project uses the Neospectra soil spectral dataset for research purposes.
