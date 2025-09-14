import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

class PCAOffsetRegressor(BaseEstimator, RegressorMixin):
    """
    A custom regressor that combines PCA dimensionality reduction with a linear regression model
    and adds a +0.1 or -0.1 offset based on a classifier.
    
    The offset is applied selectively to improve prediction accuracy.
    """
    
    def __init__(self, n_components=50, offset_value=0.1):
        self.n_components = n_components
        self.offset_value = offset_value
        self.pca = PCA(n_components=n_components)
        self.regressor = Ridge(alpha=1.0)  # Using Ridge instead of LinearRegression for better regularization
        self.offset_classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
    def fit(self, X, y):
        # Apply PCA
        X_pca = self.pca.fit_transform(X)
        
        # Fit the main regression model
        self.regressor.fit(X_pca, y)
        
        # Make initial predictions
        initial_predictions = self.regressor.predict(X_pca)
        
        # Calculate errors
        errors = y - initial_predictions
        
        # Calculate mean absolute error for each sample
        mae_per_sample = np.mean(np.abs(errors), axis=1)
        
        # Determine if each sample should get a positive or negative offset
        # If the error is positive, we should add a positive offset
        # If the error is negative, we should add a negative offset
        offset_targets = np.mean(errors, axis=1) > 0
        
        # Only apply offset to samples with high error
        # (those above the median error)
        median_error = np.median(mae_per_sample)
        high_error_samples = mae_per_sample > median_error
        
        # Store which samples should have offsets applied
        self.apply_offset_mask = high_error_samples
        
        # Train a classifier to predict whether to use positive or negative offset
        # Only for samples where we'll apply an offset
        self.offset_classifier.fit(X_pca[high_error_samples], offset_targets[high_error_samples])
        
        return self
    
    def predict(self, X):
        # Apply PCA
        X_pca = self.pca.transform(X)
        
        # Get base predictions
        base_predictions = self.regressor.predict(X_pca)
        
        # Make a copy of the predictions that we'll modify
        predictions_with_offset = base_predictions.copy()
        
        # For test data, we need to determine which samples should get offsets
        # We'll use the classifier's confidence to decide
        offset_probs = self.offset_classifier.predict_proba(X_pca)
        
        # Only apply offset if the classifier is confident
        # (probability > 0.7 for either class)
        confident_samples = np.max(offset_probs, axis=1) > 0.7
        
        # Get the predicted class (0 = negative offset, 1 = positive offset)
        offset_directions = self.offset_classifier.predict(X_pca)
        
        # For confident samples, apply the offset
        for i in range(len(X)):
            if confident_samples[i]:
                # Apply positive or negative offset based on the classifier's prediction
                offset = self.offset_value if offset_directions[i] else -self.offset_value
                predictions_with_offset[i] += offset
        
        return predictions_with_offset
        
    def get_offset_info(self, X):
        """Return information about the applied offsets for analysis"""
        X_pca = self.pca.transform(X)
        offset_probs = self.offset_classifier.predict_proba(X_pca)
        confident_samples = np.max(offset_probs, axis=1) > 0.7
        offset_directions = self.offset_classifier.predict(X_pca)
        
        return {
            'confident_samples': confident_samples,
            'offset_directions': offset_directions,
            'offset_probs': offset_probs
        }
