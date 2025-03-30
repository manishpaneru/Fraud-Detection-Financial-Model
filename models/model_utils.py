import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import pickle
import matplotlib.pyplot as plt

# Import XGBoost if available, otherwise use a placeholder
try:
    import xgboost as xgb
    XGBClassifier = xgb.XGBClassifier
except ImportError:
    # Define a placeholder that will raise an error when used
    class XGBClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("XGBoost is not installed. Install it with 'pip install xgboost'.")

def train_test_data_split(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(X_train, y_train, model_type='xgboost'):
    """Train the specified model type."""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    elif model_type == 'xgboost':
        try:
            model = XGBClassifier(
                scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        except Exception as e:
            print(f"Error initializing XGBoost: {e}")
            print("Falling back to Random Forest...")
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def train_isolation_forest(X_train, contamination=0.01):
    """Train Isolation Forest for anomaly detection."""
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return various metrics."""
    y_pred = model.predict(X_test)
    
    # Handle XGBoost predictions which might be probabilities
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None
    
    # Ensure y_pred is binary
    if not np.array_equal(np.unique(y_pred), [0, 1]) and not np.array_equal(np.unique(y_pred), [0]) and not np.array_equal(np.unique(y_pred), [1]):
        y_pred = (y_pred > 0.5).astype(int)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        results['y_pred_proba'] = y_pred_proba
    
    return results

def calculate_risk_score(transaction, model, scaler=None, amount_weight=0.3, 
                         account_risk_weight=0.2, network_risk_weight=0.1):
    """Calculate comprehensive risk score combining model prediction with other risk factors."""
    # Extract features in the same format as training data
    features = pd.DataFrame([transaction])
    
    # Scale features if scaler is provided
    if scaler:
        num_cols = features.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) > 0:  # Only scale if there are numerical columns
            try:
                features[num_cols] = scaler.transform(features[num_cols])
            except Exception:
                # If scaling fails, just use the original features
                pass
    
    # Get model probability
    try:
        if hasattr(model, 'predict_proba'):
            model_prob = model.predict_proba(features)[0][1]
        else:
            # For models without predict_proba, use a default value
            model_prob = 0.5
    except Exception:
        # If prediction fails, use a default value
        model_prob = 0.5
    
    # Calculate amount risk (simplified example)
    amount_risk = min(abs(transaction.get('amount_zscore', 0)) / 5, 1) if 'amount_zscore' in transaction else 0
    
    # Calculate other risk factors (placeholder values)
    account_risk = 0.2  # Default placeholder
    network_risk = 0.1  # Default placeholder
    
    # Weighted risk score (0-100)
    model_weight = 1 - (amount_weight + account_risk_weight + network_risk_weight)
    risk_score = (model_weight * model_prob + 
                  amount_weight * amount_risk + 
                  account_risk_weight * account_risk + 
                  network_risk_weight * network_risk) * 100
    
    return risk_score

def save_model(model, filepath):
    """Save model to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Load model from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance for models that support it."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # If there are fewer features than top_n, use all features
    top_n = min(top_n, len(feature_names))
    
    # Get indices of top features
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    return plt 