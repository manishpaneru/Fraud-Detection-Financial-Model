import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath, sample_size=None, random_state=42):
    """
    Load and perform initial preprocessing on the dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    sample_size : int, optional
        Number of rows to sample. If None, use the entire dataset
    random_state : int, optional
        Random seed for reproducibility when sampling
        
    Returns:
    --------
    df : pandas.DataFrame
        Preprocessed dataframe
    """
    try:
        # Check if we need to sample the data
        if sample_size is not None:
            # Simply read the first sample_size rows - much more efficient
            df = pd.read_csv(filepath, nrows=sample_size)
            print(f"Loaded first {sample_size} rows from {filepath}")
        else:
            # Read the entire dataset
            df = pd.read_csv(filepath)
            print(f"Loaded entire dataset from {filepath} with {len(df)} rows")
        
        # Handle missing values - use more efficient in-place operations
        df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill with mean for numeric
        df.fillna('', inplace=True)  # Fill non-numeric with empty strings
        
        # Extract basic features
        df = add_basic_features(df)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return a small empty dataframe with expected columns to avoid errors
        return pd.DataFrame(columns=['amount', 'is_fraud'])

def add_basic_features(df):
    """Add basic engineered features to the dataframe."""
    # Work with original dataframe to avoid copy
    
    # Transaction-level features
    if 'Amount' in df.columns:
        # Standardize column names
        df.rename(columns={'Amount': 'amount'}, inplace=True)
        
        # Calculate percentiles for transaction amounts
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Z-score for transaction amount - more memory efficient calculation
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        df['amount_zscore'] = (df['amount'] - amount_mean) / amount_std
    elif 'amount' in df.columns:
        # Calculate percentiles for transaction amounts
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Z-score for transaction amount
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        df['amount_zscore'] = (df['amount'] - amount_mean) / amount_std
    
    # Time-based features (if timestamp is available)
    if 'Time' in df.columns:
        df.rename(columns={'Time': 'time'}, inplace=True)
        # Convert time to hours if it's in seconds
        time_max = df['time'].max()
        if time_max > 24:
            df['hour'] = (df['time'] / 3600) % 24
            # Create day feature if possible
            if time_max > 86400:  # More than one day in seconds
                df['day'] = (df['time'] // 86400)
        else:
            df['hour'] = df['time']
    elif 'step' in df.columns:
        # If we have a 'step' column, we can use it as a time proxy
        df.rename(columns={'step': 'time'}, inplace=True)
    
    # Handle labeled data - check for different possible column names for fraud indicator
    if 'Class' in df.columns:
        df.rename(columns={'Class': 'is_fraud'}, inplace=True)
    elif 'isFraud' in df.columns:
        df.rename(columns={'isFraud': 'is_fraud'}, inplace=True)
    elif 'fraud' in df.columns:
        df.rename(columns={'fraud': 'is_fraud'}, inplace=True)
    
    # Add Benford's Law feature for first digit (only if amount column exists)
    if 'amount' in df.columns:
        # Extract first digit - memory efficient approach
        df['first_digit'] = df['amount'].astype(str).str[0].astype(int)
        
        # Calculate expected frequency based on Benford's Law
        digits = np.arange(1, 10)
        benford_expected = np.log10(1 + 1/digits)
        
        # Create a mapping of digit to expected frequency
        benford_map = dict(zip(digits, benford_expected))
        
        # Map first digits to their expected frequencies
        df['benford_expected'] = df['first_digit'].map(benford_map)
        
        # Calculate deviation from Benford's Law
        digit_counts = df['first_digit'].value_counts(normalize=True)
        actual_freq_map = dict(zip(digit_counts.index, digit_counts.values))
        df['benford_deviation'] = df['first_digit'].map(
            lambda d: actual_freq_map.get(d, 0) - benford_map.get(d, 0)
        )
    
    # Create PCA features for the V columns if they exist - optional for memory concerns
    v_columns = [col for col in df.columns if col.startswith('V')]
    if len(v_columns) > 5:  # Only do PCA if we have enough V columns
        try:
            from sklearn.decomposition import PCA
            
            # Apply PCA to reduce dimensionality
            n_components = min(5, len(v_columns))
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(df[v_columns])
            
            # Add PCA components as new features
            for i in range(pca_result.shape[1]):
                df[f'pca_{i+1}'] = pca_result[:, i]
        except MemoryError:
            print("Memory error during PCA. Skipping PCA feature creation.")
        except Exception as e:
            print(f"Error during PCA: {e}. Skipping PCA feature creation.")
            
    return df

def scale_features(df, feature_cols, copy=False):
    """
    Scale numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe to scale
    feature_cols : list
        List of column names to scale
    copy : bool, optional
        Whether to copy the dataframe (more memory intensive) or modify in place
        
    Returns:
    --------
    df_scaled : pandas.DataFrame
        Scaled dataframe
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object
    """
    if len(feature_cols) == 0:
        # No features to scale
        return df, None
        
    try:
        # Initialize scaler
        scaler = StandardScaler()
        
        # Scale features
        if copy:
            df_scaled = df.copy()
            df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
            return df_scaled, scaler
        else:
            # More memory efficient approach - modify in place
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            return df, scaler
    except MemoryError:
        print("Memory error during scaling. Trying batch scaling...")
        # Batch scaling for very large datasets
        scaler = StandardScaler()
        # Fit scaler
        scaler.fit(df[feature_cols])
        
        # Transform data in chunks
        chunk_size = 10000
        for i in range(0, len(df), chunk_size):
            end_idx = min(i + chunk_size, len(df))
            df.iloc[i:end_idx, df.columns.get_indexer(feature_cols)] = scaler.transform(
                df.iloc[i:end_idx][feature_cols]
            )
        
        return df, scaler

def prepare_features_for_modeling(df, sample_size=None):
    """
    Final preparation of features for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe to prepare
    sample_size : int, optional
        Number of rows to sample before modeling. If None, use the entire dataset
        
    Returns:
    --------
    X_scaled : pandas.DataFrame
        Scaled feature matrix
    y : pandas.Series
        Target variable
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object
    """
    try:
        # Sample data if requested
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        
        # Drop columns not needed for modeling
        drops = ['time', 'first_digit', 'benford_expected'] 
        drops = [col for col in drops if col in df.columns]
        
        # Also drop string-based ID columns that shouldn't be used for modeling
        # Check for columns that might be IDs or non-useful strings
        potential_id_cols = ['nameOrig', 'nameDest', 'account_id']
        drops.extend([col for col in potential_id_cols if col in df.columns])
        
        # Identify target variable
        has_target = 'is_fraud' in df.columns
        if has_target:
            y = df['is_fraud']
            X = df.drop(drops + ['is_fraud'], axis=1)
        else:
            y = None
            X = df.drop(drops, axis=1, errors='ignore')
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print(f"Found categorical columns: {categorical_cols}")
            
            # Option 1: One-hot encode categorical columns
            # This is memory-efficient for a small number of unique values
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            # Option 2: Drop categorical columns if one-hot encoding would create too many features
            # Uncomment this code instead if you have categorical columns with many unique values
            # X_encoded = X.drop(categorical_cols, axis=1)
            # print(f"Dropped categorical columns: {categorical_cols}")
            
            X = X_encoded
        
        # Get only numerical columns
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        # Scale features without creating a copy
        X_scaled, scaler = scale_features(X, num_cols, copy=False)
        
        return X_scaled, y, scaler
    except Exception as e:
        print(f"Error preparing features: {e}")
        # Return empty dataframes/series with proper structure to avoid errors
        return pd.DataFrame(), pd.Series(dtype=int), None 