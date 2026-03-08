import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_processed_data(save_to_disk=True):
    """Fetches, cleans, splits, preprocesses, and saves the House Prices dataset."""
    print("Fetching dataset...")
    house_prices = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    df = house_prices.frame
    
    # 1. CRITICAL: Remove severe outliers recommended by dataset author
    print("Removing outliers...")
    df = df[df['GrLivArea'] < 4000]
    
    # 2. Clean up mostly empty columns
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 3. Separate Features and Target (Applying log transformation)
    X = df.drop('SalePrice', axis=1)
    y = np.log1p(df['SalePrice']) 
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Identify columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    # 6. Build Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # 7. Transform Data
    print("Applying transformations...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 8. Save arrays for Member 3's custom optimizer
    if save_to_disk:
        print("Saving processed matrices to disk...")
        os.makedirs('../data/processed', exist_ok=True)
        np.save('../data/processed/X_train.npy', X_train_processed)
        np.save('../data/processed/X_test.npy', X_test_processed)
        np.save('../data/processed/y_train.npy', y_train.values)
        np.save('../data/processed/y_test.npy', y_test.values)
        print("Saved to data/processed/!")
        
        # Make sure Member 3 knows the shape
        print(f"Final training matrix shape: {X_train_processed.shape}")
    
    print("Data preprocessing complete!")
    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor

if __name__ == "__main__":
    # Running this directly will process and save the files
    get_processed_data()