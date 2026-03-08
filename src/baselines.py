import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from data_preprocessing import get_processed_data


def compute_baseline_metrics(X_train, X_test, y_train, y_test):
    """
    Train baseline linear, Ridge, and LASSO models and compute evaluation metrics.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Preprocessed design matrices for training and testing.
    y_train, y_test : np.ndarray
        Corresponding target vectors.

    Returns
    -------
    results : dict
        Mapping from model name to a dictionary with keys:
        - 'MSE'
        - 'Non-zero Features'
    total_features : int
        Total number of features after preprocessing.
    """
    total_features = X_train.shape[1]
    print(f"\nTotal features after preprocessing (including One-Hot Encoding): {total_features}\n")
    print("-" * 50)

    results = {}

    # 1. Linear Regression (Baseline 1)
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_mse = mean_squared_error(y_test, lr_preds)
    lr_nonzero = np.sum(np.abs(lr_model.coef_) > 1e-5)  # Account for floating point precision

    results['Linear Regression'] = {'MSE': lr_mse, 'Non-zero Features': lr_nonzero}

    # 2. Ridge Regression - L2 Penalty (Baseline 2)
    print("Training Ridge Regression (alpha=1.0)...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)

    ridge_mse = mean_squared_error(y_test, ridge_preds)
    ridge_nonzero = np.sum(np.abs(ridge_model.coef_) > 1e-5)

    results['Ridge Regression'] = {'MSE': ridge_mse, 'Non-zero Features': ridge_nonzero}

    # 3. Standard LASSO - L1 Penalty (Baseline 3)
    print("Training Standard LASSO (alpha=0.01)...")
    lasso_model = Lasso(alpha=0.01, max_iter=10000) 
    lasso_model.fit(X_train, y_train)
    lasso_preds = lasso_model.predict(X_test)

    lasso_mse = mean_squared_error(y_test, lasso_preds)
    lasso_nonzero = np.sum(np.abs(lasso_model.coef_) > 1e-5)

    results['Standard LASSO'] = {'MSE': lasso_mse, 'Non-zero Features': lasso_nonzero}

    return results, total_features


def run_baselines():
    # 1. Load the preprocessed data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, preprocessor = get_processed_data()
    
    total_features = X_train.shape[1]
    print(f"\nTotal features after preprocessing (including One-Hot Encoding): {total_features}\n")
    print("-" * 50)

    # Dictionary to store results for Member 4
    results = {}

    # 2. Linear Regression (Baseline 1)
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    lr_mse = mean_squared_error(y_test, lr_preds)
    lr_nonzero = np.sum(np.abs(lr_model.coef_) > 1e-5) # Account for floating point precision
    
    results['Linear Regression'] = {'MSE': lr_mse, 'Non-zero Features': lr_nonzero}

    # 3. Ridge Regression - L2 Penalty (Baseline 2)
    print("Training Ridge Regression (alpha=1.0)...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)
    
    ridge_mse = mean_squared_error(y_test, ridge_preds)
    ridge_nonzero = np.sum(np.abs(ridge_model.coef_) > 1e-5)
    
    results['Ridge Regression'] = {'MSE': ridge_mse, 'Non-zero Features': ridge_nonzero}

    # 4. Standard LASSO - L1 Penalty (Baseline 3)
    # This is the most important baseline to compare against your team's custom optimizer
    print("Training Standard LASSO (alpha=0.01)...")
    lasso_model = Lasso(alpha=0.01, max_iter=10000) 
    lasso_model.fit(X_train, y_train)
    lasso_preds = lasso_model.predict(X_test)
    
    lasso_mse = mean_squared_error(y_test, lasso_preds)
    lasso_nonzero = np.sum(np.abs(lasso_model.coef_) > 1e-5)
    
    results['Standard LASSO'] = {'MSE': lasso_mse, 'Non-zero Features': lasso_nonzero}

    # 5. Print Final Results Table for Member 4
    print("\n" + "=" * 50)
    print("BASELINE MODEL RESULTS (Hand off to Member 4)")
    print("=" * 50)
    print(f"{'Model':<20} | {'Test MSE':<12} | {'Non-zero Features'}")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | {metrics['MSE']:<12.4f} | {metrics['Non-zero Features']} / {total_features}")
    print("=" * 50)

if __name__ == "__main__":
    run_baselines()