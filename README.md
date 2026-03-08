
## 1. Project Title

**Adaptive LASSO using Proximal Gradient Descent with Dynamic Soft Thresholding**

This repository implements and evaluates a custom Adaptive LASSO-style regression model trained via Proximal Gradient Descent with a dynamic soft-thresholding operator. The system is built around the House Prices: Advanced Regression Techniques dataset and is designed to study feature selection in a high-dimensional regression setting.

---

## 2. Project Motivation

### 2.1 Overfitting in High-Dimensional Regression

In regression problems with many input features, it is easy for models to **overfit**: they learn noise and spurious correlations in the training data instead of the underlying signal. This typically results in:
- Low training error but
- Poor generalization performance on unseen test data.

When the number of features is large relative to the number of samples, overfitting becomes especially severe. Many features may be redundant, weakly informative, or purely noisy.

### 2.2 Why Feature Selection Matters

**Feature selection** aims to identify a subset of informative predictors and discard the rest. This has several advantages:
- **Improved generalization**: Reducing the effective dimensionality can decrease variance and improve test performance.
- **Interpretability**: Sparse models are easier to interpret because they highlight a small set of important features.
- **Efficiency**: Fewer active features mean cheaper inference and sometimes simpler modeling pipelines.

### 2.3 Why LASSO is Commonly Used

The **LASSO (Least Absolute Shrinkage and Selection Operator)** adds an \(\ell_1\) penalty to the regression objective. For a linear model \(y \approx Xw\), LASSO solves:
\[
\min_w \frac{1}{2n}\lVert y - Xw \rVert_2^2 + \lambda \lVert w \rVert_1,
\]
where \(\lambda \ge 0\) is the regularization strength.

The \(\ell_1\) penalty encourages **sparsity** in the coefficient vector \(w\): many components become exactly zero. This makes LASSO a standard tool for feature selection in high-dimensional regression.

### 2.4 Limitation of Static L1 Regularization

Standard LASSO uses a **single, static** regularization strength \(\lambda\) throughout optimization. This creates a tension:
- A **large** \(\lambda\) aggressively shrinks coefficients, producing high sparsity but potentially **underfitting** (losing important signal).
- A **small** \(\lambda\) maintains good predictive performance but may leave **too many features active**, reducing sparsity and interpretability.

In other words, static L1 regularization must trade off sparsity and accuracy with one global knob. It cannot, for example, prune aggressively early and then refine the active set with a gentler penalty later.

### 2.5 Goal of This Project

The goal of this project is to **improve feature selection** by introducing a **dynamic soft-thresholding mechanism**. Instead of using a fixed \(\lambda\), we allow the effective penalty \(\lambda_t\) to **change with iteration**. This enables:
- **Early strong regularization** to prune uninformative features, followed by
- **Later weaker regularization** to refine the coefficients of the remaining active features.

This idea is closely related to **Adaptive LASSO**, where the penalty is adapted to the importance of each coefficient. Here, the adaptation happens over **time** via a lambda schedule.

---

## 3. Optimization Theme

This project is built around the Numerical Optimization theme:

**Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression.**

We implement an **Adaptive LASSO optimizer** using **Proximal Gradient Descent**. The key elements are:
- A **LASSO-style objective** with an \(\ell_1\) penalty.
- A **proximal gradient descent** solver that separates the smooth (MSE) and non-smooth (L1) parts of the objective.
- A **dynamic soft-thresholding operator** driven by a time-varying \(\lambda_t\) schedule.

This setup allows us to study how different lambda schedules affect:
- Convergence behaviour,
- Feature sparsity, and
- Prediction accuracy.

---

## 4. Mathematical Background

### 4.1 LASSO Regression Objective

Given:
- Design matrix \(X \in \mathbb{R}^{n \times d}\),
- Target vector \(y \in \mathbb{R}^n\),
- Coefficients \(w \in \mathbb{R}^d\),

the **LASSO** objective is:
\[
\min_w \frac{1}{2n}\lVert y - Xw \rVert_2^2 + \lambda \lVert w \rVert_1,
\]
where:
- The first term is the **mean squared error (MSE)** loss,
- The second term is an **\(\ell_1\) penalty** scaled by \(\lambda\).

### 4.2 Why L1 Regularization Causes Sparsity

The \(\ell_1\) norm \(\lVert w \rVert_1 = \sum_j |w_j|\) has a **sharp corner at zero**, unlike the smooth \(\ell_2\) norm. When used as a penalty:
- The optimizer is encouraged to push some coefficients **exactly to zero**,
- This creates a **sparse** coefficient vector,
- Features with zero coefficients are effectively **deselected** from the model.

This property makes LASSO a natural tool for feature selection.

### 4.3 Proximal Gradient Descent

The LASSO objective combines:
- A smooth part: \(f(w) = \frac{1}{2n}\lVert y - Xw \rVert_2^2\),
- A non-smooth part: \(g(w) = \lambda \lVert w \rVert_1\).

**Proximal Gradient Descent** is designed for such composite objectives. At each iteration:
1. Take a **gradient step** on the smooth part \(f\),
2. Apply a **proximal operator** for the non-smooth part \(g\).

In this project we also include an **intercept** \(b\), so the smooth part is:
\[
f(w, b) = \frac{1}{2n}\lVert Xw + b - y \rVert_2^2.
\]
Its gradients are:
\[
\nabla_w f = \frac{1}{n} X^\top (Xw + b - y), \quad
\nabla_b f = \frac{1}{n} \sum_i (Xw + b - y)_i.
\]

#### Step 1: Gradient Descent on the MSE Loss

With learning rate \(\eta > 0\), a gradient step is:
\[
\tilde{w} = w - \eta \nabla_w f, \quad
b \leftarrow b - \eta \nabla_b f.
\]

#### Step 2: Soft-Thresholding Operator

The proximal operator for \(g(w) = \lambda \lVert w \rVert_1\) is the **soft-thresholding operator**:
\[
\text{prox}_{\eta \lambda}(z)_i
  = \operatorname{sign}(z_i)\,\max(|z_i| - \eta \lambda, 0).
\]

Thus, after the gradient step, we apply:
\[
w_i \leftarrow \operatorname{sign}(\tilde{w}_i)\,\max(|\tilde{w}_i| - \eta \lambda, 0).
\]

This is exactly the soft-thresholding rule implemented in this project. It:
- Shrinks coefficients toward zero, and
- Sets them **exactly to zero** when \(|\tilde{w}_i| \le \eta \lambda\).

### 4.4 Dynamic Lambda Schedules

Instead of a static \(\lambda\), this project uses a **time-varying** \(\lambda_t\) at iteration \(t\). The proximal step becomes:
\[
w_i \leftarrow \operatorname{sign}(\tilde{w}_i)\,\max(|\tilde{w}_i| - \eta \lambda_t, 0).
\]

We implement two dynamic schedules (plus a constant baseline):

- **Inverse square-root decay**:
  \[
  \lambda_t = \frac{\lambda_0}{\sqrt{t}}.
  \]
- **Exponential decay**:
  \[
  \lambda_t = \lambda_0 \exp(-k t),
  \]
  where \(k > 0\) controls how fast the decay happens.

The intuition:
- **Early iterations** (small \(t\)): \(\lambda_t\) is relatively large → strong soft-thresholding → many coefficients are driven to zero (“aggressive pruning”).
- **Later iterations** (large \(t\)): \(\lambda_t\) is smaller → weaker regularization → remaining non-zero coefficients are refined to reduce MSE.

This **aggressive early pruning + gradual stabilization** can yield models that:
- Are **sparser** than standard LASSO for a similar level of test MSE,
- Potentially offer a better interpretability–accuracy trade-off.

---

## 5. Dataset

### 5.1 House Prices: Advanced Regression Techniques

The project uses the **House Prices: Advanced Regression Techniques** dataset, a popular benchmark from Kaggle and OpenML. Each sample corresponds to a residential house sale, and the target is the **sale price**.

The dataset includes:
- **Housing attributes**: overall quality, year built, square footage, etc.
- **Structural features**: number of rooms, garage size, basement type, etc.
- **Location and quality indicators**: neighborhood, zoning, condition, etc.

This combination yields a **high-dimensional feature space** after categorical variables are one-hot encoded, making it a good testbed for feature selection techniques.

### 5.2 Preprocessing in `data_preprocessing.py`

All preprocessing steps are implemented in `src/data_preprocessing.py`:

- **Dataset loading**:
  - Uses `fetch_openml(name="house_prices", as_frame=True)` from scikit-learn.

- **Outlier removal**:
  - Removes extreme outliers in `GrLivArea` (above 4000 sq. ft.), as recommended by the dataset author, to stabilize the regression.

- **Dropping mostly-empty columns**:
  - Columns like `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu` are dropped when present, as they are sparse and mostly missing.

- **Target transformation**:
  - The target `SalePrice` is transformed as \(\log(1 + \text{SalePrice})\) (`np.log1p`) to reduce skewness and stabilize the regression.

- **Train/test split**:
  - The dataset is split into training and test sets using `train_test_split` with a fixed `random_state` for reproducibility.

- **Handling missing values and scaling**:
  - **Numerical features**:
    - Imputed using median values (`SimpleImputer(strategy="median")`).
    - Scaled using `StandardScaler` to have approximately zero mean and unit variance.
  - **Categorical features**:
    - Imputed using the most frequent category.
    - Encoded with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to produce a dense design matrix.

- **Saving processed arrays**:
  - Optionally saves:
    - `X_train.npy`, `X_test.npy`,
    - `y_train.npy`, `y_test.npy`
  - into `data/processed/` (relative to the project), enabling reuse by other scripts.

The function `get_processed_data` returns processed arrays and the fitted preprocessing pipeline:
\[
X_{\text{train}}, X_{\text{test}}, y_{\text{train}}, y_{\text{test}}, \text{preprocessor}.
\]

---

## 6. Project Pipeline

The complete workflow of the project is:

1. **Data preprocessing** (`data_preprocessing.py`)  
   - Load raw House Prices data from OpenML.
   - Remove outliers and sparse columns.
   - Impute missing values, encode categoricals, scale numericals.
   - Split into train/test sets and (optionally) save processed arrays.

2. **Baseline model training** (`baselines.py`)  
   - Train standard sklearn models:
     - Linear Regression,
     - Ridge Regression,
     - Standard LASSO.
   - Compute baseline metrics (MSE and sparsity).

3. **Adaptive LASSO optimization** (`adaptive_lasso.py`, `train_adaptive_lasso.py`)  
   - Initialize `AdaptiveLasso` with a chosen lambda schedule.
   - Run proximal gradient descent with dynamic soft-thresholding.
   - Track loss and sparsity over iterations.

4. **Experiment comparison** (`train_adaptive_lasso.py`)  
   - Evaluate all models (baselines + Adaptive LASSO) on the same test set.
   - Aggregate metrics into a unified JSON file.

5. **Visualization of results** (`visualization.py`)  
   - Plot optimization curves (loss and sparsity over iterations).
   - Plot model comparison charts (MSE vs sparsity).

This pipeline provides an end-to-end environment to study and compare feature selection behaviour in high-dimensional regression.

---

## 7. Code Structure

### 7.1 `src/data_preprocessing.py`

- Handles:
  - Dataset loading from OpenML.
  - Outlier removal.
  - Column dropping for sparse features.
  - Train/test splitting.
  - Missing value imputation.
  - Categorical encoding using `OneHotEncoder`.
  - Numerical scaling using `StandardScaler`.
  - Optional saving of processed arrays to `data/processed/`.
- Exposes `get_processed_data(save_to_disk=True)` which returns:
  - `X_train`, `X_test`, `y_train`, `y_test`, `preprocessor`.

### 7.2 `src/baselines.py`

- Implements baseline regression models using scikit-learn:
  - **Linear Regression** (no regularization),
  - **Ridge Regression** (L2 penalty),
  - **Standard LASSO** (L1 penalty).
- Core function:
  - `compute_baseline_metrics(X_train, X_test, y_train, y_test)`:
    - Trains each model.
    - Computes:
      - **Mean Squared Error (MSE)** on the test set.
      - **Number of non-zero coefficients** (using a small threshold to handle floating-point precision).
    - Returns a dictionary of metrics plus the total feature count.
- Convenience entry point:
  - `run_baselines()`:
    - Calls `get_processed_data` and trains all baselines.
    - Prints a summary table of results.

### 7.3 `src/adaptive_lasso.py`

- Defines the custom **`AdaptiveLasso`** class, implementing:
  - A **proximal gradient descent loop** for optimizing the LASSO-style objective with an intercept.
  - **Dynamic lambda scheduling** via:
    - `lambda_t = lambda0 / sqrt(t)` (inverse square-root),
    - `lambda_t = lambda0 * exp(-k * t)` (exponential decay),
    - or a constant lambda.
  - The **soft-thresholding operator**:
    \[
    w_i = \operatorname{sign}(w_i)\,\max(|w_i| - \eta \lambda_t, 0).
    \]
- API is similar to sklearn estimators:
  - `__init__(learning_rate, lambda0, n_iters, schedule, decay_k, tol, fit_intercept, random_state, verbose)`,
  - `fit(X, y)`,
  - `predict(X)`,
  - `compute_loss(X, y, w=None, b=None, lam=None)`,
  - `soft_threshold(w, threshold)` (static method),
  - `get_training_diagnostics()` for returning:
    - `loss_history`,
    - `sparsity_history`,
    - `lambda_history`.

### 7.4 `src/train_adaptive_lasso.py`

- Serves as the **main experiment runner**.
- Responsibilities:
  - Load processed data via `get_processed_data(save_to_disk=False)`.
  - Instantiate and train `AdaptiveLasso` with a specified configuration.
  - Compute **test MSE** and **number of non-zero coefficients** for Adaptive LASSO.
  - Call `compute_baseline_metrics` to train and evaluate:
    - Linear Regression,
    - Ridge Regression,
    - Standard LASSO.
  - Aggregate all metrics (baselines + Adaptive LASSO) into a single dictionary.
  - Save results and configuration into `results/metrics.json`.
  - Call visualization utilities to create plots (loss curve, sparsity curve, model comparison).
  - Print a summary comparison table in the console.

### 7.5 `src/visualization.py`

- Provides plotting utilities for analyzing experiments:
  - `plot_loss_curve(loss_history, save_path)`:
    - Plots **loss vs iterations** (objective value over time) for Adaptive LASSO.
  - `plot_sparsity_curve(sparsity_history, total_features, save_path)`:
    - Plots **non-zero coefficient count vs iterations**, conveying how sparsity evolves.
  - `plot_model_comparison(metrics, total_features, save_path)`:
    - Produces a two-panel bar chart showing:
      - Test MSE per model,
      - Non-zero coefficients per model.
- All plots are saved to the `results/` folder with informative filenames.

### 7.6 `README.md`

- The document you are currently reading. It provides:
  - Technical background,
  - Detailed description of the optimization method,
  - Explanation of the dataset and preprocessing,
  - Overview of the code structure and experiment pipeline,
  - Instructions for running and interpreting experiments.

---

## 8. Experiment Metrics

The project uses standard metrics to evaluate and compare models:

- **Mean Squared Error (MSE)**:
  \[
  \text{MSE} = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2.
  \]
  - Measures prediction accuracy on the test set.
  - Lower MSE indicates better fit.

- **Number of Non-Zero Coefficients**:
  - Counts how many entries of the coefficient vector \(w\) are non-zero (above a small threshold).
  - Indicates the **sparsity** of the model.
  - Fewer non-zero coefficients → stronger feature selection.

- **Sparsity Percentage** (conceptual, can be derived):
  \[
  \text{Sparsity \%} = \left(1 - \frac{\#\{\text{non-zero coefficients}\}}{\text{total features}}\right) \times 100.
  \]
  - Represents the fraction of features that have been effectively pruned.

The objective is to achieve **strong sparsity** (low number of non-zero coefficients / high sparsity %) while maintaining **competitive MSE** relative to standard LASSO and other baselines.

---

## 9. Result Visualizations

The following plots are generated and saved to the `results/` directory:

- **Loss vs Iterations (`loss_curve.png`)**
  - X-axis: iteration index.
  - Y-axis: objective value (MSE + L1 penalty) for Adaptive LASSO.
  - Shows the convergence behaviour of the proximal gradient algorithm.
  - A smoothly decreasing curve indicates stable optimization.

- **Sparsity vs Iterations (`sparsity_curve.png`)**
  - X-axis: iteration index.
  - Y-axis: number of non-zero coefficients in \(w\) at each iteration.
  - Shows how quickly and how strongly the model becomes sparse.
  - Typically, you see a rapid drop early on (aggressive pruning) and stabilization later.

- **Model Comparison (`model_comparison.png`)**
  - Two panels:
    - Test MSE across all models (Linear, Ridge, LASSO, Adaptive LASSO).
    - Number of non-zero coefficients across all models.
  - Helps visualize the trade-off between **accuracy** (MSE) and **sparsity** (non-zero count).
  - Adaptive LASSO is expected to achieve:
    - Similar MSE to standard LASSO,
    - Fewer non-zero coefficients, i.e., better sparsity.

These visualizations make it easier to reason about both the **optimization dynamics** and the **final model quality**.

---

## 10. How to Run the Project

### 10.1 Install Dependencies

Use Python 3.9+ and install the required packages:

```bash
pip install scikit-learn numpy pandas matplotlib
```

### 10.2 Run Preprocessing and Experiments

In most cases you can simply run the main training script, which will internally call the preprocessing pipeline if needed:

```bash
python src/train_adaptive_lasso.py
```

This will:
- Download and preprocess the House Prices dataset (if not already done).
- Train the Adaptive LASSO optimizer using proximal gradient descent with dynamic soft thresholding.
- Train baseline models (Linear, Ridge, LASSO).
- Compute evaluation metrics for all models.
- Save:
  - `results/metrics.json`,
  - `results/loss_curve.png`,
  - `results/sparsity_curve.png`,
  - `results/model_comparison.png`.

Optionally, you can:

- Run **only preprocessing**:
  ```bash
  python src/data_preprocessing.py
  ```

- Run **only baselines**:
  ```bash
  python src/baselines.py
  ```

---

## 11. Expected Outcome

The experiment is designed to demonstrate that **Adaptive LASSO with dynamic soft thresholding** can:

- **Reduce the number of selected features** compared to standard LASSO, by pruning aggressively early in training.
- **Maintain competitive prediction accuracy** (test MSE similar to standard LASSO and not much worse than Ridge or Linear Regression).
- **Improve interpretability** by producing a sparser model where a smaller subset of features carries most of the predictive power.

In summary, the Adaptive LASSO optimizer aims to achieve a better **sparsity–accuracy trade-off** than static L1 regularization.

---

## 12. Future Improvements

Several extensions and improvements are possible:

- **Adaptive per-feature lambda**  
  Instead of a single \(\lambda_t\) for all coefficients, introduce **feature-wise weights** (e.g., based on an initial estimator). This would align more closely with the classical Adaptive LASSO formulation:
  \[
  \sum_j \omega_j |w_j|, \quad \omega_j = \frac{1}{|\hat{w}_j|^\gamma}.
  \]

- **Coordinate Descent Comparison**  
  Implement a coordinate descent solver for LASSO and Adaptive LASSO, and compare:
  - Convergence speeds,
  - Final sparsity patterns,
  - Sensitivity to hyperparameters.

- **Cross-Validation Tuning**  
  Use cross-validation to select:
  - The base regularization strength \(\lambda_0\),
  - The decay rate \(k\) for exponential schedules,
  - The learning rate \(\eta\) and number of iterations.
  This would make the method more robust and production-ready.

- **Alternative Schedules and Continuation Strategies**  
  Experiment with other schedules (e.g., piecewise-constant, linear decay, or more general continuation methods) to further study their impact on feature selection and convergence behaviour.

These directions can form the basis for deeper exploration in numerical optimization and sparse modeling.
