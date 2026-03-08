import numpy as np
from typing import Dict, List, Optional, Tuple


class AdaptiveLasso:
    """
    Adaptive LASSO regression trained via Proximal Gradient Descent
    with a dynamic soft-thresholding schedule.

    The objective optimized is
        (1 / (2n)) * ||y - (Xw + b)||_2^2 + lambda_t * ||w||_1
    where lambda_t is allowed to change with iteration t.

    Parameters
    ----------
    learning_rate : float, default=1e-3
        Step size for the gradient descent update.
    lambda0 : float, default=0.1
        Base regularization strength used in the lambda schedule.
    n_iters : int, default=1000
        Maximum number of proximal gradient iterations.
    schedule : {'inverse_sqrt', 'exp_decay', 'constant'}, default='inverse_sqrt'
        Type of dynamic schedule for lambda_t:
        - 'inverse_sqrt': lambda_t = lambda0 / sqrt(t)
        - 'exp_decay'  : lambda_t = lambda0 * exp(-decay_k * t)
        - 'constant'   : lambda_t = lambda0
    decay_k : float, default=0.01
        Exponential decay rate used when schedule='exp_decay'.
    tol : float, default=1e-6
        Tolerance for early stopping based on relative loss improvement.
    fit_intercept : bool, default=True
        Whether to fit an unpenalized intercept term.
    random_state : Optional[int], default=None
        Random seed used for reproducible initialization.
    verbose : bool, default=False
        If True, prints loss and sparsity statistics during training.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Learned regression coefficients.
    intercept_ : float
        Learned intercept term (0.0 when fit_intercept=False).
    n_iter_ : int
        Number of iterations actually run.
    loss_history_ : List[float]
        Loss values (MSE + L1 penalty) at each iteration.
    sparsity_history_ : List[int]
        Number of non-zero coefficients at each iteration.
    lambda_history_ : List[float]
        Value of lambda_t used at each iteration.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        lambda0: float = 0.1,
        n_iters: int = 1000,
        schedule: str = "inverse_sqrt",
        decay_k: float = 0.01,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.lambda0 = lambda0
        self.n_iters = n_iters
        self.schedule = schedule
        self.decay_k = decay_k
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        # Attributes set during fitting
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.loss_history_: List[float] = []
        self.sparsity_history_: List[int] = []
        self.lambda_history_: List[float] = []

    def _lambda_t(self, t: int) -> float:
        """
        Compute the dynamic regularization strength lambda_t at iteration t.

        Parameters
        ----------
        t : int
            Current iteration index (1-based).

        Returns
        -------
        float
            Value of lambda_t for this iteration.
        """
        if self.schedule == "inverse_sqrt":
            return self.lambda0 / np.sqrt(t)
        if self.schedule == "exp_decay":
            return float(self.lambda0 * np.exp(-self.decay_k * t))
        if self.schedule == "constant":
            return self.lambda0
        raise ValueError(
            f"Unknown schedule '{self.schedule}'. "
            "Supported: 'inverse_sqrt', 'exp_decay', 'constant'."
        )

    @staticmethod
    def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply the element-wise soft-thresholding operator.

        For each coefficient w_i:
            w_i <- sign(w_i) * max(|w_i| - threshold, 0).

        This is the proximal operator for the L1 norm and is
        responsible for inducing sparsity in the solution.

        Parameters
        ----------
        w : np.ndarray
            Current coefficient vector.
        threshold : float
            Non-negative threshold (typically learning_rate * lambda_t).

        Returns
        -------
        np.ndarray
            Thresholded coefficient vector.
        """
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

    def compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
        b: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> float:
        """
        Compute the LASSO objective (MSE + L1 penalty) for given parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.
        y : np.ndarray of shape (n_samples,)
            Target vector.
        w : np.ndarray of shape (n_features,), optional
            Coefficient vector. If None, uses self.coef_.
        b : float, optional
            Intercept term. If None, uses self.intercept_.
        lam : float, optional
            Regularization strength. If None, uses self.lambda0.

        Returns
        -------
        float
            Value of the objective function.
        """
        if w is None:
            if self.coef_ is None:
                raise ValueError("Model is not fitted yet; pass w explicitly.")
            w = self.coef_
        if b is None:
            b = self.intercept_
        if lam is None:
            lam = self.lambda0

        n_samples = X.shape[0]
        y_pred = X @ w + b
        residual = y_pred - y
        mse_term = 0.5 * np.mean(residual**2)
        l1_term = lam * np.sum(np.abs(w))
        return float(mse_term + l1_term)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveLasso":
        """
        Fit the Adaptive LASSO model using proximal gradient descent.

        The update at iteration t is:
            1. Compute gradient of the MSE loss:
                   grad_w = (1/n) * X^T (Xw + b - y)
                   grad_b = (1/n) * sum(Xw + b - y)
            2. Gradient descent step:
                   w_tilde = w - learning_rate * grad_w
                   b      = b - learning_rate * grad_b
            3. Proximal (soft-thresholding) step:
                   w = soft_threshold(w_tilde, learning_rate * lambda_t)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.
        y : np.ndarray of shape (n_samples,) or (n_samples, 1)
            Target vector.

        Returns
        -------
        AdaptiveLasso
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)
        w = rng.normal(loc=0.0, scale=0.01, size=n_features)
        b = 0.0 if self.fit_intercept else 0.0

        self.loss_history_ = []
        self.sparsity_history_ = []
        self.lambda_history_ = []

        prev_loss = np.inf

        for t in range(1, self.n_iters + 1):
            y_pred = X @ w + b
            residual = y_pred - y

            grad_w = (X.T @ residual) / n_samples
            if self.fit_intercept:
                grad_b = float(np.sum(residual) / n_samples)
            else:
                grad_b = 0.0

            # Gradient step
            w = w - self.learning_rate * grad_w
            b = b - self.learning_rate * grad_b

            # Dynamic regularization for this iteration
            lam_t = self._lambda_t(t)

            # Proximal step (soft-thresholding)
            w = self.soft_threshold(w, self.learning_rate * lam_t)

            # Track statistics
            loss = self.compute_loss(X, y, w=w, b=b, lam=lam_t)
            non_zero = int(np.sum(np.abs(w) > 1e-6))

            self.loss_history_.append(loss)
            self.sparsity_history_.append(non_zero)
            self.lambda_history_.append(lam_t)

            if self.verbose and t % max(1, self.n_iters // 10) == 0:
                print(
                    f"Iter {t:5d} | loss={loss:.6f} | "
                    f"lambda_t={lam_t:.6f} | non-zero={non_zero}/{n_features}"
                )

            # Early stopping based on relative loss improvement
            if prev_loss < np.inf:
                improvement = prev_loss - loss
                if prev_loss > 0:
                    rel_improvement = improvement / prev_loss
                else:
                    rel_improvement = 0.0
                if rel_improvement >= 0 and rel_improvement < self.tol:
                    if self.verbose:
                        print(
                            f"Early stopping at iter {t} "
                            f"(relative improvement {rel_improvement:.2e} < tol={self.tol})"
                        )
                    self.n_iter_ = t
                    break

            prev_loss = loss
            self.n_iter_ = t

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def get_training_diagnostics(self) -> Dict[str, List[float]]:
        """
        Return diagnostic information collected during training.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'loss_history'
            - 'sparsity_history'
            - 'lambda_history'
        """
        return {
            "loss_history": list(self.loss_history_),
            "sparsity_history": list(self.sparsity_history_),
            "lambda_history": list(self.lambda_history_),
        }


__all__ = ["AdaptiveLasso"]

