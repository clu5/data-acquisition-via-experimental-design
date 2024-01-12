import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def least_norm_linear_regression(X, y):
    """
    Compute the least norm linear regression solution.

    Parameters:
    - X: Input feature matrix (n_samples, n_features)
    - y: Target values (n_samples,)

    Returns:
    - Coefficients of the linear regression model
    """

    # Compute the least squares solution using the Moore-Penrose pseudo-inverse
    coefficients = np.linalg.pinv(X).dot(y)

    return coefficients


def MSE(X, y, coeff):
    """
    Compute the Mean Squared Error (MSE) for linear regression.

    Parameters:
    - X: Input feature matrix (n_samples, n_features)
    - y: Actual target values (n_samples,)
    - coeff: Coefficients of the linear regression model (n_features,)

    Returns:
    - Mean Squared Error (MSE)
    """
    # Compute predicted y values
    y_pred = X.dot(coeff)

    # Compute the squared differences between predicted and actual y values
    squared_errors = (y - y_pred) ** 2

    # Compute the mean of squared errors to get MSE
    mse = np.mean(squared_errors)

    return mse

def plot_matrix(symmetric_matrix):
  """
  Plots the top and bottom eigenvalue as an ellipse to visualize a matrix.

  Parameters:
  - symmetric_matrix: A symmetric PSD matrix

  Returns:

  """
  # Find eigenvalues and eigenvectors
  eigenvalues, _ = np.linalg.eigh(symmetric_matrix)
  eigenvalues = np.abs(eigenvalues)
  # Sort eigenvalues in ascending order
  eigenvalues = np.sort(eigenvalues)

  # Largest and smallest eigenvalues
  largest_eigenvalue = eigenvalues[-1]
  smallest_eigenvalue = eigenvalues[0]

  # Create an ellipse using the largest and smallest eigenvalues
  theta = np.linspace(0, 2 * np.pi, 100)
  a = np.sqrt(largest_eigenvalue)
  b = np.sqrt(smallest_eigenvalue)
  x = a * np.cos(theta)
  y = b * np.sin(theta)

  # Plot the ellipse
  plt.figure(figsize=(6, 6))
  plt.plot(x, y, label='Ellipse', color='b')

  # Set plot limits, labels, and legend
  plt.xlim(-a, a)
  plt.ylim(-b, b)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend()

  # Show the plot
  plt.gca().set_aspect('equal', adjustable='box')
  plt.grid()
  plt.show()


def project_onto_subspace(v, W):
    """
    Project vector v onto the subspace spanned by the row of W.

    Args:
    - v (np.ndarray): The vector(s) to be projected. Shape (d,) or (n, d).
    - W (np.ndarray): The matrix whose columns span the subspace. Shape (k, d).

    Returns:
    - np.ndarray: The projection of v onto the subspace. Shape (d,) or (n, d).
    """
    # Compute the projection
    # proj = v @ W.T @ np.linalg.inv(W @ W.T) @ W
    proj = v @ W.T @ np.linalg.pinv(W @ W.T) @ W
    
    return proj


def measure_coverage(X_selected, X_buy):
    """
    Measure coverage as precision and recall. Higher is better for both.
    Precision is a number in [0,1] measuring how much of the selected datapoints are relevant to the buyer.
    Recall is a number in [0,1] measuring how much of the buyer's data is covered by selected datapoints.

    Args:
    - X_selected (np.ndarray): Shape (K, d).
    - X_buy (np.ndarray): Shape (m, d).

    Returns:
    - float, float both in range [0,1]: Precision, Recall
    """
    # How much of the selected datapoints are relevant to buy?
    proj_onto_buy = project_onto_subspace(X_selected, X_buy)
    precision = np.mean(np.linalg.norm(proj_onto_buy, axis=1))

    # How much of the buy is covered by selected datapoints?
    proj_onto_sell = project_onto_subspace(X_buy, X_selected)
    recall = np.mean(np.linalg.norm(proj_onto_sell, axis=1))

    return precision, recall

def evaluate_indices(X_sell, y_sell, X_buy, y_buy, data_indices, inverse_covariance=None):
    # Train a linear model from the subselected sellers data
    X_selected = X_sell[data_indices]
    coeff_hat = least_norm_linear_regression(X_selected, y_sell[data_indices])
    buy_error = MSE(X_buy, y_buy, coeff_hat)
    if inverse_covariance is None:
        inverse_covariance = np.linalg.pinv(X_selected.T @ X_selected)
    exp_loss = compute_exp_design_loss(X_buy, inverse_covariance)
    # precision, recall = measure_coverage(X_selected, X_buy)
    return {
        "exp_loss": exp_loss, 
        # "precision" : precision, 
        # "recall" : recall, 
        "mse_error" : buy_error, 
    }
    

def sherman_morrison_update_inverse(A_inv, u, v):
    """
    Update the inverse of a matrix A_inv after a rank-one update (A + uv^T).

    Parameters:
    - A_inv: The inverse of the original matrix A (d,d)
    - u: Column vector u (d,)
    - v: Column vector v (d,)

    Returns:
    - The inverse of (A + uv^T)
    """

    # Calculate the denominator term (1 + v^T * A_inv * u)
    denominator = 1. + v.T @ A_inv @ u

    # Calculate the update term (A_inv * u * v^T * A_inv)
    update_term = (A_inv @ u)[:, None] @ (v.T @ A_inv)[None, :]

    # Update the inverse using the Sherman-Morrison formula
    updated_inverse = A_inv - (update_term / denominator)

    return updated_inverse


def compute_exp_design_loss(X_buy, inverse_covariance):
    """
    Compute the experiment design loss.

    Parameters:
    - X_buy: Buyer data matrix of shape (n_buy, d)
    - inverse_covariance: Inverse covariance matrix of shape (d, d)

    Returns:
    - loss value
    """

    # Compute the matrix product E[x_0^T P x_0]
    return np.mean((X_buy @ inverse_covariance) * X_buy) * X_buy.shape[-1]
    # return np.einsum('ij,jk,ik->ik', X_buy, inverse_covariance, X_buy).mean()

def compute_neg_gradient(X_sell, X_buy, inverse_covariance):
    """
    Compute the negative gradient vector of the exp design loss.

    Parameters:
    - X_sell: Sellers data matrix of shape (n_sell, d)
    - X_buy: Buyer data matrix of shape (n_buy, d)
    - inverse_covariance: Inverse covariance matrix of shape (d, d)

    Returns:
    - Gradient vector of shape (n_sell,)
    """

    # Compute the intermediate matrix product  x_i^T P x_0
    product_matrix = X_sell @ inverse_covariance @ X_buy.T

    # Calculate the squared norms of rows E(x_i^T P x_0)^2
    neg_gradient = np.mean(product_matrix ** 2, axis=1)

    return neg_gradient

# Define the experiment design loss function
def opt_step_size(X_sell_data, X_buy, inverse_covariance, old_loss, lower=1e-3):
    """
    Compute the optimal step size to minimize exp design loss along chosen coordinate .

    Parameters:
    - X_sell_data: Sellers data being updated (d,)
    - X_buy: Buyer data matrix of shape (n_buy, d)
    - inverse_covariance: Inverse covariance matrix of shape (d, d)
    - old_loss: previous value of loss

    Returns:
    - optimal step size (value in [0,1])
    - new loss
    """
    # OPTION I: recopmute loss for different updated inverse matrix.
    # def new_loss(alpha):
    #     updated_inv = sherman_morrison_update_inverse(
    #         inverse_covariance / (1-alpha),
    #         alpha * X_sell_data,
    #         X_sell_data,
    #     )
    #     return np.mean((X_buy @ updated_inv) * X_buy)

    # OPTION II: efficient line search by reusing computations
    a = old_loss
    # # E(x_0 P x_i)^2
    prod = (X_sell_data.T @ inverse_covariance) @ X_buy.T
    b = np.mean(prod ** 2)
    c = X_sell_data @ inverse_covariance @ X_sell_data

    # print(a, b, c)
    # Compute optimal step size
    loss = lambda x: (1 / (1 - x)) * (a - (x * b) / (1 - x * (1 - c)))
    # result = minimize_scalar(loss, bounds=(lower, 0.9))
    result = minimize_scalar(loss, bounds=(0, 0.9))
    return result.x, result.fun


# One-step baseline
def one_step(X_sell, X_buy):
    # inv_cov = np.linalg.inv(X_sell.T @ X_sell)
    inv_cov = np.linalg.inv(np.cov(X_sell.T))
    one_step_values = np.mean((X_sell @ inv_cov @ X_buy.T) ** 2, axis=1)
    return one_step_values