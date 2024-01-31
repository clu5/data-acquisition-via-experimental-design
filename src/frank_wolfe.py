import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm


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
    plt.plot(x, y, label="Ellipse", color="b")

    # Set plot limits, labels, and legend
    plt.xlim(-a, a)
    plt.ylim(-b, b)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Show the plot
    plt.gca().set_aspect("equal", adjustable="box")
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


def evaluate_indices(
    X_sell, y_sell, X_buy, y_buy, data_indices, inverse_covariance=None
):
    """
    Evaluate the performance of selected data indices on the buyer's data.

    Parameters:
    - X_sell (numpy.ndarray): Sellers data matrix (n_sell, n_features)
    - y_sell (numpy.ndarray): Sellers target values (n_sell,)
    - X_buy (numpy.ndarray): Buyers data matrix (n_buy, n_features)
    - y_buy (numpy.ndarray): Buyers target values (n_buy,)
    - data_indices (numpy.ndarray): Indices of selected data points
    - inverse_covariance (numpy.ndarray, optional): Inverse covariance matrix (n_features, n_features)

    Returns:
    - dict: Dictionary containing expected loss, and MSE error
    """
    # Train a linear model from the subselected sellers data
    X_selected = X_sell[data_indices]
    coeff_hat = least_norm_linear_regression(X_selected, y_sell[data_indices])
    buy_error = MSE(X_buy, y_buy, coeff_hat)
    if inverse_covariance is None:
        inverse_covariance = np.linalg.pinv(X_selected.T @ X_selected)
    exp_loss = compute_exp_design_loss(X_buy, inverse_covariance)
    return {
        "exp_loss": exp_loss,
        "mse_error": buy_error,
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
    denominator = 1.0 + v.T @ A_inv @ u

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
    - float: loss value
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
    neg_gradient = np.mean(product_matrix**2, axis=1)

    return neg_gradient


# Define the experiment design loss function
def opt_step_size(X_sell_data, X_buy, inverse_covariance, old_loss, lower=1e-3):
    """
    Compute the optimal step size to minimize exp design loss along chosen coordinate .

    Parameters:
    - X_sell_data: Sellers data being updated (n_sell, d,)
    - X_buy: Buyer data matrix of shape (n_buy, d)
    - inverse_covariance: Inverse covariance matrix of shape (d, d)
    - old_loss: previous value of loss
    - lower (float): Lower bound for the step size optimization


    Returns:
    - float: Optimal step size (value in [0,1])
    - float: New loss after applying the optimal step size
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
    b = np.mean(prod**2)
    c = X_sell_data @ inverse_covariance @ X_sell_data

    # print(a, b, c)
    # Compute optimal step size
    loss = lambda x: (1 / (1 - x)) * (a - (x * b) / (1 - x * (1 - c)))
    # result = minimize_scalar(loss, bounds=(lower, 0.9))
    result = minimize_scalar(loss, bounds=(0, 0.9))
    return result.x, result.fun


def one_step(X_sell, X_buy):
    """
    Compute one-step baseline
    Parameters:
    - X_sell: Sellers data being updated (n_sell, d)
    - X_buy: Buyer data matrix of shape (n_buy, d)

    Returns:
    - numpy.ndarray: shape (n_sell)
    """
    inv_cov = np.linalg.pinv(np.dot(X_sell.T, X_sell))
    one_step_values = np.mean((X_sell @ inv_cov @ X_buy.T) ** 2, axis=1)
    return one_step_values


def design_selection(
    X_sell,
    y_sell,
    X_buy,
    y_buy,
    num_select=10,
    num_iters=1000,
    alpha=0.01,
    line_search=True,
    recompute_interval=50,
    early_stop_threshold=None,
    sampling_selection_error=True,
    costs=None,
    return_grads=False,
    reg_lambda=0.0,
):
    """
    Select data points based on experimental design optimization.

    Parameters:
    - X_sell (numpy.ndarray): Sellers data matrix (n_sell, n_features)
    - y_sell (numpy.ndarray): Sellers target values (n_sell,)
    - X_buy (numpy.ndarray): Buyers data matrix (n_buy, n_features)
    - y_buy (numpy.ndarray): Buyers target values (n_buy,)
    - num_select (int): Number of seller data points to evaluate each step
    - num_iters (int): Number of step iterations
    - alpha (float, optional): Set manual step size for weight update
    - line_search (bool): Whether to use line search for optimizing step size
    - recompute_interval (int): Interval for recomputing the inverse covariance matrix
    - early_stop_threshold (float, optional): Early stopping threshold for step size
    - sampling_selection_error (bool): Whether to use sampling weights for evaluating seller indices
    - costs (numpy.ndarray, optional): Costs associated with each seller data point
    - use_identity (bool): Initialize inverse covariance to be the identity matrix

    Returns:
    - dict: Dictionary containing tracking information of the optimization process
    """

    # initialize seller weights
    n_sell = X_sell.shape[0]
    weights = np.ones(n_sell) / n_sell

    # Compute inverse covariance matrix
    if reg_lambda > 0:
        reg = np.eye(X_sell.shape[1]) @ np.diag(X_sell.std(0))
        cov = X_sell.T @ np.diag(weights) @ X_sell
        reg_cov = (1-reg_lambda) * cov + reg_lambda * reg
        inv_cov = np.linalg.pinv(reg_cov)
    else:
        inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)

    # experimental design loss i.e. E[X_buy.T @ inv_cov @ X]
    loss = compute_exp_design_loss(X_buy, inv_cov)

    # track losses and errors
    losses = {}
    errors = {}
    coords = {}
    alphas = {}
    if return_grads:
        grads = {}

    if costs is not None:
        err_msg = f"cost vector should have same length as seller data"
        assert costs.shape[0] == n_sell, f"{err_msg}: should be {n_sell}"
        err_msg = f"cost vector should be strictly positive"
        assert (costs > 0).all(), f"{err_msg}"

    for i in tqdm(range(num_iters)):
        # Recomute actual inverse covariance to periodically recalibrate
        if recompute_interval > 0 and i % recompute_interval == 0:
            inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)

        # Pick coordinate with largest gradient to update
        neg_grad = compute_neg_gradient(X_sell, X_buy, inv_cov)

        if return_grads:
            grads[i] = neg_grad

        if costs is not None:
            neg_grad *= 1 / costs

        update_coord = np.argmax(neg_grad)

        coords[i] = update_coord

        # Optimize step size with line search
        if line_search or alpha is None:
            alpha, line_loss = opt_step_size(X_sell[update_coord], X_buy, inv_cov, loss)

        # Terminate early if step size is too small
        if early_stop_threshold is not None and alpha < early_stop_threshold:
            break

        alphas[i] = alpha

        # Update weight vector
        weights *= 1 - alpha  # shrink weights by 1 - alpha
        weights[update_coord] += alpha  # increase magnitude of picked coordinate

        # Update inverse covariance matrix
        inv_cov /= 1 - alpha  # Update with respect to weights shrinking

        # update with respect to picked coordinate increasing
        inv_cov = sherman_morrison_update_inverse(
            inv_cov,
            alpha * X_sell[update_coord, :],
            X_sell[update_coord, :],
        )

        if sampling_selection_error:
            selected_seller_indices = np.random.choice(
                np.arange(weights.shape[0]),
                size=num_select,
                p=weights / weights.sum(),
                replace=False,
            )
        else:
            selected_seller_indices = weights.argsort()[::-1][:num_select]

        results = evaluate_indices(
            X_sell,
            y_sell,
            X_buy,
            y_buy,
            selected_seller_indices,
            inverse_covariance=inv_cov,
        )
        losses[i] = compute_exp_design_loss(X_buy, inv_cov)
        errors[i] = results["mse_error"]

    ret = dict(
        losses=losses,
        errors=errors,
        weights=weights,
        coords=coords,
        alphas=alphas,
    )
    if return_grads:
        ret['grads'] = grads
    return ret
