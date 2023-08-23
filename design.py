from time import perf_counter
import numpy as np
import cvxpy as cp


class Valuator:
    """ Data valuation using optimal experimental design with continous relaxation.

    Parameters
    ----------
    buyer_data: ndarray
        buyer data in samples x features format
    number_of_components: int, optional
        componetns to use in PCA

    """
    def __init__(self, print_info=True):
        self.print_info = print_info

    def optimize(self, buyer_data: np.ndarray, seller_data: np.ndarray) -> np.ndarray:
        """Return valuation for each seller data point.

        Returns
        -------
        np.ndarray
            Optimized weights for each seller data point
        """
        def objective(buyer_data, seller_data, seller_weights):
            m = buyer_data.shape[0]
            W = cp.diag(seller_weights)
            cost = cp.matrix_frac(buyer_data.T, (seller_data.T @ W @ seller_data) )
            return cost / m

        seller_weights = cp.Variable(seller_data.shape[0])
        constraints = [seller_weights >= 0, sum(seller_weights) == 1]
        prob = cp.Problem(cp.Minimize(objective(buyer_data, seller_data, seller_weights)), constraints)
        start = perf_counter()
        prob.solve()
        end = perf_counter()
        runtime = end - start
        if self.print_info:
            print(prob.status, f"{runtime=:.1f}\t{prob.value=:.2f}")
        return seller_weights.value.flatten()
