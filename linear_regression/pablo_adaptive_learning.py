import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# Set up the data


# Generate more data points with a linear relationship plus some noise
# Generate synthetic data with a linear relationship plus noise
np.random.seed(42)  # For reproducibility
N = 100  # Number of samples
X_raw = np.random.uniform(0, 10, N)  # Generate random x values between 0 and 10
X = np.column_stack((np.ones(N), X_raw))  # Add bias column
W = np.array([5.0, 2.0])  # True weights: larger intercept=5, steeper slope=2 for noisier data
noise = 0.2 * np.random.randn(N)  # Add some Gaussian noise
Y = X @ W + noise  # Generate target values
# W = np.linalg.lstsq(X, Y, rcond=None)[0]


# # Initialize Precision Vectors

# pi_0 = np.array([0.25, 0.5, 1.0, 1.5, 2.0])  # More options for intercept around 1.66
# pi_1 = np.array([0.125, 0.25, 0.375, 0.5, 0.625])  # More options for slope around 0.53
# precision_vector_list = [pi_0, pi_1]


# Create precision matrix
def getPrecisionMatrix(precision_vector_list):
    def create_binary_matrix(i, k):
        arr = np.zeros((k, k))
        arr[i,i] = 1
        return arr

    P_list = []

    for i in range(len(precision_vector_list)):
        I = create_binary_matrix(i, 2)
        P = np.kron(I, precision_vector_list[i])
        P_list.append(P)

    P = np.sum(P_list, axis=0)
    return P


def build_QUBO(X, Y, P):
    XtX = X.T @ X
    XtY = X.T @ Y
    
    A = P.T @ XtX @ P   # (D*K x D*K)
    b = -2 * (P.T @ XtY)
    return A, b


def check_if_z_is_in_list(z, z_list):
    for z_i in z_list:
        if np.array_equal(z, z_i):
            return True
    return False


def mock_qubo_solver(A, b, num_samples=2000):
    """
    A simple (and very naive) "solver" that samples random binary vectors
    and picks the one with the lowest QUBO energy.  This is purely illustrative:
    in a real application, you would call a real solver (e.g. a quantum annealer).
    
    QUBO:  z^T A z + z^T b   where z in {0,1}^M.
    
    Returns:
        best_z (M,): the best binary solution found
    """
    M = A.shape[0]
    best_z = None
    best_energy = np.inf
    z_list = []

    for _ in range(num_samples):
        candidate_z = np.random.randint(0, 2, size=M)  
        if not check_if_z_is_in_list(candidate_z, z_list):
            z_list.append(candidate_z)
            print(candidate_z)
            # QUBO energy = z^T A z + z^T b + Y^T Y
            energy = candidate_z @ A @ candidate_z + candidate_z @ b
            if energy < best_energy:
                best_energy = energy
                best_z = candidate_z
        
    return best_z


def adaptive_precision_search(X, Y, initial_ranges=None, num_iterations=3):
    """
    Adaptively search for better precision vectors without estimating w directly
    """
    if initial_ranges is None:
        # Start with wide ranges based on data scale
        y_scale = np.abs(Y).max()
        initial_ranges = {
            'intercept': np.array([-y_scale, y_scale]),
            'slope': np.array([-y_scale/X[:, 1].max(), y_scale/X[:, 1].max()])
        }
    
    best_energy = np.inf
    best_z = None
    best_precision_vectors = None
    
    for iteration in range(num_iterations):
        # Create precision vectors for current ranges
        pi_0 = np.linspace(initial_ranges['intercept'][0], 
                          initial_ranges['intercept'][1], 
                          5)
        pi_1 = np.linspace(initial_ranges['slope'][0], 
                          initial_ranges['slope'][1], 
                          5)
        
        precision_vector_list = [pi_0, pi_1]
        P = getPrecisionMatrix(precision_vector_list)
        A, b = build_QUBO(X, Y, P)
        
        # Solve QUBO
        z = mock_qubo_solver(A, b)
        energy = z @ A @ z + z @ b
        
        if energy < best_energy:
            best_energy = energy
            best_z = z
            best_precision_vectors = precision_vector_list
            
            # Refine ranges around the best solution found
            w_current = P @ z
            
            # Narrow the search range around the current best solution
            initial_ranges = {
                'intercept': np.array([
                    w_current[0] - (pi_0[1] - pi_0[0])/2,
                    w_current[0] + (pi_0[1] - pi_0[0])/2
                ]),
                'slope': np.array([
                    w_current[1] - (pi_1[1] - pi_1[0])/2,
                    w_current[1] + (pi_1[1] - pi_1[0])/2
                ])
            }
        else:
            # If no improvement, expand the ranges slightly
            initial_ranges = {
                'intercept': initial_ranges['intercept'] * 1.5,
                'slope': initial_ranges['slope'] * 1.5
            }
    
    return best_precision_vectors, best_z


if __name__ == "__main__":
    precision_vector_list, best_z = adaptive_precision_search(X, Y)
    P = getPrecisionMatrix(precision_vector_list)
    w = P @ best_z
    print("w:", w)
    y_hat = X@w
    print("Predicted y:", y_hat)
    print("Actual y:", Y)
    print("RSS:", np.sum((Y - y_hat)**2))
    print("R^2 score:", r2_score(Y, y_hat))
    # Create points for the line
    x_line = np.linspace(0, 10, 100)
    y_line = W[0] + W[1] * x_line

    y_hat_line = w[0] + w[1] * x_line

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1], Y, color='blue', label='Data points')
    plt.plot(x_line, y_line, 'r--', label=f'y = {W[0]:.2f} + {W[1]:.2f}x')
    plt.plot(x_line, y_hat_line, 'g--', label=f'y_hat = {w[0]:.2f} + {w[1]:.2f}x')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()