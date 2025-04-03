import numpy as np

###############################################################################
# Mock Data Generation
###############################################################################
def generate_mock_data(N=100, D=5, noise=0.01, seed=42):
    """
    Generates a mock linear regression dataset with N samples, D features.
    Returns:
        X (N x D): Features with a column of ones for the bias term
        Y (N,):    Targets
        w_true (D,): True weight vector used to generate Y
    """
    np.random.seed(seed)
    # Generate random features
    X_raw = np.random.randn(N, D-1)
    # Add bias column of 1s as the first "feature"
    X = np.hstack([np.ones((N, 1)), X_raw])
    
    # Generate random true weights
    w_true = np.random.rand(D)
    
    # Generate targets with added Gaussian noise
    Y = X @ w_true + noise * np.random.randn(N)
    return X, Y, w_true

###############################################################################
# QUBO Construction
###############################################################################
def build_qubo_matrices(X, Y, precision_vector):
    """
    Builds QUBO matrices A, b for the binary-encoded regression problem:
         min_{ŵ in {0,1}}   (Pŵ)^T X^T X (Pŵ) - 2 (Pŵ)^T X^T Y
    where w = Pŵ, with P determined by the chosen precision_vector.
    
    We arrange ŵ = [ŵ(1), ..., ŵ(K), ..., ŵ(D*K)] in one large binary vector.
    
    Returns:
        A (M x M):  Quadratic terms in QUBO
        b (M,):     Linear terms in QUBO
    """
    # Number of (rows, features)
    N, D = X.shape
    K = len(precision_vector)  # length of the precision vector
    
    # Construct matrix P of size (D x D*K)
    # w = P @ ŵ
    # Each row i of P is [0,..., 0, precision_vector, 0, ..., 0]
    # so that w_i = sum_{k=1..K} precision_vector[k] * ŵ_{i,k}.
    P = np.zeros((D, D * K))
    for i in range(D):
        for k in range(K):
            P[i, i*K + k] = precision_vector[k]
    
    # For convenience in QUBO terms, define:
    # E = w^T (X^T X) w - 2 w^T (X^T Y).
    # Then substituting w = P ŵ, we get:
    # E = ŵ^T (P^T X^T X P) ŵ - 2 ŵ^T (P^T X^T Y).
    
    XtX = X.T @ X
    XtY = X.T @ Y
    
    A = P.T @ XtX @ P   # (D*K x D*K)
    b = -2 * (P.T @ XtY)
    return A, b

###############################################################################
# R^2 Calculation
###############################################################################
def r2_score(X, Y, w):
    """
    Coefficient of determination for predictions X@w vs. true targets Y.
    """
    preds = X @ w
    ss_res = np.sum((Y - preds)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    return 1.0 - ss_res / ss_tot

###############################################################################
# Mock QUBO Solver
###############################################################################
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
    
    for _ in range(num_samples):
        candidate_z = np.random.randint(0, 2, size=M)
        # QUBO energy = z^T A z + z^T b
        energy = candidate_z @ A @ candidate_z + candidate_z @ b
        if energy < best_energy:
            best_energy = energy
            best_z = candidate_z
    
    return best_z

###############################################################################
# Adaptive Precision Algorithm (simplified)
###############################################################################
def adaptive_precision_linreg(X, Y, 
                              init_precision, 
                              rate=0.1, 
                              rate_desc=2.0, 
                              rate_asc=1.5, 
                              max_iters=10, 
                              num_samples=2000):
    """
    Demonstrates a simple adaptive approach to refine the precision vector
    for the QUBO-based linear regression.  Here we keep a single precision vector
    shared by all coefficients, for brevity.
    
    Args:
        X, Y : data
        init_precision : initial array of length K for the binary encoding
        rate, rate_desc, rate_asc : scaling parameters controlling how the 
            precision is shrunk/expanded
        max_iters : maximum number of refinement steps
        num_samples : how many random solutions to sample in the mock solver
    Returns:
        best_w : best real-valued weight vector found
        best_r2 : best R^2
        best_precision : the precision vector that achieved best results
    """
    # Setup
    D = X.shape[1]
    K = len(init_precision)
    precision_vec = init_precision.copy()
    
    # Compute baseline (e.g. random guess)
    A, b = build_qubo_matrices(X, Y, precision_vec)
    best_z = mock_qubo_solver(A, b, num_samples=num_samples)
    # Convert best_z from binary to real weights
    # w_i = sum_{k=1..K} precision_vec[k] * z_{i*K + k}
    w_current = np.zeros(D)
    for i in range(D):
        for k in range(K):
            w_current[i] += precision_vec[k] * best_z[i*K + k]
    best_r2 = r2_score(X, Y, w_current)
    
    # Keep track of the best solution
    best_w = w_current.copy()
    old_r2 = best_r2
    
    for iteration in range(max_iters):
        # Build QUBO with the current precision
        A, b = build_qubo_matrices(X, Y, precision_vec)
        z_candidate = mock_qubo_solver(A, b, num_samples=num_samples)
        
        # Convert candidate solution to real weights
        w_candidate = np.zeros(D)
        for i in range(D):
            for k in range(K):
                w_candidate[i] += precision_vec[k] * z_candidate[i*K + k]
        
        new_r2 = r2_score(X, Y, w_candidate)
        
        # Decide whether we improved
        if new_r2 > old_r2:
            # We improved the fit, so refine (shrink) the precision step
            # and recenter toward the new solution
            w_current = 0.5 * (w_current + w_candidate)
            # We'll shift each entry in precision_vec slightly
            # so that the sum(precision_vec) is roughly near the new w scale
            # This is just a toy example for demonstration.
            shift_factor = (rate / rate_desc)
            precision_vec *= (1.0 - shift_factor)
            precision_vec += shift_factor * precision_vec.mean()
            
            if new_r2 > best_r2:
                best_r2 = new_r2
                best_z = z_candidate
                best_w = w_candidate.copy()
        else:
            # We got worse, expand precision range
            expansion_factor = (rate * rate_asc)
            precision_vec *= (1.0 + expansion_factor)
            
        old_r2 = new_r2
    
    return best_w, best_r2, precision_vec

###############################################################################
# Putting It All Together
###############################################################################
if __name__ == "__main__":
    # 1) Generate data
    X, Y, w_true = generate_mock_data(N=200, D=4, noise=0.01)
    
    # 2) Initialize a small precision vector (just 3 bits in this example)
    init_precision = np.array([0.5, 0.25, 0.125])
    
    # 3) Run the adaptive procedure
    best_w, best_r2, best_precision = adaptive_precision_linreg(
        X, Y,
        init_precision=init_precision,
        rate=0.1,
        rate_desc=2.0,
        rate_asc=1.5,
        max_iters=5,       # small for demonstration
        num_samples=1000   # small for demonstration
    )

    # Plot the results
    import matplotlib.pyplot as plt
    
    # Plot true vs estimated weights
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(w_true, best_w, alpha=0.6)
    plt.plot([-1, 1], [-1, 1], 'r--')  # diagonal line for reference
    plt.xlabel('True Weights')
    plt.ylabel('Estimated Weights')
    plt.title('True vs Estimated Weights')
    
    # Plot the predictions vs actual values with regression line
    y_pred = X @ best_w
    plt.subplot(1, 2, 2)
    plt.scatter(Y, y_pred, alpha=0.6)
    
    # Add regression line
    z = np.polyfit(Y, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(Y, p(Y), "r-", label='Regression Line')
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("True weights:", w_true)
    print("Estimated weights:", best_w)
    print("Final R^2:", best_r2)
    print("Refined precision vector:", best_precision)
