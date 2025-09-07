import numpy as np
import matplotlib.pyplot as plt

# Function to find eigenvalues of a matrix
def find_eigenvalues(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    return eigenvalues

# Function to generate a 2x2 density matrices from mx, my, mz
def generate_matrix(mx, my, mz):
    return np.array([[1 + mz, mx - 1j * my], [mx + 1j * my, 1 - mz]]) / 2

# Function to check if the matrix is positive semi-definite
def check_postivity(mx, my, mz):
    eg_values = find_eigenvalues(generate_matrix(mx, my, mz))
    return np.all(eg_values >= 0)

# Function to plot percentage of positive matrices among samples drawn
def counter(mx, my, mz, method=2, plot=False):
    counts = 0
    inside_points = []
    outside_points = []

    if method == 3:
        X, Y, Z = np.meshgrid(mx, my, mz, indexing='ij')
        mask = (X**2 + Y**2 + Z**2) <= 1
        counts = np.sum(mask)
        if plot:
            inside_idx = np.where(mask)
            outside_idx = np.where(~mask)
            inside_points = list(zip(X[inside_idx], Y[inside_idx], Z[inside_idx]))
            outside_points = list(zip(X[outside_idx], Y[outside_idx], Z[outside_idx]))       
    else:
        for i in mx:
            for j in my:
                for k in mz:
                    if method == 1:
                        is_pos = bool(check_postivity(i, j, k))
                    else:
                        is_pos = (i**2 + j**2 + k**2 <= 1)
                    counts += int(is_pos)
                    if plot: (inside_points if is_pos else outside_points).append((i, j, k))

    return counts, inside_points, outside_points


# Function to calculate percentage of positive matrices
def percent(mx, my, mz, method=1):
    counts, _, _ = counter(mx, my, mz, method)
    return counts / (len(mx) * len(my) * len(mz)) * 100

# Function to plot percentage of positive matrices vs sample count
def plotter(x, y, figsize=(6, 4)):
    plt.figure(figsize=figsize)   
    plt.plot(x, y)
    plt.xlabel('Graining (Sample count)')
    plt.ylabel('Percentage of Positive Matrices (%)')
    plt.title('Plot of Percentage of Positive Matrices vs Graining')
    plt.grid(True)
    plt.show()

# Function to visualize the bloch sphere with positive and non-positive matrices
def bloch_visualizer(mx, my, mz, method=1, figsize=(8, 6)):
    counts, inside_points, outside_points = counter(mx, my, mz, method, plot=True)
    total = len(mx) * len(my) * len(mz)
    print("Percentage of positive matrices:",
          counts / total * 100)

    fig = plt.figure(figsize=figsize)          
    ax = fig.add_subplot(111, projection='3d')
    
    if inside_points:
        inside_points = np.array(inside_points)
        ax.scatter(
            inside_points[:, 0], inside_points[:, 1], inside_points[:, 2],
            c='b', marker='o', label='Positive Matrices'
        )

    if outside_points:
        outside_points = np.array(outside_points)
        ax.scatter(
            outside_points[:, 0], outside_points[:, 1], outside_points[:, 2],
            c='r', marker='^', alpha=0.1,  # translucent outside points
            label='Non-Positive Matrices'
        )

    ax.legend()
    ax.set_xlabel('mx')
    ax.set_ylabel('my')
    ax.set_zlabel('mz')
    plt.title('The Bloch Sphere')
    plt.show()


