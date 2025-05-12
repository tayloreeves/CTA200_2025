import numpy as np

def ass3_func(xmin, xmax, ymin, ymax, width, height, max_iter):
    
    '''
    This function takes in the values given from the code, and returns an array of the divergence of the function.

    Inputs
    xmin, ymin: These are the minimum x and y values for the complex plane's range
    xmax, ymax: These are the maximum x and y values for the complex plane's range
    width, height: These represent how small the values can be for each point in the plane
    max_iter: This is the maximum number of iterations allowed for each data point

    Outputs:
    array: Creates an array of data that can be charted with pyplot
    '''
    
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    Z = np.zeros(C.shape, dtype=complex)
    array = np.zeros(C.shape, dtype=int)

    mask = np.ones(C.shape, dtype=bool)
    #masked points refers to those that are bounded
    
    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        mask_now = np.abs(Z) <= 2
        array += mask & ~mask_now
        mask = mask_now

    return array


def lorenz(t, W, sigma=10.0, r=28.0, b=8.0/3.0):
    
    """
    Computes the derivatives of the Lorenz system.

    Parameters:
        t (float): Time (not used explicitly in the equations).
        W (array): State vector [X, Y, Z].
        sigma (float): Prandtl number.
        r (float): Rayleigh number.
        b (float): Geometry-related parameter.

    Returns:
        list: Derivatives [dX/dt, dY/dt, dZ/dt].
    """
    
    X, Y, Z = W
    dX = -sigma * (X - Y)
    dY = r * X - Y - X * Z
    dZ = -b * Z + X * Y
    return [dX, dY, dZ]
    