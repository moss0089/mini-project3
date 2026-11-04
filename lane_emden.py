"""
PH306 | Mini Project 3 - ODEs
Functions to find numerical solutions of the Lane-Emden equation for a given 
polytropic index. 
"""

# --- Imports --- #
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Numerical Solution --- #
def LE_sol(index: float, squig: tuple[float]=(0, 10), sample_rate: int=50, tha0: int=1, phi0: int=0):
    """ Finds numerical solution of Lane-Emden equation for a given index.

    Parameters
    ----------
    index : float
        Polytropic index.
    
    Returns
    -------

    Raises
    ------
    """

    def lane_emden(t, y, n):
        theta, phi = y
        dtha_dsqg = -phi / t**2
        dphi_dsqg = (np.float_power(theta, n, dtype=np.complex128).real)*(t**2)
        return dtha_dsqg, dphi_dsqg

    def boundary(t, y, n):
        return y[0]

    sq0, sqF = squig

    if sq0 < 0:
        raise ValueError('Lower bound cannot be less than zero.')
    elif sq0 == 0:
        sq0 += 0.001

    t_span = (sq0, sqF)
    t_eval = np.linspace(sq0, sqF, sample_rate)
    sol = solve_ivp(lane_emden, t_span, (tha0, phi0), t_eval=t_eval, args=(index,), events=boundary)
    return sol

# --- Plot Numerical Solutions --- #
def LE_plot(index, squig=(0, 10), sample_rate=50, tha0=1, phi0=0):
    sq0, sqF = squig
    num = len(index)
    cmap = plt.get_cmap('viridis', num)

    fig, ax = plt.subplots()

    for n in range(0, num):
        sol = LE_sol(index[n], squig, sample_rate, tha0, phi0)
        label = (f'$n={index[n]}$')
        color_m = cmap(n)
        ax.plot(sol.t, sol.y[0], label=label, color=color_m)

    ax.set_title('$\\text{Solutions of the Lane-Emden Equation}$')
    ax.set_xlabel('$\\zeta, \\text{radius}$')
    ax.set_ylabel('$\\theta, \\text{density proxy}$')
    plt.ylim(-0.4, 1.2)
    ax.yaxis.grid(True)
    ax.legend(fontsize='x-small')

    plt.show
    return fig