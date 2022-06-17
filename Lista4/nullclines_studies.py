import numpy as np
import matplotlib.pyplot as plt


def calc_nullclines(I=0):
    """Calculate the nullclines of the FitzHugh-Nagumo model.

    Parameters
    ----------
    I : float
        External current, in mA.
    """

    dvdt_nullcline = lambda v: v - v**3 / 3 + I
    dwdt_nullcline = lambda v: (v + 0.7) / 0.8
    # dwdt_nullcline = lambda v: 0.8 * v - 0.7

    v = np.linspace(-4, 4, 1000)

    dvdt_nullcline = dvdt_nullcline(v)
    dwdt_nullcline = dwdt_nullcline(v)

    return v, dvdt_nullcline, dwdt_nullcline


def plot_nullclines(v, dvdt_nullcline, dwdt_nullcline):
    """Plot nulclines for FitzHugh-Nagumo model.

    Parameters
    ----------
    v : array
        Array of points in v.
    dvdt_nullcline : array
        Array of dvdt nulcline values.
    dwdt_nullcline : array
        Array of dwdt nulcline values.
    """
    plt.plot(v, dvdt_nullcline, label="dvdt", color="red")
    plt.plot(v, dwdt_nullcline, label="dwdt", color="blue")
    plt.xlabel("v")
    plt.ylabel("w")
    plt.legend()
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)
    plt.show()


def find_equilibrium(dvdt_nulcline, dwdt_nulcline):
    """Find the equilibrium point of the FitzHugh-Nagumo model.

    Parameters
    ----------
    dvdt_nulcline : array
        Array of dvdt nulcline values.
    dwdt_nulcline : array
        Array of dwdt nulcline values.
    """
    idx = np.argwhere(np.diff(np.sign(dvdt_nulcline - dwdt_nulcline))).flatten()

    return idx


def plot_equilibrium_nullclines(v, dvdt_nullcline, dwdt_nullcline, idx):
    """Plot equilibrium point in nulclines for FitzHugh-Nagumo model.

    Parameters
    ----------
    v : array
        Array of points in v.
    dvdt_nullcline : array
        Array of dvdt nulcline values.
    dwdt_nullcline : array
        Array of dwdt nulcline values.
    idx : int
        Index of equilibrium point.
    """
    plt.plot(v, dvdt_nullcline, label="dvdt", color="red")
    plt.plot(v, dwdt_nullcline, label="dwdt", color="blue")
    plt.xlabel("v")
    plt.ylabel("w")
    plt.legend()
    plt.plot(v[idx], dvdt_nullcline[idx], "o", color="black")
    plt.plot(v[idx], dwdt_nullcline[idx], "o", color="black")

    v_point = -dvdt_nullcline[idx] / dwdt_nullcline[idx]
    w_point = v[idx]
    plt.text(
        v[idx] + 0.2,
        dvdt_nullcline[idx],
        f"v_eq = ({round(w_point[0], 2)}, {round(v_point[0], 2)})",
    )
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)
    plt.show()


if __name__ == "__main__":
    plot_nullclines()
