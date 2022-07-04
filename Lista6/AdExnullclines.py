import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from AdEx import AdEx

EQUILIBRIUM_COLOR = {
    "Stable node": "C0",
    "Unstable node": "C1",
    "Saddle": "C4",
    "Stable focus": "C3",
    "Unstable focus": "C2",
    "Center": "C5",
}


def AdEx_calc(x, t, G, V_r, delta_L, V_L, a, I):
    """Calculate the nullclines of the FitzHugh-Nagumo model.

    Parameters
    ----------
    I : float
        External current, in mA.
    """
    if isinstance(x[0], float):
        if x[0] > 50e-3:
            return np.array([-80e-3, x[1] + 20e-12])

    return np.array(
        [
            (
                -G * (x[0] - V_r)
                + G * delta_L * np.exp((x[0] - V_L) / delta_L)
                - x[1]
                + I
            )
            / (5e-3 / 500e6),
            a * (x[0] - V_r),
        ]
    )


def plot_vector_field(ax, params, xrange, yrange, steps=50):
    # Compute the vector field
    x = np.linspace(xrange[0], xrange[1], steps)
    y = np.linspace(yrange[0], yrange[1], steps)
    X, Y = np.meshgrid(x, y)

    dx, dy = AdEx_calc([X, Y], 0, **params)

    # streamplot is an alternative to quiver
    # that looks nicer when your vector filed is
    # continuous.
    ax.streamplot(X, Y, dx, dy, color=(0, 0, 0, 0.1))

    ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))


def nullclineV(v, G, V_r, delta_L, V_L, a, I):
    return -G * (v - V_r) + G * delta_L * np.exp((v - V_L) / delta_L) + I


def nullclineU(v, G, V_r, delta_L, V_L, a, I):
    return a * (v - V_r)


def plot_isocline(
    ax, G, V_r, delta_L, V_L, a, I, color="k", style="--", opacity=0.5, vmin=-1, vmax=1
):
    """Plot the null iscolines of the Fitzhugh nagumo system"""
    v = np.linspace(vmin, vmax, 1000)
    ax.plot(
        v,
        nullclineV(v, G, V_r, delta_L, V_L, a, I),
        style,
        color=color,
        alpha=opacity,
    )
    ax.plot(
        v, nullclineU(v, G, V_r, delta_L, V_L, a, I), style, color=color, alpha=opacity
    )

    return v


def find_roots(params):
    # We store the position of the equilibrium.
    roots, infos, _, _ = fsolve(
        nullclineV, 0, args=tuple(params.values()), full_output=True
    )

    roots_u, infos_u, _, _ = fsolve(
        nullclineV, 0, args=tuple(params.values()), full_output=True
    )
    return np.array([[r, infos["fvec"][0], u] for r, u in zip(roots, roots_u)])


def stability(jacobian):
    """Stability of the equilibrium given its associated 2x2 jacobian matrix.
    Use the eigenvalues.
    Args:
        jacobian (np.array 2x2): the jacobian matrix at the equilibrium point.
    Return:
        (string) status of equilibrium point.
    """
    eigv = np.linalg.eigvals(jacobian)

    if all(np.real(eigv) == 0) and all(np.imag(eigv) != 0):
        nature = "Center"
    elif np.real(eigv)[0] * np.real(eigv)[1] < 0:
        nature = "Saddle"
    else:
        stability = "Unstable" if all(np.real(eigv) > 0) else "Stable"
        nature = stability + (" focus" if all(np.imag(eigv) != 0) else " node")
    return nature


def jacobian_AdEx(v, G, V_r, delta_L, V_L, a, I):
    return np.array([[G * np.exp((-V_L + v) / delta_L) - G, 0], [a, 0]])


def plot_phase_diagram(params, ax=None, title=None):
    """Plot a complete Fitzhugh-Nagumo phase Diagram in ax.
    Including isoclines, flow vector field, equilibria and their stability"""
    if ax is None:
        ax = plt.gca()

    ax.set(xlabel="v", ylabel="u", title=title)

    # Isocline and flow...
    xlimit = (-0.1, 0.01)
    ylimit = (-0.6, 0.9)
    plot_vector_field(ax, params, xlimit, ylimit)

    plot_isocline(ax, **params, vmin=xlimit[0], vmax=xlimit[1])

    eqnproot = find_roots(params)
    eqstability = [stability(jacobian_AdEx(v=e[0], **params)) for e in eqnproot]

    for e, n in zip(eqnproot, eqstability):
        ax.scatter(e[0], e[1], color=EQUILIBRIUM_COLOR[n])

        adex = AdEx(
            exp_time=1,
            C=5e-3 / 500e6,
            G=1 / 500e6,
            V_pico=20e-3,
            a=-0.5e-9,
            b=7e-12,
            tau_u=100e-3,
        )
        traj = adex.run(J=[65e-12], times=[(0, 2)], plot=False)
        ax.plot(traj[0], traj[1], color="k")

    # Legend
    labels = frozenset(eqstability)
    ax.legend(
        [mpatches.Patch(color=EQUILIBRIUM_COLOR[n]) for n in labels],
        labels,
        loc="lower right",
    )
    plt.show()


def find_jacobian():
    # Symbolic computation of the Jacobian using sympy...
    import sympy as sp

    sp.init_printing()

    # Define variable as symbols for sp
    v, u = sp.symbols("v, u")
    G, a, V_r, V_L, delta_l, I = sp.symbols("G, a, V_r, V_L, delta_l, I")

    # Symbolic expression of the system
    dvdt = -G * (v - V_r) + G * delta_l * sp.exp((v - V_L) / delta_l) + I
    dudt = a * (v - V_r)

    # Symbolic expression of the matrix
    sys = sp.Matrix([dvdt, dudt])
    var = sp.Matrix([v, u])
    jac = sys.jacobian(var)

    # jac to a function:
    jacobian_AdEx_symbolic = sp.lambdify(
        (v, u, G, a, V_r, V_L, delta_l, I), jac, dummify=False
    )
    return jac


if __name__ == "__main__":
    params = {
        "G": 1 / 500e6,
        "V_r": -0.07,
        "delta_L": 0.002,
        "V_L": -0.05,
        "a": -0.5e-9,
        "I": 221e-12,
    }

    plot_phase_diagram(params)
