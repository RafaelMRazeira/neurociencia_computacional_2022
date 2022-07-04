import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Izhikevich_mod import Izhikevich

EQUILIBRIUM_COLOR = {
    "Stable node": "C0",
    "Unstable node": "C1",
    "Saddle": "C4",
    "Stable focus": "C3",
    "Unstable focus": "C2",
    "Center": "C5",
}


class IzhikevichNullclines:
    def __init__(self, params, C=150e-3, c=-56, V_pico=50):
        self.params = params
        self.C = C
        self.c = c
        self.V_pico = V_pico

    def Izhi_mod_calc(self, x, t, k, a, b, p, V_r, V_L, I):
        """Calculate the nullclines of the FitzHugh-Nagumo model.

        Parameters
        ----------
        I : float
            External current, in mA.
        """
        return np.array(
            [
                (k * (x[0] - V_r) * (x[0] - V_L) - x[1] + I) / self.C,
                a * (b * (x[0] - V_r) + p * (x[0] - V_r) ** 3 - x[1]),
            ]
        )

    def plot_vector_field(self, ax, params, xrange, yrange, steps=50):
        # Compute the vector field
        x = np.linspace(xrange[0], xrange[1], steps)
        y = np.linspace(yrange[0], yrange[1], steps)
        X, Y = np.meshgrid(x, y)

        dx, dy = self.Izhi_mod_calc([X, Y], 0, **params)
        # streamplot is an alternative to quiver
        # that looks nicer when your vector filed is
        # continuous.
        ax.streamplot(X, Y, dx, dy, color=(0, 0, 0, 0.1))

        ax.set(xlim=(xrange[0], xrange[1]), ylim=(yrange[0], yrange[1]))

    def nullclineV(self, v, k, a, b, p, V_r, V_L, I):
        return (k * (v - V_r) * (v - V_L)) + I

    def nullclineU(self, v, k, a, b, p, V_r, V_L, I):
        return b * (v - V_r) + p * (v - V_r) ** 3

    def plot_isocline(
        self,
        ax,
        k,
        a,
        b,
        p,
        V_r,
        V_L,
        I,
        color="k",
        style="--",
        opacity=0.5,
        vmin=-1,
        vmax=1,
    ):
        """Plot the null iscolines of the Fitzhugh nagumo system"""
        v = np.linspace(vmin, vmax, 1000)
        ax.plot(
            v,
            self.nullclineV(v, k, a, b, p, V_r, V_L, I),
            style,
            color=color,
            alpha=opacity,
        )
        ax.plot(
            v,
            self.nullclineU(v, k, a, b, p, V_r, V_L, I),
            style,
            color=color,
            alpha=opacity,
        )

        return v

    def find_roots(self, params):
        # We store the position of the equilibrium.
        roots, infos, _, _ = fsolve(
            self.nullclineV, 0, args=tuple(params.values()), full_output=True
        )
        return np.array([[r, infos["fvec"][0]] for r in roots])

    def stability(self, jacobian):
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

    def jacobian_AdEx(self, v, k, a, b, p, V_r, V_L, I):
        return np.array(
            [
                [k * (v - V_L) + k * (v - V_r), -1],
                [a * (b + (3 * p * (v + V_r) ** 2)), -a],
            ]
        )

    def plot_phase_diagram_Izhi(self, params, ax=None, title=None):
        """Plot a complete Fitzhugh-Nagumo phase Diagram in ax.
        Including isoclines, flow vector field, equilibria and their stability"""
        if ax is None:
            ax = plt.gca()
        if title is None:
            title = "Phase space, {}".format(params)

        ax.set(xlabel="v", ylabel="u", title=title)

        # Isocline and flow...
        xlimit = (-800, 800)
        ylimit = (-800, 800)
        self.plot_vector_field(ax, params, xlimit, ylimit)

        self.plot_isocline(ax, **params, vmin=xlimit[0], vmax=xlimit[1])

        eqnproot = self.find_roots(params)
        eqstability = [
            self.stability(self.jacobian_AdEx(v=e[0], **params)) for e in eqnproot
        ]

        for e, n in zip(eqnproot, eqstability):
            ax.scatter(*e, color=EQUILIBRIUM_COLOR[n])

            izhikevich = Izhikevich(
                C=self.C,
                V_r=params["V_r"],
                V_L=params["V_L"],
                k=params["k"],
                a=params["a"],
                b=params["b"],
                c=self.c,
                V_pico=self.V_pico,
                exp_time=0.5,
            )
            traj = izhikevich.run(J=[params["I"]], times=[(0, 0.5)], plot=False)
            ax.plot(traj[0], traj[1], color="k")

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
        k, a, V_r, V_L, b, p, I = sp.symbols("k, a, V_r, V_L, b, p, I")

        # Symbolic expression of the system
        dvdt = (k * (v - V_r) * (v - V_L)) - u + I
        dudt = a * (b * (v - V_r) + p * (v - V_r) ** 3 - u)

        # Symbolic expression of the matrix
        sys = sp.Matrix([dvdt, dudt])
        var = sp.Matrix([v, u])
        jac = sys.jacobian(var)

        # jac to a function:
        jacobian_AdEx_symbolic = sp.lambdify(
            (v, u, k, a, V_r, V_L, b, I), jac, dummify=False
        )
        return jac


if __name__ == "__main__":
    params = {"k": 1, "a": 0.5, "b": 25, "p": 0.009, "V_r": -50, "V_L": -30, "I": 0}
    Izhinull = IzhikevichNullclines(params, C=25, c=-40, V_pico=10)
    Izhinull.plot_phase_diagram_Izhi(params)
