import numpy as np
import pylab as plt
from scipy.integrate import odeint


class MorrisLecarMod:
    """Full Morris-Lecar Model implemented in Python.

    Parameters
    ----------
    C_m : float
        Membrane Capacitance, in uF/cm^2.
    g_Ca : float
        Calcium (Ca) maximum conductances, in mS/cm^2.
    g_K : float
        Postassium (K) maximum conductances, in mS/cm^2.
    g_L : float
        Leak maximum conductances, in mS/cm^2.
    E_Ca : float
        Calcium (Ca) Nernst reversal potentials, in mV.
    E_K : float
        Postassium (K) Nernst reversal potentials, in mV.
    E_L : float
        Leak Nernst reversal potentials, in mV.
    exp_time : float
        Total time of the experiment, in ms.
    """

    def __init__(
        self,
        C_m=1.0,
        g_Ca=1.0,
        g_K=2.0,
        g_L=0.5,
        E_Ca=100.0,
        E_K=-70.0,
        E_L=-50,
        exp_time=450.0,
    ):
        self.C_m = C_m
        self.g_Ca = g_Ca
        self.g_K = g_K
        self.g_L = g_L
        self.E_Ca = E_Ca
        self.E_K = E_K
        self.E_L = E_L

        """ The time to integrate over """
        self.t = np.arange(0.0, exp_time, 0.01)

    def m_inf(self, V):
        """Calcium (Ca) activation gating variable.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """

        return 0.5 * (1 + np.tanh((V + 1) / 15))

    def n_inf(self, V):
        """Potassium (K) activation gating variable.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.5 * (1 + np.tanh((V -10) / 14.5))

    def tau_n(self, V):
        """Potassium (K) time constant.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 3 / (np.cosh((V- 10) / 29))

    def I_Ca(self, V):
        """Membrane current (in uA/cm^2) Calcium (Ca = element name).

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return self.g_Ca * self.m_inf(V) * (V - self.E_Ca)

    def I_K(self, V, n):
        """Membrane current (in uA/cm^2) Potassium (K = element name).

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        n : float
            Potassium (K) activation gating variable, in dimensionless.
        """
        return self.g_K * n * (V - self.E_K)

    def I_L(self, V):
        """Membrane current (in uA/cm^2) Leak.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        h : float
            Sodium (Na) inactivation gating variable, in dimensionless.
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t, J, times):
        """External Current.

        Calculates :
            step up to J uA/cm^2 at t>ti
            step down to 0 uA/cm^2 at t>tf

        Parameters
        ----------
        t : float
            Time, in ms.
        J : float
            Current amplitude, in uA/cm^2.
        times : list[tuple]
            List of time_initials and finals for pulses.
        """
        for Ji, (ti, tf) in zip(J, times):
            if t > ti and t < tf:
                return Ji * (t > ti) - Ji * (t > tf)
        return 0

    @staticmethod
    def dALLdt(X, t, J, times, self):
        """Integrate. Calculates membrane potential & activation variables.

        Parameters
        ----------
        X : array
            State variables [V m h n], in mV, dimensionless, and dimensionless, respectively.
        t : float
            Time, in ms.
        times : list[tuple]
            List of time_initials and finals for pulses.
        """
        V, n = X

        dVdt = (
            -self.I_Ca(V) - self.I_K(V, n) - self.I_L(V) + self.I_inj(t, J, times)
        ) / self.C_m

        dndt = (self.n_inf(V) - n) / self.tau_n(V)

        return dVdt, dndt

    def run(self, J, times, V0=-65, plot=True):
        """Run demo for the Hodgkin Huxley neuron model.

        Parameters
        ----------
        J : float
            Current amplitude, in uA/cm^2.
        times : list[tuple]
            List of time_initials and finals for pulses.
        plot : bool
            If True, plots the results.
        """
        X = odeint(self.dALLdt, [V0, 0.69], self.t, args=(J, times, self))
        
        V = X[:, 0]
        n = X[:, 1]

        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

        if plot:
            plt.figure(figsize=(16, 10))

            plt.subplot(3, 1, 1)
            plt.title("Hodgkin-Huxley Neuron")
            plt.plot(self.t, V, "k")
            plt.ylabel("V (mV)")

            plt.subplot(3, 1, 2)
            plt.plot(self.t, n, "b", label="n")
            plt.ylabel("Gating Value")
            plt.legend()

            plt.subplot(3, 1, 3)

            plt.plot(self.t, i_inj_values, "k")
            plt.xlabel("t (ms)")
            plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")
            plt.ylim(-40, 40)

            plt.show()

        return V, n, i_inj_values

class MorrisLecarModNullclines():
    """Morris-Lecar neuron model with nullclines.
    """

    def calc_nullclines(self, I=0):
        """Calculate the nullclines of the FitzHugh-Nagumo model.

        Parameters
        ----------
        I : float
            External current, in mA.
        """

        dvdt_nullcline = lambda v: (-0.55 * (1 + np.tanh((v + 1) / 15)) * (v - 100) + 0.5 * (v + 50) + I) / (2 * (v + 70))
        dndt_nullcline = lambda v: 0.5 * (1 + np.tanh((v - 10)/14.5))

        v = np.linspace(-40, 40, 100)

        dvdt_nullcline = dvdt_nullcline(v)
        dndt_nullcline = dndt_nullcline(v)

        return v, dvdt_nullcline, dndt_nullcline

    def plot_equilibrium_nullclines(self, v, dvdt_nullcline, dndt_nullcline, idx):
        """Plot equilibrium point in nulclines for FitzHugh-Nagumo model.

        Parameters
        ----------
        v : array
            Array of points in v.
        dvdt_nullcline : array
            Array of dvdt nulcline values.
        dndt_nullcline : array
            Array of dwdt nulcline values.
        idx : int
            Index of equilibrium point.
        """
        idx = idx+1
        plt.plot(v, dvdt_nullcline, label="dvdt", color="red")
        plt.plot(v, dndt_nullcline, label="dndt", color="blue")
        plt.xlabel("v")
        plt.ylabel("n")
        plt.legend()
        plt.plot(v[idx], dvdt_nullcline[idx], "o", color="black")
        plt.plot(v[idx], dndt_nullcline[idx], "o", color="black")

        v_point = dvdt_nullcline[idx]
        w_point = v[idx]
        plt.text(
            v[idx],
            dvdt_nullcline[idx],
            f"v_eq = ({round(w_point[0], 2)}, {round(v_point[0], 2)})",
        )
        plt.show()

if __name__ == "__main__":
    runner = MorrisLecarMod(exp_time=100)
    runner.run(J=[15], times=[(0, 100)])
