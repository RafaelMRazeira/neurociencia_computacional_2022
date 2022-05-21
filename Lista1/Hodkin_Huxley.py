import numpy as np
import pylab as plt
from scipy.integrate import odeint


class HodgkinHuxley:
    """Full Hodgkin-Huxley Model implemented in Python.

    Parameters
    ----------
    C_m : float
        Membrane Capacitance, in uF/cm^2.
    g_Na : float
        Sodium (Na) maximum conductances, in mS/cm^2.
    g_K : float
        Postassium (K) maximum conductances, in mS/cm^2.
    g_L : float
        Leak maximum conductances, in mS/cm^2.
    E_Na : float
        Sodium (Na) Nernst reversal potentials, in mV.
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
        g_Na=120.0,
        g_K=36.0,
        g_L=0.3,
        E_Na=50.0,
        E_K=-77.0,
        E_L=-54.387,
        exp_time=450.0,
    ):
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L

        """ The time to integrate over """
        self.t = np.arange(0.0, exp_time, 0.01)

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.125 * np.exp(-(V + 65) / 80.0)

    def I_Na(self, V, m, h):
        """Membrane current (in uA/cm^2) Sodium (Na = element name).

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        m : float
            Sodium (Na) activation gating variable, in dimensionless.
        h : float
            Sodium (Na) inactivation gating variable, in dimensionless.
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """Membrane current (in uA/cm^2) Potassium (K = element name).

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        n : float
            Potassium (K) activation gating variable, in dimensionless.
        """
        return self.g_K * n**4 * (V - self.E_K)

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
        V, m, h, n = X

        dVdt = (
            self.I_inj(t, J, times) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)
        ) / self.C_m
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        return dVdt, dmdt, dhdt, dndt

    def run(self, J, times, plot=True):
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
        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(J, times, self))
        V = X[:, 0]
        m = X[:, 1]
        h = X[:, 2]
        n = X[:, 3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

        V += 65.0

        if plot:
            plt.figure(figsize=(16, 10))

            plt.subplot(4, 1, 1)
            plt.title("Hodgkin-Huxley Neuron")
            plt.plot(self.t, V, "k")
            plt.ylabel("V (mV)")

            plt.subplot(4, 1, 2)
            plt.plot(self.t, ina, "c", label="$I_{Na}$")
            plt.plot(self.t, ik, "y", label="$I_{K}$")
            plt.plot(self.t, il, "m", label="$I_{L}$")
            plt.ylabel("Current")
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(self.t, m, "r", label="m")
            plt.plot(self.t, h, "g", label="h")
            plt.plot(self.t, n, "b", label="n")
            plt.ylabel("Gating Value")
            plt.legend()

            plt.subplot(4, 1, 4)

            plt.plot(self.t, i_inj_values, "k")
            plt.xlabel("t (ms)")
            plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")
            plt.ylim(-40, 40)

            plt.show()

        return V, m, h, n, ina, ik, il, i_inj_values


if __name__ == "__main__":
    runner = HodgkinHuxley(exp_time=100)
    runner.run(J=[35, 45, 5], times=[(50, 100), (10, 20), (0, 5)])
