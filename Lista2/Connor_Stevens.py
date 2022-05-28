import numpy as np
import pylab as plt
from scipy.integrate import odeint


class ConnorStevens:
    """Full Connor-Stevens Model implemented in Python.

    Parameters
    ----------
    C_m : float
        Membrane Capacitance, in uF/cm^2.
    g_Na : float
        Sodium (Na) maximum conductances, in mS/cm^2.
    g_K : float
        Postassium (K) maximum conductances, in mS/cm^2.
    g_A : float
        Calcium (Ca) maximum conductances, in mS/cm^2.
    g_L : float
        Leak maximum conductances, in mS/cm^2.
    E_Na : float
        Sodium (Na) Nernst reversal potentials, in mV.
    E_K : float
        Postassium (K) Nernst reversal potentials, in mV.
    E_A : float
        Calcium (Ca) Nernst reversal potentials, in mV.
    E_L : float
        Leak Nernst reversal potentials, in mV.
    exp_time : float
        Total time of the experiment, in ms.
    """

    def __init__(
        self,
        C_m=1.0,
        g_Na=120.0,
        g_K=20.0,
        g_A=47.7,
        g_L=0.3,
        E_Na=55.0,
        E_K=-72.0,
        E_A=-75.0,
        E_L=-17,
        exp_time=450.0,
    ):
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_A = g_A
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_A = E_A
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
        return 0.1 * (V + 29.7) / (1 - (np.exp(-0.1 * (V + 29.7))))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 4.0 * np.exp(-0.0556 * (V + 54.7))

    def tau_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1 / 3.8 * (self.alfa_m(V) + self.beta_m(V))

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.07 * np.exp(-0.05 * (V + 48.0))

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1.0 / (1.0 + np.exp(-0.1 * (V + 18.0)))

    def tau_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1 / 3.8 * (self.alfa_h(V) + self.beta_h(V))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.01 * (V + 45.7) / (1.0 - np.exp(-(V + 45.7) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.125 * np.exp(-0.0125 * (V + 55.7))

    def tau_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 2 / 3.8 * (self.alfa_n(V) + self.beta_n(V))

    def tau_a(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 0.3632 + (1.158 / (1 + np.exp(0.0497 * (V + 55.96))))

    def tau_b(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1.24 + (2.678 / (1 + np.exp(0.0624 * (V + 50.0))))

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

    def I_A(self, V, a, b):
        """Membrane current (in uA/cm^2) Potassium (K = element name).

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        n : float
            Potassium (K) activation gating variable, in dimensionless.
        """
        return self.g_A * a**3 * b * (V - self.E_A)

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

    def steady_A(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return (
            0.0761 * np.exp(0.0314 * (V + 94.22)) / (1 + np.exp(0.0346 * (V + 1.17)))
        ) ** (1 / 3)

    def steady_B(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return (1 / (1 + np.exp(0.0688 * (V + 53.3)))) ** (4)

    def steady_N(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

    def steady_M(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

    def steady_H(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))

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
        V, m, h, n, a, b, use_question1_f = X

        dVdt = (
            self.I_inj(t, J, times)
            - self.I_Na(V, m, h)
            - self.I_K(V, n)
            - self.I_A(V, a, b)
            - self.I_L(V)
        ) / self.C_m

        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        dadt = (self.steady_A(V) - a) / self.tau_a(V)
    
        if use_question1_f:
            dbdt = 0.25 * (self.steady_B(V) - b) / self.tau_b(V)
        else:
            dbdt = (self.steady_B(V) - b) / self.tau_b(V)

        return dVdt, dmdt, dhdt, dndt, dadt, dbdt, use_question1_f

    def run(self, J, times, V0=-65.0, use_question1_f=False, plot=True):
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

        X = odeint(
            self.dALLdt,
            [
                V0,
                self.steady_M(V0),
                self.steady_H(V0),
                self.steady_N(V0),
                self.steady_A(V0),
                self.steady_B(V0),
                use_question1_f,
            ],
            self.t,
            args=(J, times, self),
        )


        V = X[:, 0]
        m = X[:, 1]
        h = X[:, 2]
        n = X[:, 3]
        a = X[:, 4]
        b = X[:, 5]

        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        ia = self.I_A(V, a, b)

        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

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
            plt.plot(self.t, ia, "r", label="$I_{A}$")
            plt.ylabel("Current")
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(self.t, m, "r", label="m")
            plt.plot(self.t, h, "g", label="h")
            plt.plot(self.t, n, "b", label="n")
            plt.plot(self.t, a, "y", label="a")
            plt.plot(self.t, b, "m", label="b")
            plt.ylabel("Gating Value")
            plt.legend()

            plt.subplot(4, 1, 4)

            plt.plot(self.t, i_inj_values, "k")
            plt.xlabel("t (ms)")
            plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")
            plt.ylim(-40, 40)

            plt.show()

        return V, m, h, n, ina, ik, il, ia, i_inj_values


if __name__ == "__main__":
    runner = ConnorStevens(exp_time=100)
    runner.run(J=[0], times=[])
