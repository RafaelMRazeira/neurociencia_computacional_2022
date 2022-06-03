import numpy as np
import pylab as plt
from tqdm import tqdm


class ReleTalamico:
    """Full Connor-Stevens Model implemented in Python.

    Parameters
    ----------
    C_m : float
        Membrane Capacitance, in uF/cm^2.
    g_Na : float
        Sodium (Na) maximum conductances, in mS/cm^2.
    g_K : float
        Postassium (K) maximum conductances, in mS/cm^2.
    g_CaT : float
        Calcium (Ca) maximum conductances, in mS/cm^2.
    g_L : float
        Leak maximum conductances, in mS/cm^2.
    E_Na : float
        Sodium (Na) Nernst reversal potentials, in mV.
    E_K : float
        Postassium (K) Nernst reversal potentials, in mV.
    E_Ca : float
        Calcium (Ca) Nernst reversal potentials, in mV.
    E_L : float
        Leak Nernst reversal potentials, in mV.
    exp_time : float
        Total time of the experiment, in ms.
    """

    def __init__(
        self,
        C_m=0.1e-9,
        g_L=10e-9,
        g_Na=3.6e-6,
        g_K=1.6e-6,
        g_CaT=0.22e-6,
        E_L=-0.070,
        E_Na=55e-3,
        E_K=-0.090,
        E_Ca=0.120,
        exp_time=450.0,
    ):
        self.C_m = C_m
        self.g_L = g_L
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_CaT = g_CaT
        self.E_L = E_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_Ca = E_Ca

        """ The time to integrate over """
        self.t = np.arange(0.0, exp_time, 0.01)

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1e5 * (V + 0.035) / (1 - np.exp(-100 * (V + 0.035)))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 4000 * np.exp(-(V + 0.06) / 0.018)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 350 * np.exp(-50 * (V + 0.058))

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 5000 / (1 + np.exp(-100 * (V + 0.028)))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 5e4 * (V + 0.034) / (1 - np.exp(-100 * (V + 0.034)))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 625 * np.exp(-12.5 * (V + 0.044))

    def tau_h_T(self, V):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        if V < -0.080:
            return 0.001 * np.exp(15 * (V + 0.467))

        return 0.028 + 0.001 * np.exp(-(V + 0.022) / 0.0105)

    def mca_inf(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1.0 / (1.0 + np.exp(-(V + 0.052) / 0.0074))

    def m_inf(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

    def h_inf(self, V):
        """Steady state activation gating variable H.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))

    def n_inf(self, V):
        """Steady state activation gating variable H.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

    def h_T_inf(self, V):
        """Steady state activation gating variable A.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 1.0 / (1.0 + np.exp(500 * (V + 0.076)))

    def I_Na(self, V, h):
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
        m = self.m_inf(V)

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

    def I_Ca(self, V, h_T):
        """Membrane current (in uA/cm^2) Calcium (Ca = element name).

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        n : float
            Calcium (Ca) activation gating variable, in dimensionless.
        """
        mca = self.mca_inf(V)
        return self.g_CaT * mca**2 * h_T * (V - self.E_Ca)

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
            if t >= ti and t <= tf:
                return Ji * (t >= ti) - Ji * (t >= tf)
        return 0

    @staticmethod
    def dALLdt(V, n, h, h_T, I_inj, self):
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
        dVdt = (
            I_inj - self.I_Na(V, h) - self.I_K(V, n) - self.I_Ca(V, h_T) - self.I_L(V)
        ) / self.C_m

        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dhtdt = (self.h_T_inf(V) - h_T) / self.tau_h_T(V)

        return np.array([dVdt, dndt, dhdt, dhtdt])

    def runge_kutta(self, X, i_inj_values, dt=0.01e-3):
        """Calculate numericaly Runge-Kutta ODE's of fourth grade and step dt.

        Parameters
        ----------
        X : array
            State variables [V m h n], in mV, dimensionless, and dimensionless, respectively.
        i_inj_values : array
            Current injection values, in uA/cm^2.
        dt : float
            Time step, in ms.
        """
        points_total = len(i_inj_values)
        V_out = np.ones(points_total)
        h_out = np.ones(points_total)
        n_out = np.ones(points_total)
        h_T_out = np.ones(points_total)

        V, n, h, h_T = X

        for t, i_j in enumerate(tqdm(i_inj_values)):
            k1 = dt * self.dALLdt(V, n, h, h_T, i_j, self)
            k2 = dt * self.dALLdt(
                V + 0.5 * k1[0],
                n + 0.5 * k1[1],
                h + 0.5 * k1[2],
                h_T + 0.5 * k1[3],
                i_j,
                self,
            )
            k3 = dt * self.dALLdt(
                V + 0.5 * k2[0],
                n + 0.5 * k2[1],
                h + 0.5 * k2[2],
                h_T + 0.5 * k2[3],
                i_j,
                self,
            )
            k4 = dt * self.dALLdt(
                V + k3[0], n + k3[1], h + k3[2], h_T + k3[3], i_j, self
            )

            V = V + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            V_out[t] = V

            n = n + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            n_out[t] = n

            h = h + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
            h_out[t] = h

            h_T = h_T + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
            h_T_out[t] = h_T

        return V_out, h_out, n_out, h_T_out

    def run(self, J, times, V0=-70e-3, plot=True):
        """Run demo for the Rele Thalamic neuron model.

        Parameters
        ----------
        J : float
            Current amplitude, in uA/cm^2.
        times : list[tuple]
            List of time_initials and finals for pulses.
        plot : bool
            If True, plots the results.
        """
        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

        X = self.runge_kutta(
            [
                V0,
                self.n_inf(V0),
                self.h_inf(V0),
                0,
            ],
            i_inj_values,
        )

        V = X[0]
        n = X[1]
        h = X[2]
        h_T = X[3]

        ina = self.I_Na(V, h)
        ik = self.I_K(V, n)
        ica = self.I_Ca(V, h_T)
        il = self.I_L(V)

        if plot:
            plt.figure(figsize=(16, 10))

            plt.subplot(4, 1, 1)
            plt.title("Rele-Thalamic Neuron")
            plt.plot(self.t, V, "k")
            plt.ylabel("V (mV)")

            plt.subplot(4, 1, 2)
            plt.plot(self.t, ina, "c", label="$I_{Na}$")
            plt.plot(self.t, ik, "y", label="$I_{K}$")
            plt.plot(self.t, il, "m", label="$I_{L}$")
            plt.plot(self.t, ica, "r", label="$I_{Ca}$")
            plt.ylabel("Current")
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(self.t, h, "r", label="h")
            plt.plot(self.t, n, "g", label="n")
            plt.plot(self.t, h_T, "b", label="h_T")
            plt.ylabel("Gating Value")
            plt.legend()

            plt.subplot(4, 1, 4)

            plt.plot(self.t, i_inj_values, "k")
            plt.xlabel("t (ms)")
            plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")

            plt.show()

        return V, n, h, h_T, ina, ik, ica, il, i_inj_values


if __name__ == "__main__":
    runner = ReleTalamico(exp_time=250)
    runner.run(J=[-200e-12], times=[(0, 250)])
