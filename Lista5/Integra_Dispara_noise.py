import numpy as np
import pylab as plt
from tqdm import tqdm


class IntegraDisparaNoise:
    """Full Integra-Dispara Model implemented in Python.

    Parameters
    ----------
    C : float
        Membrane Capacitance, in uF/cm^2.
    V_rep : float
        Repolarization Potential, in mV.
    R : float
        Membrane Resistance, in M/ohms.
    V_L : float
        Voltage Leak, mV.
    V_redef : float
        Voltage redefinition, in mV.
    exp_time : float
        Total time of the experiment, in ms.
    """

    def __init__(
        self,
        C=250e-6,
        V_rep=-70.0,
        R=40e3,
        V_L=-50,
        V_redef=-65,
        sigma=1,
        dt=1e-3,
        exp_time=2.0,
    ):
        self.C = C
        self.V_rep = V_rep
        self.R = R
        self.V_L = V_L
        self.V_redef = V_redef
        self.sigma = sigma
        self.tau = C * R
        self.dt = dt

        """ The time to integrate over """
        self.t = np.arange(0.0, exp_time, dt)

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
    def dALLdt(self, V, i_j):
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
            ((self.V_rep - V) / (self.R * self.C))
            + (i_j / self.C)
            + self.omega * self.omega_t(t)
        )

        return dVdt

    def f_v(self, V, i_j):
        """Calculate the activation variable f."""
        return ((self.V_rep - V) / self.tau) + (i_j / self.C)

    def euler_maruyama(self, V, i_inj_values, spikes):
        """Calculate numericaly euler-maruyama ODE's of fourth grade and step dt.

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

        W_n = np.sqrt(self.dt) * np.random.randn(points_total)

        for t, i_j in enumerate(tqdm(i_inj_values)):
            V += (self.f_v(V, i_j) * self.dt) + self.sigma * W_n[t]

            V_out[t] = V

            if V >= self.V_L:
                V = self.V_redef
                if spikes:
                    V_out[t] = 0.0

        return V_out

    def run(self, J, times, spikes=False, plot=True):
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

        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

        V = self.euler_maruyama(-70, i_inj_values, spikes)

        if plot:
            plt.figure(figsize=(16, 10))

            plt.subplot(2, 1, 1)
            plt.title("Hodgkin-Huxley Neuron")
            plt.plot(self.t, V, "k")
            plt.ylabel("V (mV)")

            plt.subplot(2, 1, 2)

            plt.plot(self.t, i_inj_values, "k")
            plt.xlabel("t (ms)")
            plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")

            plt.show()

        return V, i_inj_values


if __name__ == "__main__":
    runner = IntegraDisparaNoise(exp_time=2, sigma=70)
    runner.run(J=[0.01], times=[(0, 200)], spikes=True)
