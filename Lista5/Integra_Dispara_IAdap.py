import numpy as np
import pylab as plt
from tqdm import tqdm


class IntegraDisparaIAdap:
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
        C=250e-12,
        V_rep=-0.070,
        R=40e6,
        V_L=-0.050,
        V_redef=-0.080,
        tau_a=200e-3,
        Ga_init=0.0,
        b=1e-9,
        dt=1e-3,
        exp_time=2.0,
    ):
        self.C = C
        self.V_rep = V_rep
        self.R = R
        self.V_L = V_L
        self.V_redef = V_redef
        self.tau_a = tau_a
        self.Ga_init = Ga_init
        self.b = b
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
    def dALLdt(self, V, Ga, i_j):
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
            + (Ga * (self.V_redef - V)) / self.C
            + i_j / self.C
        )
        dGadt = -Ga / self.tau_a

        return np.array([dVdt, dGadt])

    def runge_kutta(self, V, Ga, i_inj_values):
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
        Ga_out = np.ones(points_total)

        for t, i_j in enumerate(tqdm(i_inj_values)):
            k1 = self.dt * self.dALLdt(self, V, Ga, i_j)
            k2 = self.dt * self.dALLdt(
                self,
                V + 0.5 * k1[0],
                Ga + 0.5 * k1[1],
                i_j,
            )
            k3 = self.dt * self.dALLdt(
                self,
                V + 0.5 * k2[0],
                Ga + 0.5 * k2[1],
                i_j,
            )
            k4 = self.dt * self.dALLdt(self, V + k3[0], Ga + k3[1], i_j)

            V = V + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            Ga = Ga + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6

            V_out[t] = V
            Ga_out[t] = Ga

            if V >= self.V_L:
                V = self.V_redef
                Ga = self.Ga_init + self.b

        return V_out, Ga_out

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

        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

        X = self.runge_kutta(-0.070, self.Ga_init, i_inj_values)

        V = X[0]
        Ga = X[1]

        if plot:
            plt.figure(figsize=(16, 10))

            plt.subplot(3, 1, 1)
            plt.title("Hodgkin-Huxley Neuron")
            plt.plot(self.t, V, "k")
            plt.ylabel("V (mV)")

            plt.subplot(3, 1, 2)
            plt.plot(self.t, Ga, "r")
            plt.ylabel("Ga")

            plt.subplot(3, 1, 3)
            plt.plot(self.t, i_inj_values, "k")
            plt.xlabel("t (ms)")
            plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")

            plt.show()

        return V, Ga, i_inj_values


if __name__ == "__main__":
    runner = IntegraDisparaIAdap(exp_time=2)
    runner.run(J=[501e-12], times=[(0, 2)])
