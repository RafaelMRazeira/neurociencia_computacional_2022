import numpy as np
import pylab as plt
from tqdm import tqdm


class Izhikevish:
    """Full Izhikevish Model implemented in Python.

    Parameters
    ----------
    C : float
        Membrane Capacitance, in uF/cm^2.
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
        C=150,
        V_r=-75,
        V_L=-45,
        k=1.2,
        a=0.01,
        b=5,
        c=-56,
        d=130,
        V_pico=50,
        dt=1e-3,
        exp_time=2.0,
    ):
        self.C = C
        self.V_r = V_r
        self.V_L = V_L
        self.k = k
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V_pico = V_pico
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
    def dALLdt(self, V, u, i_j):
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
        dVdt = (self.k * (V - self.V_r) * (V - self.V_L)) - u + i_j
        dVdt /= self.C
        
        dUdt = self.a * (self.b * (V - self.V_r)) - u

        return np.array([dVdt, dUdt])

    def runge_kutta(self, V, u, i_inj_values):
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
        U_out = np.ones(points_total)
        v_spikes = np.zeros(points_total)

        for t, i_j in enumerate(tqdm(i_inj_values)):
            k1 = self.dt * self.dALLdt(self, V, u, i_j)
            k2 = self.dt * self.dALLdt(
                self,
                V + 0.5 * k1[0],
                u + 0.5 * k1[1],
                i_j,
            )
            k3 = self.dt * self.dALLdt(
                self,
                V + 0.5 * k2[0],
                u + 0.5 * k2[1],
                i_j,
            )
            k4 = self.dt * self.dALLdt(self, V + k3[0], u + k3[1], i_j)

            V = V + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            u = u + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6

            if V >= self.V_pico:
                V = self.c
                u = u + self.d
                v_spikes[t] = 1

            V_out[t] = V
            U_out[t] = u

        return V_out, U_out, v_spikes

    def run(self, J, times, V_init=-75, u_init=0, plot=True):
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

        X = self.runge_kutta(V_init, u_init, i_inj_values)
        V = X[0]
        u = X[1]
        spikes = X[2]

        if plot:
            plt.figure(figsize=(16, 10))

            plt.subplot(3, 1, 1)
            plt.title("Izhikevish Neuron")
            plt.plot(self.t, V, "k")
            plt.ylabel("V (mV)")

            plt.subplot(3, 1, 2)
            plt.plot(self.t, u, "b")
            plt.ylabel("u (dimensionless)")

            plt.subplot(3, 1, 3)
            plt.plot(self.t, i_inj_values, "k")
            plt.ylabel("$I_{inj}$ ($\\mu{A}/cm^2$)")
            plt.xlabel("t (ms)")

            plt.show()

        return V, u, spikes, i_inj_values


if __name__ == "__main__":
    runner = Izhikevish(exp_time=10)
    runner.run(J=[14000], times=[(0.5, 5)])
