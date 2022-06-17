import numpy as np
import pylab as plt
from scipy.integrate import odeint


class FitzHughNagumo:
    """Full FitzHugh-Nagumo Model implemented in Python.

    Parameters
    ----------
    exp_time : float
        Total time of the experiment, in ms.
    """

    def __init__(
        self,
        exp_time=40.0,
    ):

        """The time to integrate over"""
        self.t = np.arange(0.0, exp_time, 1)

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
        V, w = X

        dVdt = V - (V**3 / 3) - w + self.I_inj(t, J, times)
        dwdt = 0.08 * (V + 0.7 - 0.8 * w)

        return dVdt, dwdt

    def run(self, J, times, V0=-0.65, plot=True):
        """Run demo for the FitzHugh-Nagumo Model.

        Parameters
        ----------
        J : float
            Current amplitude, in uA/cm^2.
        times : list[tuple]
            List of time_initials and finals for pulses.
        plot : bool
            If True, plots the results.
        """
        X = odeint(self.dALLdt, [V0, -1.21], self.t, args=(J, times, self))
        V = X[:, 0]
        w = X[:, 1]

        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

        if plot:
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 1, 1)
            plt.title("FitzHugh-Nagumo Model")
            plt.plot(self.t, V, "k")
            plt.ylabel("V (mV)")

            plt.subplot(2, 1, 2)
            plt.plot(self.t, w, "c", label="w")
            plt.ylabel("w")
            plt.legend()
            plt.show()

        return V, w, i_inj_values


if __name__ == "__main__":
    runner = FitzHughNagumo(exp_time=40)
    runner.run(J=[0.0], times=[(0, 40)], V0=0.0)
