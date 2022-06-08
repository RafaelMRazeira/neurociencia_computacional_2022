import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class PinskyRinzel:
    """Full Pinsky-Rinzel Model implemented in Python.

    Parameters
    ----------
    g_c : float
        Conductance of the Ca channel in mS/cm^2.
    C_s : float
        Membrane Capacitance, in uF/cm^2.
    C_d : float
        Membrane Capacitance, in uF/cm^2.
    p : float
        Proportion of membrane area taken by the soma.
    g_L : float
        Leak maximum conductances, in mS/cm^2.
    g_Na : float
        Sodium (Na) maximum conductances, in mS/cm^2.
    g_K : float
        Potassium (K) maximum conductances, in mS/cm^2.
    g_Ca : float
        Calcium (Ca) maximum conductances, in mS/cm^2.
    g_KCa : float
        Calcium (Ca) maximum conductances dependent of Potassium, in mS/cm^2.
    g_AHP : float
        AHP maximum conductances, in mS/cm^2.
    E_Na : float
        Sodium (Na) Nernst reversal potentials, in mV.
    E_K : float
        Postassium (K) Nernst reversal potentials, in mV.
    E_L : float
        Leak Nernst reversal potentials, in mV.
    E_Ca : float
        Calcium (Ca) Nernst reversal potentials, in mV.
    E_h : float
        Hyperpolarization Nernst reversal potentials, in mV.
    exp_time : float
        Time to integrate over, in ms.
    """

    def __init__(
        self,
        g_c=20e-9,
        g_h=0,
        C_s=100e-12,
        C_d=100e-12,
        p=1 / 3,
        g_Ls=5e-9,
        g_Ld=5e-9,
        g_Na=3e-6,
        g_K=2e-6,
        g_Ca=2e-6,
        g_KCa=2.5e-6,
        g_AHP=40e-9,
        E_L=-60e-3,
        E_Na=0.060,
        E_K=-75e-3,
        E_Ca=80e-3,
        E_h=-20e-3,
        tau_ca=None,
        k=None,
        exp_time=1e3,
        dt=0.05e-3
    ):

        self.p = p
        self.g_c = g_c
        self.g_h = g_h
        self.dt = dt

        self.C_s = p * C_s
        self.g_Ls = p * g_Ls
        self.g_Ld = (1 - p) * g_Ld
        self.g_Na = p * g_Na
        self.g_K = p * g_K

        self.C_d = (1 - p) * C_d
        self.g_Ca = (1 - p) * g_Ca
        self.g_KCa = (1 - p) * g_KCa
        self.g_AHP = (1 - p) * g_AHP

        self.E_L = E_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_Ca = E_Ca
        self.E_h = E_h

        self.tau_ca = tau_ca or 50e-3
        self.k = k or 2.5e6 / (1 - self.p)
        """ The time to integrate over """
        self.t = np.arange(0.0, exp_time, dt)

    def alpha_m(self, Vs):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """

        if Vs == -0.0469:
            return 1280
        V1 = Vs + 0.0469
        alpha = 320e3 * V1 / (1 - np.exp(-250 * (V1)))
        return alpha

    def beta_m(self, Vs):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        if Vs == -0.0199:
            return 1400
        V2 = Vs + 0.0199
        beta = 280*10**3 * V2 / (np.exp(200 * V2) - 1)
        return beta

    def alpha_h(self, Vs):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        alpha = 128 * np.exp(-55.556 * (0.043 + Vs))
        return alpha

    def beta_h(self, Vs):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        V5 = Vs + 0.020
        beta = 4000 / (1 + np.exp(-200 * V5))
        return beta

    def alpha_n(self, Vs):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        if Vs == -0.0249:
            return 80
        V3 = Vs + 0.0249
        alpha = 16e3 * V3 / (1 - np.exp(-200 * V3))

        return alpha

    def beta_n(self, Vs):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        V4 = Vs + 0.040
        beta = 250 * np.exp(-25 * V4)
        return beta

    def alpha_mca(self, Vd):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        alpha = 1600 / (1 + np.exp(-72 * (Vd - 0.005)))
        return alpha

    def beta_mca(self, Vd):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        if Vd == -0.0089:
            return 100
        V6 = Vd + 0.0089
        beta = 2e4 * V6 / (np.exp(200 * V6) - 1)
        return beta

    def alpha_mkca(self, Vd):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        V7 = Vd + 0.0535
        V8 = Vd + 0.0500
        if Vd > -0.010:
            alpha = 2000 * np.exp(-37.037 * V7)
        else:
            alpha = np.exp(V8 / 0.011 - V7 / 0.027) / 0.018975
        return alpha

    def beta_mkca(self, Vd):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        V7 = Vd + 0.0535
        if Vd > -0.010:
            beta = 0
        else:
            beta = 2000 * np.exp(-V7 / 0.027) - self.alpha_mkca(Vd)
        return beta

    def alpha_mkahp(self, Ca):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        alpha = min(20, 20000 * Ca)
        return alpha

    def beta_mkahp(self, Ca):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return 4

    def m_h_inf(self, V_D):
        return 1 / (1 + np.exp(166.667 * (V_D + 0.070)))

    def tau_m_h(self, V_D):
        return 0.272 + 1.499 / (1 + np.exp(-114.548 * (V_D + 0.0422)))

    def chi(self, Ca):
        """Channel gating kinetics. Functions of membrane voltage.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """
        return min(1, 4000 * Ca)

    def I_Ls(self, V):
        """Membrane current (in uA/cm^2) Leak.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """

        return self.g_Ls * (V - self.E_L)

    def I_Ld(self, V):
        """Membrane current (in uA/cm^2) Leak.

        Parameters
        ----------
        V : float
            Membrane voltage, in mV.
        """

        return self.g_Ld * (V - self.E_L)

    def I_Na(self, Vs, m, h):
        return self.g_Na * m**2 * h * (Vs - self.E_Na)

    def I_K(self, Vs, n):
        return self.g_K * n**2 * (Vs - self.E_K)

    def I_ds(self, Vd, Vs):
        return self.g_c * (Vd - Vs)

    def I_Ca(self, Vd, mca):
        return self.g_Ca * mca**2 * (Vd - self.E_Ca)

    def I_AHP(self, Vd, mkahp):
        return self.g_AHP * mkahp * (Vd - self.E_K)

    def I_KCa(self, Vd, mkca, Ca):
        return self.g_KCa * mkca * self.chi(Ca) * (Vd - self.E_K)

    def I_h(self, Vd, mh):
        return -self.g_h * mh * (Vd - self.E_h)

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

    def dALLdt(self, Vs, Vd, m, n, h, mca, mkca, mkahp, Ca, mh, I_inj):

        I_leak_s = self.I_Ls(Vs)
        I_leak_d = self.I_Ld(Vd)
        I_sd = -self.I_ds(Vd, Vs)

        dVsdt = (1 / self.C_s) * (
            - I_leak_s
            - self.I_Na(Vs, m, h)
            - self.I_K(Vs, n)
            + self.I_ds(Vd, Vs)
            + I_inj
        )

        dVddt = (1 / self.C_d) * (
            - I_leak_d
            - self.I_Ca(Vd, mca)
            - self.I_KCa(Vd, mkca, Ca)
            - self.I_AHP(Vd, mkahp)
            + self.I_h(Vd, mh)
            + I_sd
        )

        dmdt = self.alpha_m(Vs) * (1 - m) - self.beta_m(Vs) * m
        dhdt = self.alpha_h(Vs) * (1 - h) - self.beta_h(Vs) * h
        dndt = self.alpha_n(Vs) * (1 - n) - self.beta_n(Vs) * n
        dmcadt = self.alpha_mca(Vd) * (1 - mca) - self.beta_mca(Vd) * mca
        dmkcadt = self.alpha_mkca(Vd) * (1 - mkca) - self.beta_mkca(Vd) * mkca
        dmkahpdt = self.alpha_mkahp(Ca) * (1 - mkahp) - self.beta_mkahp(Ca) * mkahp
        dmhdt = (self.m_h_inf(Vd) - mh) / self.tau_m_h(Vd)
        dCadt = - Ca / self.tau_ca - self.k * self.I_Ca(Vd, mca)

        return np.array([dVsdt, dVddt, dmdt, dndt, dhdt, dmcadt, dmkcadt, dmkahpdt, dCadt, dmhdt])

    def runge_kutta(self, X, i_inj_values, g_AHP=40e-9):
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

        Vs, Vd, m, n, h, mca, mkca, mkahp, ca, mh = X
        
        Vs_out = np.ones(points_total) * Vs
        Vd_out = np.ones(points_total) * Vd
        m_out = np.ones(points_total) * m
        n_out = np.ones(points_total) * n
        h_out = np.ones(points_total) * h
        mca_out = np.ones(points_total) * mca
        mkca_out = np.ones(points_total) * mkca
        mkahp_out = np.ones(points_total) * mkahp
        ca_out = np.ones(points_total) * ca
        mh_out = np.ones(points_total) * mh
        IKCa_out = np.zeros(points_total) 
        IKAHP_out = np.zeros(points_total)
        Ih_out = np.zeros(points_total)


        for t in tqdm(range(1, points_total)):
            k1 = self.dt * self.dALLdt(Vs, Vd, m, n, h, mca, mkca, mkahp, ca, mh, i_inj_values[t])
            k2 = self.dt * self.dALLdt(
                Vs + 0.5 * k1[0],
                Vd + 0.5 * k1[1],
                m + 0.5 * k1[2],
                n + 0.5 * k1[3],
                h + 0.5 * k1[4],
                mca + 0.5 * k1[5],
                mkca + 0.5 * k1[6],
                mkahp + 0.5 * k1[7],
                ca + 0.5 * k1[8],
                mh + 0.5 * k1[9],
                i_inj_values[t],
            )

            k3 = self.dt * self.dALLdt(
                Vs + 0.5 * k2[0],
                Vd + 0.5 * k2[1],
                m + 0.5 * k2[2],
                n + 0.5 * k2[3],
                h + 0.5 * k2[4],
                mca + 0.5 * k2[5],
                mkca + 0.5 * k2[6],
                mkahp + 0.5 * k2[7],
                ca + 0.5 * k2[8],
                mh + 0.5 * k2[9],
                i_inj_values[t],
            )

            k4 = self.dt * self.dALLdt(
                Vs + k3[0],
                Vd + k3[1],
                m + k3[2],
                n + k3[3],
                h + k3[4],
                mca + k3[5],
                mkca + k3[6],
                mkahp + k3[7],
                ca + k3[8],
                mh + k3[9],
                i_inj_values[t],
            )

            Vs = Vs + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            Vs_out[t] = Vs

            Vd = Vd + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            Vd_out[t] = Vd

            m = m + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
            m_out[t] = m

            n = n + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
            n_out[t] = n

            h = h + (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6
            h_out[t] = h

            mca = mca + (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]) / 6
            mca_out[t] = mca

            mkca = mkca + (k1[6] + 2 * k2[6] + 2 * k3[6] + k4[6]) / 6
            mkca_out[t] = mkca

            mkahp = mkahp + (k1[7] + 2 * k2[7] + 2 * k3[7] + k4[7]) / 6
            mkahp_out[t] = mkahp

            ca = ca + (k1[8] + 2 * k2[8] + 2 * k3[8] + k4[8]) / 6
            ca_out[t] = ca

            mh = mh + (k1[9] + 2 * k2[9] + 2 * k3[9] + k4[9]) / 6
            mh_out[t] = mh

            IKCa_out[t] = self.I_KCa(Vd, mkca, ca)
            IKAHP_out[t] = self.I_AHP(Vd, mkahp)
            Ih_out[t] = self.I_h(Vd, mh)

        return (
            Vs_out,
            Vd_out,
            m_out,
            n_out,
            h_out,
            mca_out,
            mkca_out,
            mkahp_out,
            ca_out,
            mh_out,
            IKCa_out,
            IKAHP_out,
            Ih_out,
        )

    def run(self, J, times, V=None, Ca=None):
        """Run the simulation.

        Parameters
        ----------
        J : list[float]
            Stimulus current in mA/cm^2 at each time.
        times : list[tuple]
            List of time_initials and finals for pulses.
        """

        i_inj_values = [self.I_inj(t, J, times) for t in self.t]

        Vs0 = V if V is not None else -60e-3
        Vd0 = V if V is not None else -60e-3
        m0 = 0
        n0 = 0.4
        h0 = 0.5
        mca0 = 0.0
        mkca0 = 0.2
        mkahp0 = 0.2
        Ca0 = Ca if Ca is not None else 0
        mh0 = 0

        X = [Vs0, Vd0, m0, n0, h0, mca0, mkca0, mkahp0, Ca0, mh0]

        sol = self.runge_kutta(X, i_inj_values)

        Vs = sol[0]
        Vd = sol[1]
        m = sol[2]
        n = sol[3]
        h = sol[4]
        mca = sol[5]
        mkca = sol[6]
        mkahp = sol[7]
        ca = sol[8]
        breakpoint()
        plt.figure(figsize=(16, 10))

        plt.subplot(4, 1, 1)
        plt.title("Pinsky Rinzel Model")
        plt.plot(self.t, Vs, "k")
        plt.ylabel("Vs (mV)")

        plt.subplot(4, 1, 2)
        plt.plot(self.t, m, "r", label="m")
        plt.plot(self.t, h, "g", label="h")
        plt.plot(self.t, n, "b", label="n")
        plt.ylabel("Gating Values from Vs")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(self.t, Vd, "k")
        plt.ylabel("Vd (mV)")

        plt.subplot(4, 1, 4)
        plt.plot(self.t, mca, "r", label="mca")
        plt.plot(self.t, mkca, "g", label="mkca")
        plt.plot(self.t, mkahp, "b", label="mkahp")
        plt.plot(self.t, ca, "y", label="ca")
        plt.ylabel("Gating Values from Vd")
        plt.legend()

        plt.show()

        return sol


if __name__ == "__main__":
    PR = PinskyRinzel(exp_time=1.0)
    PR.run(J=[0], times=[(0, 100)])
