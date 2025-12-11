import numpy as np
import streamlit as st
#==================================================== Beta function ======================================================#
def Beta(d, sigma):
    d = np.array(d).reshape(-1)
    n = len(d)
    d_min = 1e-6

    ef = np.zeros((n, n))
    Beta_M = np.zeros((n, n))

    ef_max = np.pi * sigma * (2 * (d / (2 ** (1/3)))**2 - d**2)
    ef_min = np.pi * sigma * ((d**3 - d_min**3)**(2/3) + d_min**2 - d**2)

    for i in range(n):
        for j in range(i, n):
            ef[i, j] = max(
                np.pi * sigma * ((d[j]**3 - d[i]**3)**(2/3) + d[i]**2 - d[j]**2),
                0.0
            )

    for j in range(1, n):
        num = ef_min[j] + ef_max[j] - ef[:j, j]
        Beta_M[:j, j] = num / np.sum(num)

    return Beta_M, ef_min, ef_max, ef
#=========================================================================================================================#

#=================================================== g value function ====================================================#
def g(d, Kb, epsl, mu_o, rho_o, rho_w, ef_min, ef_max):
    d = np.array(d).reshape(-1)
    n = len(d)

    a = 1.1
    c1 = 1.3

    ud = 1.03 * (epsl * d)**(1/3)
    Ec = 0.5 * (ef_max + ef_min)
    Ev = a * (np.pi/6 * epsl**(1/3) * d**(7/3) * mu_o * np.sqrt(rho_w / rho_o))

    g_V = np.zeros(n)

    for i in range(n):
        de = np.logspace(-6, np.log10(d[i]), 200)

        Sed = np.pi/4 * (de + d[i])**2
        ue  = 2.27 * (epsl * de)**(1/3)
        e   = 1.35 * rho_w * de**(11/3) * epsl**(2/3)

        BE = np.exp(-(Ec[i] + Ev[i]) / (c1 * e))
        dg = Kb * Sed * np.sqrt(ue**2 + ud[i]**2) * BE * 0.812 * de**(-4)

        g_V[i] = np.trapz(dg, de)

    return g_V
#=========================================================================================================================#

#============================================== second droplet calculation ===============================================#
def secondDrop(d):
    d = np.array(d).reshape(-1)
    n = len(d)
    vol = np.pi/6 * d**3

    indxMat = np.zeros((n, n), dtype=int)
    fracMat = np.zeros((n, n))
    secondVol = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            secondVol[j, i] = vol[i] - vol[j]
            dnew = (d[i]**3 - d[j]**3)**(1/3)

            ismall = np.searchsorted(d, dnew) - 1
            if ismall < 0:
                indxMat[j, i] = 1
                fracMat[j, i] = 1.0
            else:
                delta = d[ismall+1] - d[ismall]
                frac = (dnew - d[ismall]) / delta
                indxMat[j, i] = ismall
                fracMat[j, i] = frac

    return indxMat, fracMat, secondVol
#=========================================================================================================================#

#======================================================= main loop =======================================================#
def run_vdrop(Kb,epsl,mu_o,rho_o,mu_w,rho_w,sigma,dmax,nBins,dt,tmax,tInter, alpha):
    
    # initialization ------------------------------------------------------------------
    dmin = dmax / nBins
    d = np.linspace(dmin, dmax, nBins)
    vol = np.pi/6 * d**3
    
    time = np.arange(0, tmax+dt, dt)
    nt=int(tmax/tInter)
    
    tSave = np.zeros(nt+1)
    
    nSave = np.zeros((nt+1, len(d)))
    n_current = np.zeros(len(d))
    n_current[-1] = alpha / vol[-1] 
    nSave[0,:]=n_current
    
    
    indxMat, fracMat, secondVol = secondDrop(d)
    Beta_M, ef_min, ef_max, ef = Beta(d, sigma)
    g_V = g(d, Kb, epsl, mu_o, rho_o, rho_w, ef_min, ef_max)

    if np.any(g_V > 1/dt):
        st.error("Time step is too large: reduce dt.")
        st.stop()
        return

    # VDROP loop -----------------------------------------------------------------------
    for ti in range(1, len(time)):
        n_prev = n_current
        n_break = g_V * n_prev * dt
        n_break = np.minimum(n_break, n_prev)

        N_break = Beta_M * n_break
        break_n_1 = np.sum(N_break, axis=1)
        break_n_2 = np.zeros(len(d))

        for i in range(len(d)):
            for j in range(i):
                idx = indxMat[j, i]
                frac = fracMat[j, i]
                newVol = secondVol[j, i]
                v1 = vol[idx]
                v2 = vol[idx+1]

                break_n_2[idx] += Beta_M[j, i] * n_break[i] * (1-frac) * newVol / v1
                break_n_2[idx+1] += Beta_M[j, i] * n_break[i] * frac * newVol / v2

        dn = break_n_1 + break_n_2 - n_break
        n_current = n_prev + dn
        
        if ti*dt % tInter == 0:
            tSave[int(ti*dt / tInter)]=ti*dt
            nSave[int(ti*dt / tInter),:]=n_current
    return d, vol, tSave, nSave