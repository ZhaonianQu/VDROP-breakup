# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from VDROP import run_vdrop


#try:
if "results" not in st.session_state:
    st.session_state.results = None

st.set_page_config(layout="wide")


# ======= MAIN LAYOUT: left (inputs) | right (results) =======
left_col, right_col = st.columns([1, 2.5])  # adjust ratio if you want

# ---------------- LEFT: INPUT PANEL ----------------
with left_col:
    st.markdown("### Input")

    # Form so we only run when the button is clicked
    with st.form("vdrop_inputs"):

        # Example: 3×2 "table" of inputs
        st.markdown("<p style='font-size:18px; font-weight:600; margin-bottom:5px;'>System parameters</p>", 
            unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            Kb_str   = st.text_input("Kb", "0.01")
            
            
        with c2:
            epsl_str = st.text_input("Turbulent dissipation rate (W/kg)", "0.1")
            
            
        st.markdown("<p style='font-size:18px; font-weight:600; margin-bottom:5px;'>Fluid parameters</p>", 
            unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            rho_o_str = st.text_input("Density of dispersed phase (kg/m³)", "840")
            mu_o_str  = st.text_input("Viscosity of dispersed phase (Pa·s)", "0.004")
            alpha_str = st.text_input("Holdup of dispersed phase", "0.1")
            
        with c4:
            rho_w_str = st.text_input("Density of continuous phase (kg/m³)", "1025")
            mu_w_str  = st.text_input("Viscosity of continuous phase (Pa·s)", "1.1e-3")
            sigma_str = st.text_input("IFT (N/m)", "0.0256")
            
        st.markdown("<p style='font-size:18px; font-weight:600; margin-bottom:5px;'>Discretization</p>", 
            unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            dmax_str = st.text_input("Largest droplet size (m)", "0.001")
            nBins    = st.number_input("Number of droplet bins", 5, 200, 20, 1)
            
        with c6:
            tmax_str  = st.text_input("Simulation time (s)", "600")
            dt_str    = st.text_input("Time step size (s)", "0.1")
            tInter_str = st.text_input("Save interval (s)", "60")
            
            
        run_button = st.form_submit_button("Run")
        
if run_button:
        # convert to floats/ints
    Kb    = float(Kb_str)
    epsl  = float(epsl_str)
    alpha = float(alpha_str)
    mu_o  = float(mu_o_str)
    rho_o = float(rho_o_str)
    mu_w  = float(mu_w_str)
    rho_w = float(rho_w_str)
    sigma = float(sigma_str)
    dmax  = float(dmax_str)
    dt    = float(dt_str)
    tmax  = float(tmax_str)
    tInter = float(tInter_str)

    d, vol, tSave, nSave = run_vdrop(
        Kb, epsl, mu_o, rho_o, mu_w, rho_w,
        sigma, dmax, int(nBins), dt, tmax, tInter, alpha
    )
    # store in session state
    #st.session_state.results = (Kb,epsl,alpha,mu_o,rho_o,mu_w,rho_w,sigma,dmax,dt,tmax,tInter,d, vol, tSave, nSave)
    st.session_state.results = (d, vol, tSave, nSave)


# ---------------- RIGHT: RESULTS PANEL ----------------
if st.session_state.results is not None:
    d, vol, tSave, nSave = st.session_state.results
    d_um = d * 1e6
    vf=np.zeros((len(d)))
    CV=np.zeros((len(d)))
    V_total = np.sum(vol * nSave[-1,:])
    vf = vol * nSave[-1,:] / V_total
    CV = np.cumsum(nSave[-1,:] * vol / V_total)
            
    with right_col:
    
        st.markdown("### Results")
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:    
            width = (d_um[1] - d_um[0]) * 0.8  # wide bars
            fig, ax = plt.subplots()
            ax.set_xlim(min(d_um) - width, max(d_um) + width)
            ax.set_xlabel("Droplet diameter (um)")
            ax.set_ylabel("Volume fraction")
            ax.bar(d_um, vf, width=width,color='black')
            ax.set_title(f"Final Volume fraction")
            st.pyplot(fig) 

        with plot_col2:
            fig, ax = plt.subplots()
            ax.set_xlim(min(d_um) - width, max(d_um) + width)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Droplet diameter (um)")
            ax.set_ylabel("Cumulative volume fraction")
            ax.plot(d_um, CV, 'k-o')
            ax.set_title(f"Final cumulative volume fraction")
            st.pyplot(fig) 

        st.markdown("### Data download")
        
        placeHolder = np.array(["-"] * (len(d)-1), dtype=object)
        # Convert arrays
        d = np.asarray(d, dtype=object)
        vol = np.asarray(vol, dtype=object)
        nSave = np.asarray(nSave, dtype=object)

        # ----- Header rows -----
        row_d   = np.concatenate((["d (m)"], d))
        row_vol = np.concatenate((["vol (m3)"], vol))
        row_pH  = np.concatenate((["tSave (s)"],["number concentration (/m3)"], placeHolder))

        # ----- Build time labels correctly -----
        time_labels = np.array(
            tSave,
            dtype=object
        ).reshape(-1, 1)

        # ----- Stack times + nSave block -----
        n_block = np.hstack((time_labels, nSave))
        
        param_data = [
            ["System parameters", ""],
            ["Kb", Kb_str], # Replace with actual variable
            ["Turbulent dissipation rate (W/kg)", epsl_str],
            ["", ""], # Empty row spacer
            ["Fluid parameters", ""],
            ["Density of dispersed phase (kg/m3)", rho_o_str],
            ["Viscosity of dispersed phase (Pa.s)", mu_o_str],
            ["Density of continuous phase (kg/m3)", rho_w_str],
            ["Viscosity of continuous phase (Pa.s)", mu_w_str],
            ["IFT (N/m)", sigma_str],
            ["Holdup of dispersed phase", alpha_str],
            ["", ""], # Empty row spacer
            ["Discretization", ""],
            ["Largest droplet size (m)", dmax_str],
            ["Simulation time (s)", tmax_str],
            ["Time step size (s)", dt_str]
        ]
        
        # ----- Build final output -----
        left_block = np.array(param_data, dtype=object)
        right_block = np.vstack((row_d, row_vol, row_pH, n_block)).astype(object)
        combined_csv_rows = []
        combined_csv_rows.append(["", "", "", "VDROP output"] + [""] * (len(d) - 1))
        
        rows_left= len(param_data)
        rows_right=len(right_block)
        max_rows = max(rows_left, rows_right)
        # --- Pad left block ---
        if rows_left < max_rows:
            padding_rows_left = max_rows - rows_left
            pad_left = np.full((padding_rows_left, left_block.shape[1]), "", dtype=object)
            padded_left_block = np.vstack((left_block, pad_left))
        else:
            padded_left_block = left_block

        # --- Pad right block ---
        if rows_right < max_rows:
            padding_rows_right = max_rows - rows_right
            pad_right = np.full((padding_rows_right, right_block.shape[1]), "", dtype=object)
            padded_right_block = np.vstack((right_block, pad_right))
        else:
            padded_right_block = right_block

        # ----- Spacer column -----
        spacer_block = np.full((max_rows, 1), "", dtype=object)

        # ----- Combine left + spacer + right -----
        full_block = np.hstack((padded_left_block, spacer_block, padded_right_block))
        title_row = ["", "", ""] + ["VDROP output"] + [""] * (padded_right_block.shape[1] - 1)
        csv_matrix = np.vstack((np.array(title_row, dtype=object), full_block))
        
        csv = "\n".join(",".join(map(str, row)) for row in csv_matrix)
        st.download_button(
            "Number concentration",
            csv,
            "VDROP_output.csv",
            "text/csv"
        )
else:
    with right_col:
        st.markdown("### Results")
        st.info("Set parameters on the left and click **Run**.")
#except:
    #st.error("❌ Error occurs. Please check your inputs.")
    #st.stop()