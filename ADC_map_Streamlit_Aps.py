# mri_simulator_updated.py
# Run with:
# pip install streamlit torch matplotlib pandas numpy
# streamlit run mri_simulator_updated.py

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

torch.set_default_dtype(torch.float32)
st.title("MRI Physics Simulator with Tissue Presets (bSSFP + comparisons)")

# -------------------------------------------------------------------
# Tissue presets (times in ms, D in mm^2/s)
# -------------------------------------------------------------------
tissue_presets = {
    "Brain White Matter": {"T1": 900, "T2": 80, "T2*": 60, "D": 0.7},
    "Brain Gray Matter": {"T1": 1300, "T2": 100, "T2*": 50, "D": 0.8},
    "CSF": {"T1": 4000, "T2": 2000, "T2*": 100, "D": 3.0},
    "Muscle": {"T1": 900, "T2": 40, "T2*": 30, "D": 1.0},
    "Fat": {"T1": 300, "T2": 70, "T2*": 50, "D": 0.3},
}

tissue_choice = st.sidebar.selectbox("Select Tissue Type", list(tissue_presets.keys()))
preset = tissue_presets[tissue_choice]

# -------------------------------------------------------------------
# Sidebar sliders
# -------------------------------------------------------------------
T1_val  = st.sidebar.slider("T1 (ms)", 200, 5000, value=preset["T1"])
T2_val  = st.sidebar.slider("T2 (ms)", 20, 3000, value=preset["T2"])
T2s_val = st.sidebar.slider("T2* (ms)", 10, 300, value=preset["T2*"])

alpha_deg = st.sidebar.slider("Flip Angle α (°)", 1, 90, 30)
TR_val    = st.sidebar.slider("TR (ms)", 50, 5000, 500)
TE_val    = st.sidebar.slider("TE (ms)", 1, 200, 20)

# Diffusion gradient pulse sequence parameters for b-value
st.sidebar.markdown("### Diffusion Gradients (for b-value)")
G_mTm   = st.sidebar.slider("Gradient strength G (mT/m)", 0.0, 80.0, 40.0)
delta_ms  = st.sidebar.slider("δ (ms) – gradient duration", 5.0, 50.0, 30.0)
Delta_ms  = st.sidebar.slider("Δ (ms) – gradient spacing", 20.0, 120.0, 60.0)

noise_std = st.sidebar.slider("Noise σ (a.u.)", 0.001, 0.2, 0.05)

# -------------------------------------------------------------------
# Convert to tensors / units
# -------------------------------------------------------------------
T1 = torch.tensor(T1_val / 1000.0)       # s
T2 = torch.tensor(T2_val / 1000.0)       # s
T2_star = torch.tensor(T2s_val / 1000.0) # s
TR = torch.tensor(TR_val / 1000.0)       # s
TE = torch.tensor(TE_val / 1000.0)       # s
alpha_rad = torch.tensor(np.deg2rad(alpha_deg), dtype=torch.float32)

sigma = torch.tensor(noise_std)
M0 = 1.0

# Time axes (for illustrative plots)
t  = torch.linspace(0, 1, 1000)   # s
TI = torch.linspace(0, 2, 1000)   # s

# -------------------------------------------------------------------
# Compute b-value from gradient parameters (SI -> s/mm^2)
# b = γ^2 * G^2 * δ^2 * (Δ - δ/3)
# γ in rad/s/T; G in T/m; δ, Δ in s; result b in s/m^2 -> convert to s/mm^2 by /1e6
# -------------------------------------------------------------------
gamma = 2.675e8  # rad/s/T (proton gyromagnetic ratio)
G_Tpm = torch.tensor(G_mTm / 1000.0)      # T/m
delta_s = torch.tensor(delta_ms / 1000.0) # s
Delta_s = torch.tensor(Delta_ms / 1000.0) # s

Delta_eff = torch.clamp(Delta_s - delta_s / 3.0, min=0.0)
b_SI = (gamma**2) * (G_Tpm**2) * (delta_s**2) * (Delta_eff)  # s/m^2
b = b_SI / 1e6  # s/mm^2

st.sidebar.info(f"Computed b-value ≈ {b.item():.0f} s/mm²")

# -------------------------------------------------------------------
# MRI signal models (single-tissue view)
# -------------------------------------------------------------------
# Longitudinal recovery and transverse decay
Mz_T1 = M0 * (1 - torch.exp(-t / T1))
Mxy_T2 = M0 * torch.exp(-t / T2)
Mxy_T2_star = M0 * torch.exp(-t / T2_star)

# Simple FID with off-resonance (illustrative; 3T)
FID = M0 * torch.exp(-t / T2_star) * torch.exp(1j * 2 * np.pi * 42.577e6 * 3.0 * t)

# Spin Echo (SE) and Gradient Echo (GRE/SPGR)
SpinEcho = M0 * (1 - torch.exp(-TR / T1)) * torch.exp(-TE / T2)
GradEcho = M0 * (torch.sin(alpha_rad) * (1 - torch.exp(-TR / T1))) / \
           (1 - torch.cos(alpha_rad) * torch.exp(-TR / T1)) * torch.exp(-TE / T2_star)
SPGR = GradEcho.clone()

# S0 for diffusion (use Spin Echo steady state; SE-EPI readout commonly used in DWI)
S0_SE = M0 * (1 - torch.exp(-TR / T1)) * torch.exp(-TE / T2)

# Tissue diffusion coefficient (mm^2/s) for the chosen tissue (used in single-voxel demo plots)
D_val_mm2s = tissue_presets[tissue_choice]["D"]
D_single = torch.tensor(D_val_mm2s)  # mm^2/s

# Diffusion-weighted signal using b from gradients
S_b = S0_SE * torch.exp(-b * D_single)

# Estimated ADC from S(b) and S0 (avoid div-by-zero if b≈0)
ADC_est = torch.tensor(0.0) if b.item() <= 1e-9 else -(1.0 / b) * torch.log(torch.clamp(S_b / S0_SE, min=1e-12, max=1.0))

SNR = M0 / sigma
Mz_IR = M0 * (1 - 2 * torch.exp(-TI / T1))

# -------------------------------------------------------------------
# Main plots (keep existing grid)
# -------------------------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# T1 recovery
axs[0, 0].plot(t.numpy(), Mz_T1.numpy())
axs[0, 0].set_title(f"T1 Recovery (T1={T1_val} ms)")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Mz")

# T2/T2*
axs[0, 1].plot(t.numpy(), Mxy_T2.numpy(), label="T2")
axs[0, 1].plot(t.numpy(), Mxy_T2_star.numpy(), label="T2*")
axs[0, 1].set_title("T2 / T2* Decay")
axs[0, 1].legend()
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Mxy")

# FID
axs[1, 0].plot(t.numpy(), torch.real(FID).numpy(), label='Real')
axs[1, 0].plot(t.numpy(), torch.imag(FID).numpy(), label='Imaginary')
axs[1, 0].set_title("FID Signal")
axs[1, 0].legend()
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Signal")

# Echo comparison
axs[1, 1].bar(['Spin Echo', 'Grad Echo', 'SPGR'],
              [SpinEcho.item(), GradEcho.item(), SPGR.item()])
axs[1, 1].set_title("Echo Signal Comparison (single tissue)")

# Inversion recovery
axs[2, 0].plot(TI.numpy(), Mz_IR.numpy())
axs[2, 0].set_title("Inversion Recovery")
axs[2, 0].set_xlabel("TI (s)")
axs[2, 0].set_ylabel("Mz")

# DWI, SNR & ADC (single-voxel for selected tissue)
axs[2, 1].bar(['S(b)', 'S0 (SE)', 'SNR', 'ADC_est'],
              [S_b.item(), S0_SE.item(), SNR.item(), ADC_est.item() if b.item() > 1e-9 else 0.0])
axs[2, 1].set_title(f"DWI (b={b.item():.0f} s/mm²), SNR & ADC (tissue: {tissue_choice})")

plt.tight_layout()
st.pyplot(fig)

# -------------------------------------------------------------------
# Dynamic pseudo-square visualization across all tissues (add bSSFP)
# -------------------------------------------------------------------
contrast_signals = {}
for tissue, vals in tissue_presets.items():
    T1_t = vals["T1"] / 1000.0
    T2_t = vals["T2"] / 1000.0
    T2s_t = vals["T2*"] / 1000.0
    D_t = vals["D"]            # mm^2/s

    # SE, GRE/SPGR for reference contrasts
    SE = M0 * (1 - np.exp(-TR.item() / T1_t)) * np.exp(-TE.item() / T2_t)
    GE = M0 * (np.sin(alpha_rad.item()) * (1 - np.exp(-TR.item() / T1_t))) / \
         (1 - np.cos(alpha_rad.item()) * np.exp(-TR.item() / T1_t)) * np.exp(-TE.item() / T2s_t)
    SPGR_sig = GE

    # ----- bSSFP steady-state (general TE) -----
    # E1 and E2 computed over TR
    E1_t = np.exp(-TR.item() / T1_t)
    E2_t = np.exp(-TR.item() / T2_t)
    denom = 1.0 - (E1_t - E2_t) * np.cos(alpha_rad.item()) - (E1_t * E2_t)
    # Avoid division by zero
    if abs(denom) < 1e-12:
        bssfp_base = 0.0
    else:
        bssfp_base = (1.0 - E1_t) * np.sin(alpha_rad.item()) / denom
    # Multiply by explicit transverse decay from RF to TE (use T2 because bSSFP refocuses static inhomogeneity)
    bSSFP_sig = M0 * bssfp_base * np.exp(-TE.item() / T2_t)

    # Diffusion: S0 from SE + b from gradients
    S0_t = M0 * (1 - np.exp(-TR.item() / T1_t)) * np.exp(-TE.item() / T2_t)
    Sb_t = S0_t * np.exp(-b.item() * D_t)
    ADC_val = 0.0 if b.item() <= 1e-9 else (-(1.0 / b.item()) * np.log(max(Sb_t / S0_t, 1e-12)))

    SNR_sig = M0 / sigma.item()

    contrast_signals[tissue] = {
        "SE": SE, "GE": GE, "SPGR": SPGR_sig, "bSSFP": bSSFP_sig,
        "DWI": Sb_t, "S0": S0_t, "SNR": SNR_sig, "ADC": ADC_val
    }

# Build arrays for normalized pseudo-images
tissue_list = list(contrast_signals.keys())
SE_arr   = np.array([contrast_signals[t]["SE"]   for t in tissue_list])
GE_arr   = np.array([contrast_signals[t]["GE"]   for t in tissue_list])
SPGR_arr = np.array([contrast_signals[t]["SPGR"] for t in tissue_list])
bSSFP_arr= np.array([contrast_signals[t]["bSSFP"] for t in tissue_list])
DWI_arr  = np.array([contrast_signals[t]["DWI"]  for t in tissue_list])
S0_arr   = np.array([contrast_signals[t]["S0"]   for t in tissue_list])
SNR_arr  = np.array([contrast_signals[t]["SNR"]  for t in tissue_list])
ADC_arr  = np.array([contrast_signals[t]["ADC"]  for t in tissue_list])

def normalize(arr):
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / (rng + 1e-12)

SE_norm   = normalize(SE_arr)
GE_norm   = normalize(GE_arr)
SPGR_norm = normalize(SPGR_arr)
bSSFP_norm= normalize(bSSFP_arr)
DWI_norm  = normalize(DWI_arr)
S0_norm   = normalize(S0_arr)
SNR_norm  = normalize(SNR_arr)
ADC_norm  = normalize(ADC_arr)

fig2, axs2 = plt.subplots(8, len(tissue_list), figsize=(14, 16))

for i, tissue in enumerate(tissue_list):
    axs2[0, i].imshow(np.ones((20, 20)) * SE_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[0, i].set_title(f"{tissue}\nSE", fontsize=8)
    axs2[0, i].axis("off")

    axs2[1, i].imshow(np.ones((20, 20)) * GE_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[1, i].set_title(f"{tissue}\nGE", fontsize=8)
    axs2[1, i].axis("off")

    axs2[2, i].imshow(np.ones((20, 20)) * SPGR_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[2, i].set_title(f"{tissue}\nSPGR", fontsize=8)
    axs2[2, i].axis("off")

    axs2[3, i].imshow(np.ones((20, 20)) * bSSFP_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[3, i].set_title(f"{tissue}\nbSSFP", fontsize=8)
    axs2[3, i].axis("off")

    axs2[4, i].imshow(np.ones((20, 20)) * S0_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[4, i].set_title(f"{tissue}\nS0 (SE)", fontsize=8)
    axs2[4, i].axis("off")

    axs2[5, i].imshow(np.ones((20, 20)) * DWI_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[5, i].set_title(f"{tissue}\nS(b)", fontsize=8)
    axs2[5, i].axis("off")

    axs2[6, i].imshow(np.ones((20, 20)) * SNR_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[6, i].set_title(f"{tissue}\nSNR", fontsize=8)
    axs2[6, i].axis("off")

    axs2[7, i].imshow(np.ones((20, 20)) * ADC_norm[i], cmap="gray", vmin=0, vmax=1)
    axs2[7, i].set_title(f"{tissue}\nADC", fontsize=8)
    axs2[7, i].axis("off")

plt.tight_layout()
st.pyplot(fig2)

# -------------------------------------------------------------------
# Signal comparison bar plot across tissues (SE, SPGR, bSSFP at chosen TE)
# -------------------------------------------------------------------
labels = tissue_list
x = np.arange(len(labels))
width = 0.25

SE_vals = SE_arr
SPGR_vals = SPGR_arr
bSSFP_vals = bSSFP_arr

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.bar(x - width, SE_vals, width, label='SE')
ax3.bar(x, SPGR_vals, width, label='SPGR')
ax3.bar(x + width, bSSFP_vals, width, label='bSSFP')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=45, ha='right')
ax3.set_ylabel("Signal (a.u.)")
ax3.set_title(f"Signal comparison at TE={TE_val} ms, TR={TR_val} ms, α={alpha_deg}°")
ax3.legend()
plt.tight_layout()
st.pyplot(fig3)

# -------------------------------------------------------------------
# Show numerical table (b, S0, S(b), ADC) and the contrast signals
# -------------------------------------------------------------------
rows = []
for t in tissue_list:
    rows.append({
        "Tissue": t,
        "b (s/mm²)": b.item(),
        "S_SE": contrast_signals[t]["SE"],
        "S_SPGR": contrast_signals[t]["SPGR"],
        "S_bSSFP": contrast_signals[t]["bSSFP"],
        "S(b)": contrast_signals[t]["DWI"],
        "ADC_est (mm²/s)": contrast_signals[t]["ADC"]
    })
df = pd.DataFrame(rows).set_index("Tissue")

st.subheader("Numerical Values")
st.dataframe(df.style.format({
    "b (s/mm²)": "{:.0f}",
    "S_SE": "{:.4f}",
    "S_SPGR": "{:.4f}",
    "S_bSSFP": "{:.4f}",
    "S(b)": "{:.4f}",
    "ADC_est (mm²/s)": "{:.4f}",
}))

st.markdown(
    "Notes:\n"
    "- SPGR uses T2* in the echo decay factor.\n"
    "- bSSFP formula implemented is the on-resonance steady-state magnitude evaluated at arbitrary TE by multiplying the closed-form base by `exp(-TE/T2)`.\n"
    "- This is a single-voxel / analytic simulator, not an image-formation pipeline. Use small TR and TE values for realistic bSSFP behavior."
)
