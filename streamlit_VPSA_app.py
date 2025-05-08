import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# This must be the first Streamlit command
st.set_page_config(page_title="VPSA Output Predictor", layout="wide")

# Load scalers
X_scaler = joblib.load("1X_scaler.pkl")
y_scaler = joblib.load("2y_scaler.pkl")

# --------------------------
# Define Dataset class (same as in Code 1)
# --------------------------
class NNDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# --------------------------
# Load model with MC Dropout (corrected class name)
# --------------------------
class MCPytorchNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=57, dropout_rate=0.127):
        super(MCPytorchNN, self).__init__()
        self.mc_dropout = False 

        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.log_variance = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        def apply_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        if self.mc_dropout:
            self.feature_layers.apply(apply_dropout)

        features = self.feature_layers(x)
        return self.final_layer(features)
    
    def enable_mc_dropout(self):
        self.mc_dropout = True

input_dim = 7
output_dim = 4
model = MCPytorchNN(input_dim, output_dim)
model.load_state_dict(torch.load("final_model.pt"))
model.eval()

# --------------------------
# Prediction with MC Dropout (aligned with Code 1)
# --------------------------
def predictions_with_uncertainties(model, data_loader, n_samples=100):
    model.eval()
    model.enable_mc_dropout()
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            predicts = []
            for X_batch in data_loader:
                y_preds = model(X_batch)
                predicts.append(y_preds.numpy())
            predictions.append(np.vstack(predicts))
    predictions = np.stack(predictions)
    mean_predictions = np.mean(predictions, axis=0)  # Point predictions
    std_predictions = np.std(predictions, axis=0)  # Epistemic uncertainty
    return mean_predictions, std_predictions

def make_prediction_with_mc_dropout(input_data, n_samples=100):
    # Scale input data
    input_scaled = X_scaler.transform(input_data)
    
    # Create DataLoader for input
    dataset = NNDataset(input_scaled)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Get predictions
    mean_output, std_dev = predictions_with_uncertainties(model, data_loader, n_samples)
    
    # Inverse transform
    mean_output_real = y_scaler.inverse_transform(mean_output)
    
    # Get confidence intervals (mean ± 1.96*std for 95% CI)
    lower_ci = mean_output - 1.96 * std_dev
    upper_ci = mean_output + 1.96 * std_dev
    
    lower_ci_real = y_scaler.inverse_transform(lower_ci)
    upper_ci_real = y_scaler.inverse_transform(upper_ci)
    
    # Clip to physical bounds
    labels = ["Purity", "Recovery", "Energy", "Productivity"]
    bounds = [(0, 100), (0, 100), (0, np.inf), (0, np.inf)]
    
    results = []
    for i in range(4):
        mu = mean_output_real[0, i]
        ci_low = max(bounds[i][0], lower_ci_real[0, i])
        ci_high = min(bounds[i][1] if bounds[i][1] != np.inf else mu * 10, upper_ci_real[0, i])
        results.append((mu, ci_low, ci_high))
    
    return results

# --------------------------
# UI Layout
# --------------------------
st.title("🔬 VPSA Output Predictor")

with st.sidebar:
    st.header("📥 Input Parameters")
    
    ads_time = st.number_input("Adsorption Time (s)", min_value=20.0, max_value=100.0, value=100.0)
    bd_time = st.number_input("Blowdown Time (s)", min_value=30.0, max_value=200.0, value=50.0)
    evac_time = st.number_input("Evacuation Time (s)", min_value=30.0, max_value=200.0, value=50.0)
    ads_pressure = st.number_input("Adsorption Pressure (bar)", min_value=1.0, max_value=10.0, value=4.0)
    bd_pressure = st.number_input("Blowdown Pressure (bar)", min_value=0.14, max_value=3.0, value=0.5)
    evac_pressure = st.number_input("Evacuation Pressure (bar)", min_value=0.08, max_value=0.5, value=0.1)
    feed_flowrate = st.number_input("Feed Flowrate (m/s)", min_value=0.1, max_value=2.0, value=0.2)

    if bd_pressure > ads_pressure:
        st.error("Blowdown Pressure cannot be higher than Adsorption Pressure.")
    if evac_pressure > bd_pressure:
        st.error("Evacuation Pressure cannot be higher than Blowdown Pressure.")

    n_samples = st.slider("MC Dropout Samples", min_value=10, max_value=100, value=20, 
                          help="Number of Monte Carlo samples for uncertainty estimation")
    
    predict_button = st.button("⚡ Predict Outputs")

# --------------------------
# Show VPSA Diagram
# --------------------------
st.markdown("---")
st.subheader("🧱 4 Step VPSA Process Diagram")
st.image("vpsa.drawio.png", caption="Figure: Schematic of a Vacuum Pressure Swing Adsorption (VPSA) process.", width=800)

# --------------------------
# Show results
# --------------------------
if predict_button:
    with st.spinner("Running Monte Carlo simulations..."):
        input_array = np.array([[
            ads_time, bd_time, evac_time,
            ads_pressure, bd_pressure, evac_pressure, feed_flowrate
        ]])
        result = make_prediction_with_mc_dropout(input_array, n_samples)
    
    (purity, purity_low, purity_high), \
    (recovery, rec_low, rec_high), \
    (energy, energy_low, energy_high), \
    (productivity, prod_low, prod_high) = result

    if purity > 100 or energy < 0 or productivity < 0 or recovery > 100:
        st.error("❌ Infeasible solution: Cycle configuration would not achieve CSS")
    else:
        st.success("✅ Prediction completed with MC Dropout uncertainty estimation")
        st.subheader("📊 Predicted Outputs with Confidence Intervals")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Purity (%)", f"{purity:.2f}", f"[{purity_low:.2f} - {purity_high:.2f}]")
        col2.metric("Recovery (%)", f"{recovery:.2f}", f"[{rec_low:.2f} - {rec_high:.2f}]")
        col3.metric("Energy (kWh/ton)", f"{energy:.2f}", f"[{energy_low:.2f} - {energy_high:.2f}]")
        col4.metric("Productivity\n(mol/s/m³)", f"{productivity:.6f}", f"[{prod_low:.6f} - {prod_high:.6f}]")

# --------------------------
# Zeolite 13X Info
# --------------------------
st.markdown("---")
st.subheader("🧱 Material: Zeolite 13X")
st.markdown("""
The VPSA system is based on **Zeolite 13X**, a robust and selective adsorbent commonly used for CO₂/N₂ separation.

### 📄 Isotherm Parameters

| Parameter                  | CO₂               | N₂                |
|---------------------------|-------------------|-------------------|
| b₀ [m³ mol⁻¹]             | 8.65 × 10⁻⁷       | 2.50 × 10⁻⁶       |
| d₀ [m³ mol⁻¹]             | 2.63 × 10⁻⁸       | 0.00              |
| ΔU<sub>b,i</sub> [J mol⁻¹] | -36,641.21        | -15,800.00        |
| ΔU<sub>d,i</sub> [J mol⁻¹] | -35,690.66        | 0.00              |
| q<sub>sb,i</sub> [mmol g⁻¹] | 3.09              | 5.84              |
| q<sub>sd,i</sub> [mmol g⁻¹] | 2.54              | 0.00              |
""", unsafe_allow_html=True)

with st.expander("🧮 Model Information"):
    st.markdown("""
    1. A dry feed mixture consisting of 15 mol% CO₂ and 85 mol% N₂ was assumed.
    2. The feed pressurisation step was carried out over a fixed duration of 20 seconds.
    3. Isotherm parameters for Zeolite 13X were sourced from the study by R. Haghpanah et al. [1].    
    4. The model was trained on data generated using a mathematical simulation.
    5. The neural network was implemented in PyTorch, with Monte Carlo dropout employed for uncertainty estimation.
    6. The surrogate model achieved an R² value of 0.96, indicating excellent predictive performance.
    """)

# --------------------------
# References Section
# --------------------------
with st.expander("📚 References"):
    st.markdown("[1] R. Haghpanah, A. Majumder, R. Nilam, A. Rajendran, S. Farooq, I. A. Karimi, and M. Amanullah, \"Multiobjective Optimization of a Four-Step Adsorption Process for Postcombustion CO₂ Capture Via Finite Volume Simulation,\" Industrial & Engineering Chemistry Research, vol. 52, no. 11, pp. 4249–4265, 2013.")
    st.markdown("[2] Y. Liao, A. Wright, and J. Li, \"Simulation and optimisation of vacuum (pressure) swing adsorption with simultaneous consideration of real vacuum pump data and bed fluidisation,\" Sep. Purif. Technol., vol. 358, Art. no. 130354, 2025.")
    st.markdown("[3] A. H. Farmahini, S. Krishnamurthy, D. Friedrich, S. Brandani, and L. Sarkisov, \"Performance-based screening of porous materials for carbon capture,\" Chem. Rev., vol. 121, no. 17, pp. 11151–11191, 2021.")
    st.markdown("[4] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,\" in Proceedings of the 33rd International Conference on Machine Learning, 2016.")
