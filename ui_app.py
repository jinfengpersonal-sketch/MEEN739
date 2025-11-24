import os
import streamlit as st
import matplotlib.pyplot as plt

from UIcode import (
    run_optimization,
    plot_intraday_power,
    plot_mean_daily_power,
    plot_optimal_sizes_vs_loadcapex,
    plot_cost_vs_utilization,
    plot_cost_vs_loadcapex,
    plot_utilization_vs_loadcapex,
    plot_LCOE_vs_utilization,
    plot_LCOE_and_utilization_vs_loadcapex,
)

# Folder where your 1 MW solar CSVs are stored
SOLAR_FOLDER = "1MW Arrays"
MAP_FOLDER = "maps"

CSV_TO_MAP = {
    "Chile.csv":     "ChileGHI.png",
    "Germany.csv":   "GermanyGHI.png",
    "Texas.csv":     "TexasGHI.png",
    "China.csv":     "ChinaGHI.png",
    "Mexico.csv":    "MexicoGHI.png",
    "Pakistan.csv":  "PakistanGHI.png",
    "Qatar.csv":     "Qatar.png"
}

# Example preset for a "data center" scenario
DATA_CENTER_PRESET = {
    "battery_cost": 220_000.0,
    "solar_cost": 800_000.0,
    "gen_capex": 1_000_000.0,
    "fuel_cost": 35.0,
    "include_generator": True,
}

AQUEDUCT_PRESET = {
    "battery_cost": 230_000.0,
    "solar_cost": 850_000.0,
    "gen_capex": 1_100_000.0,
    "fuel_cost": 40.0,
    "include_generator": True,
}

ELECTRO_PRESET = {
    "battery_cost": 240_000.0,
    "solar_cost": 860_000.0,
    "gen_capex": 1_200_000.0,
    "fuel_cost": 41.0,
    "include_generator": True,
}

def apply_data_center_preset():
    preset = DATA_CENTER_PRESET
    st.session_state["battery_cost"] = preset["battery_cost"]
    st.session_state["solar_cost"] = preset["solar_cost"]
    st.session_state["gen_capex"] = preset["gen_capex"]
    st.session_state["fuel_cost"] = preset["fuel_cost"]
    st.session_state["include_generator"] = preset["include_generator"]
    
def apply_aqueduct_preset():
    preset = AQUEDUCT_PRESET
    st.session_state["battery_cost"] = preset["battery_cost"]
    st.session_state["solar_cost"] = preset["solar_cost"]
    st.session_state["gen_capex"] = preset["gen_capex"]
    st.session_state["fuel_cost"] = preset["fuel_cost"]
    st.session_state["include_generator"] = preset["include_generator"]

def apply_electro_preset():
    preset = ELECTRO_PRESET
    st.session_state["battery_cost"] = preset["battery_cost"]
    st.session_state["solar_cost"] = preset["solar_cost"]
    st.session_state["gen_capex"] = preset["gen_capex"]
    st.session_state["fuel_cost"] = preset["fuel_cost"]
    st.session_state["include_generator"] = preset["include_generator"]

# Page config + slightly wider centered block
st.set_page_config(layout="centered", page_title="Solar + Battery + Generator Sizing Tool")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;     /* default ~700px */
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_solar_files(folder: str):
    """Return list of CSV files inside the solar folder."""
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(".csv")]


def main():
    # Initialize defaults in session_state once
    if "battery_cost" not in st.session_state:
        st.session_state["battery_cost"] = 200_000.0

    if "solar_cost" not in st.session_state:
        st.session_state["solar_cost"] = 600_000.0

    if "gen_capex" not in st.session_state:
        st.session_state["gen_capex"] = 800_000.0

    if "fuel_cost" not in st.session_state:
        st.session_state["fuel_cost"] = 40.0

    if "include_generator" not in st.session_state:
        st.session_state["include_generator"] = True

    st.title("Solar + Battery + Generator Sizing Tool")
    st.write("Simple UI wrapper around the optimization model in `UIcode.py`.")

    # Overall three-column layout:
    # left_spacer (mostly empty), main_col (form + plots), preset_col (presets)
    left_spacer, main_col, preset_col_1, preset_col_2 = st.columns([1.5, 3, 1, 1])

    # ------------------------------------------------------------------
    # MAIN COLUMN: existing UI (inputs + run + plots)
    # ------------------------------------------------------------------
    with main_col:
        # --- Solar time series selection ---
        st.header("1. Solar input data")

        solar_files = get_solar_files(SOLAR_FOLDER)
        if not solar_files:
            st.error(f"No CSV files found in folder: {SOLAR_FOLDER}")
            st.stop()

        selected_file = st.selectbox("Select a 1 MW solar CSV", solar_files)
        solar_path = os.path.join(SOLAR_FOLDER, selected_file)
        
        # Decide which map to show based on the selected CSV
        map_filename = CSV_TO_MAP.get(selected_file)
        map_path = None
        if map_filename is not None:
            map_path = os.path.join(MAP_FOLDER, map_filename)

        # --- Economic inputs ---
        st.header("2. Economic parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="font-weight:600; font-size:0.95rem;">
                    Battery cost ($/MWh)
                </div>
                <div style="color:#aaaaaa; font-size:0.8rem; margin-bottom:0.25rem;">
                    Capital cost of battery storage per MWh of usable energy capacity.
                    Bounds: 200,000–300,000 $/MWh.
                </div>
                """,
                unsafe_allow_html=True,
            )
            battery_cost = st.number_input(
                "",
                min_value=200_000.0,
                max_value=300_000.0,
                step=10_000.0,
                format="%.0f",
                key="battery_cost",
            )

            st.markdown(
                """
                <div style="font-weight:600; font-size:0.95rem; margin-top:2rem;">
                    Solar cost ($/MW)
                </div>
                <div style="color:#aaaaaa; font-size:0.8rem; margin-bottom:0.25rem;">
                    Installed cost of 1 MW of solar PV capacity.
                    Bounds: 600,000–1,200,000 $/MW.<br><br><br>
                </div>
                """,
                unsafe_allow_html=True,
            )
            solar_cost = st.number_input(
                "",
                min_value=600_000.0,
                max_value=1_200_000.0,
                step=50_000.0,
                format="%.0f",
                key="solar_cost",
            )

        with col2:
            st.markdown(
                """
                <div style="font-weight:600; font-size:0.95rem;">
                    Generator capex ($/MW)
                </div>
                <div style="color:#aaaaaa; font-size:0.8rem; margin-bottom:0.25rem;">
                    Installed capital cost of backup generator capacity per MW.
                    Bounds: 200,000–3,000,000 $/MW.
                </div>
                """,
                unsafe_allow_html=True,
            )
            gen_capex = st.number_input(
                "",
                min_value=200_000.0,
                max_value=3_000_000.0,
                step=50_000.0,
                format="%.0f",
                key="gen_capex",
            )

            st.markdown(
                """
                <div style="font-weight:600; font-size:0.95rem; margin-top:2rem;">
                    Fuel cost ($/MWh)
                </div>
                <div style="color:#aaaaaa; font-size:0.8rem; margin-bottom:0.25rem;">
                    Variable fuel + O&amp;M cost per MWh of generator electricity.
                    Bounds: 10–60 $/MWh.<br>
                    (Example: ≈15 $/MWh for U.S. natural gas, ≈35 $/MWh for Europe/LNG,
                    plus ≈5 $/MWh for generator O&amp;M.)
                </div>
                """,
                unsafe_allow_html=True,
            )
            fuel_cost = st.number_input(
                "",
                min_value=10.0,
                max_value=60.0,
                step=1.0,
                format="%.1f",
                key="fuel_cost",
            )

        include_generator = st.checkbox(
            "Include generator",
            key="include_generator",
        )

        # --- Run optimization + plots ---
        st.header("3. Run optimization")

        if st.button("Run model"):
            with st.spinner("Running optimization..."):
                results = run_optimization(
                    solar_array_file=solar_path,
                    battery_cost=battery_cost,
                    solar_cost=solar_cost,
                    gen_capex=gen_capex,
                    fuel_cost=fuel_cost,
                    include_generator=include_generator,
                )

            st.success("Optimization finished.")

            st.subheader("Solar profile plots")
            fig1 = plot_intraday_power(results)
            st.pyplot(fig1)

            fig2 = plot_mean_daily_power(results)
            st.pyplot(fig2)

            st.subheader("System sizing and cost plots")
            fig3 = plot_optimal_sizes_vs_loadcapex(results)
            st.pyplot(fig3)

            fig4 = plot_cost_vs_utilization(results)
            st.pyplot(fig4)

            fig5 = plot_cost_vs_loadcapex(results)
            st.pyplot(fig5)

            fig6 = plot_utilization_vs_loadcapex(results)
            st.pyplot(fig6)

            st.subheader("LCOE and utilization plots")
            fig7 = plot_LCOE_vs_utilization(results)
            st.pyplot(fig7)

            fig8 = plot_LCOE_and_utilization_vs_loadcapex(results)
            st.pyplot(fig8)

    # ------------------------------------------------------------------
    # PRESET COLUMN: always-visible data center preset
    # ------------------------------------------------------------------
    with preset_col_1:
        st.subheader("Presets")

        # If you have an image file, put it next to ui_app.py and uncomment:
        st.image("DataCenter.png", width=260)

        st.button(
        "Load Data Center Preset",
        key="preset_button",
        on_click=apply_data_center_preset,
        )
        
        st.image("Aqueduct.png", width=260)

        st.button(
        "Load Aquedcut Preset",
        key="aqueduct_preset_button",
        on_click=apply_aqueduct_preset,
        )


        
        
    with preset_col_2:
        st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)  # small vertical top spacer

        st.image("electrolysis.png", width=260)

        st.button(
        "Load Electrolysis Preset",
        key="electrolysis_preset_button",
        on_click=apply_aqueduct_preset,
        )

    with left_spacer:
        if map_path and os.path.isfile(map_path):
            st.subheader("Solar resource")
            st.image(map_path, width=300)  # tweak width as needed
        else:
            st.subheader("Solar resource")
            st.write("No map available for this dataset.")


if __name__ == "__main__":
    main()
