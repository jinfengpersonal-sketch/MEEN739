import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from UIcode import (
    run_optimization,
    plot_optimal_sizes_vs_loadcapex,
    plot_LCOE_and_utilization_vs_loadcapex,
)

# Folder where your 1 MW solar CSVs are stored
SOLAR_FOLDER = "1MW Arrays"
MAP_FOLDER = "maps"

CSV_TO_MAP = {
    "Chile.csv":     "ChileGHI.png",
    "Germany.csv":   "GermanyGHI.png",
    "Texas.csv":     "TexasGHI.png",
}

CSV_TO_GRID_PRICE = {
    "Chile.csv":   190.0,   # $/MWh
    "Germany.csv": 161.0,   # $/MWh
    "Texas.csv":   65.7,    # $/MWh
}

FACILITY_LOAD_CAPEX = {
    "Data Center":       1e7,   # $/MW
    "Electrolysis":      1e6,   # $/MW
    "Aqueduct Pumping":  1e5    # $/MW
}



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
    
    
    with preset_col_1:
        st.subheader("Case Studies")

        st.image("DataCenter.png", width=260)
        if st.button("Analyze Data Center"):
            if "results" not in st.session_state:
                st.warning("Run the optimization first before analyzing a facility.")
            else:
                st.session_state.selected_facility = "Data Center"
                st.session_state.selected_load_capex = FACILITY_LOAD_CAPEX["Data Center"]


        st.image("Aqueduct.png", width=260)
        if st.button("Analyze Aqueduct Pumping Station"):
            if "results" not in st.session_state:
                st.warning("Run the optimization first before analyzing a facility.")
            else:
                st.session_state.selected_facility = "Aqueduct Pumping"
                st.session_state.selected_load_capex = FACILITY_LOAD_CAPEX["Aqueduct Pumping"]


    with preset_col_2:
        st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)

        st.image("electrolysis.png", width=260)
        if st.button("Analyze Electrolysis Plant"):
            if "results" not in st.session_state:
                st.warning("Run the optimization first before analyzing a facility.")
            else:
                st.session_state.selected_facility = "Electrolysis"
                st.session_state.selected_load_capex = FACILITY_LOAD_CAPEX["Electrolysis"]
    
    
    
    
    

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
        
        grid_price = CSV_TO_GRID_PRICE.get(selected_file)

        
        # Decide which map to show based on the selected CSV
        map_filename = CSV_TO_MAP.get(selected_file)
        map_path = None
        if map_filename is not None:
            map_path = os.path.join(MAP_FOLDER, map_filename)
            
        # Store grid price and selected CSV in session_state
        grid_price = CSV_TO_GRID_PRICE.get(selected_file)
        st.session_state.grid_price = grid_price
        st.session_state.selected_csv = selected_file

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

            st.session_state.results = results 
            # RESET any previous case-study selection
            st.session_state.selected_facility = None
            st.session_state.selected_load_capex = None
            
            st.success("Optimization finished.")


        # If we have results from a previous run, always show the plots
        if "results" in st.session_state:
            results = st.session_state.results
            st.subheader("Results")

            highlight_x = st.session_state.get("selected_load_capex", None)

            fig3 = plot_optimal_sizes_vs_loadcapex(results, highlight_x=highlight_x)
            st.pyplot(fig3)

            fig8 = plot_LCOE_and_utilization_vs_loadcapex(results, highlight_load_capex=highlight_x)
            st.pyplot(fig8)
            
            # If a facility has been selected, report utilization and cost at that load capex
            if highlight_x is not None:
                load_costs = np.array(results["load_costs"])
                uptimes = np.array(results["uptimes"])
                total_LCOE = np.array(results["total_LCOE"])
                fuel_part = np.array(results["fuel_part"])

                # find the index of the closest load capex point to the selected value
                idx = int(np.argmin(np.abs(load_costs - highlight_x)))

                display_capex = highlight_x
                chosen_capex = load_costs[idx]
                chosen_util = uptimes[idx]
                chosen_total = total_LCOE[idx]
                
                # get local grid price for this CSV
                grid_price = st.session_state.get("grid_price", None)

                # build comparison text
                if grid_price is not None:
                    if chosen_total < grid_price:
                        verdict = "Microgrid is cheaper than local grid :)"
                    else:
                        verdict = "Microgrid is more expensive than local grid :("
                    grid_line = f"- Local grid cost ≈ {grid_price:.1f} $/MWh\n\n{verdict}"
                else:
                    grid_line = "(No grid price available for this dataset.)"

                st.info(
                    f"For load capex ≈ ${display_capex:,.0f}/MW:\n"
                    f"- Optimal utilization ≈ {chosen_util:.2f}\n"
                    f"- Total cost ≈ {chosen_total:.1f} $/MWh\n"
                    f"{grid_line}"
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
