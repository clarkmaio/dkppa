import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import os
import polars as pl
import matplotlib.pyplot as plt 
from huggingface_hub import HfFileSystem

from src.utils import normalized_logistic, download_era5_dataset


# ---------------------------------------------------------
# Setup Functions
# ---------------------------------------------------------

#@st.cache_resource
def ensure_data_loaded():
    """Ensure that ERA5 data is downloaded locally on the first run."""
    dataset_path = 'dataset/era5'
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        download_era5_dataset()



@st.cache_data
def load_price_scenario():
    """Load price scenario data from HuggingFace."""
    fs = HfFileSystem()
    folder_path = "datasets/clarkmaio/dkppa/price/scenario"
    files = [f"hf://{path}" for path in fs.glob(f"{folder_path}/*.parquet")]
    df = pl.scan_parquet(files).collect().with_columns(
        ordinal_day = pl.col('time').dt.ordinal_day(),
    )
    return df


@st.cache_data
def load_point_weather(lat: float, lon: float):
    """Load weather data for a specific point from local dataset."""
    basepath = 'dataset/era5/*.parquet'
    df = (
        pl.scan_parquet(basepath)
        .filter(pl.col('latitude') == lat)
        .filter(pl.col('longitude') == lon)
        .with_columns(
            ordinal_day = pl.col('time').dt.ordinal_day(), 
            month = pl.col('time').dt.month(),
            s100 = (pl.col('u100')**2 + pl.col('v100')**2)**0.5,
            scenario = pl.col('time').dt.year()
        )
        .collect()
    )
    return df


def compute_fairprice_stats(price: pl.DataFrame, weather: pl.DataFrame):
    """Compute PPA fair price and other statistics for each scenario."""
    # join price and weather on ordinal_day and scenario
    df = price.join(weather, on=['ordinal_day', 'scenario'], how='left')
    
    # group by scenario and aggregate
    stats = df.group_by('scenario').agg(
        fairprice = (pl.col('price') * pl.col('energy')).sum() / pl.col('energy').sum(),
        s100 = pl.col('s100').mean(),
        price = pl.col('price').mean(),
        energy = pl.col('energy').sum()
    ).drop_nulls()
    
    return stats


# ---------------------------------------------------------
# UI Components
# ---------------------------------------------------------

def init_page():
    st.set_page_config(layout="wide", page_title="Denmark Map Dashboard", initial_sidebar_state="collapsed")
    st.markdown("<h1 style='text-align: center;'>dkppa</h1>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        capacity = st.slider(label='Capacity [Mw]', min_value=1, max_value=100, value=10, step=1)
    return capacity


def create_base_map():
    """Create the base folium map"""


    m = folium.Map(location=[56.15, 10.2], zoom_start=6)
    m.add_child(folium.LatLngPopup())
    
    # Add a rectangle around Denmark
    # Coordinates: [South, West], [North, East]
    MIN_LAT=54.2499
    MAX_LAT=58.001
    MIN_LON=7.499999
    MAX_LON=13.001
    denmark_bounds = [[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]]
    folium.Rectangle(
        bounds=denmark_bounds,
        color="red",
        weight=2,
        fill=False,
    ).add_to(m)
    
    return m


def render_metrics(fairprice_df, weather_df, columns):
    """Render high-level metrics."""
    c1, c2, c3 = columns
    
    with c1:
        st.metric(
            label="Average Fair Price", 
            value=f"{fairprice_df['fairprice'].mean():.1f} EUR/MWh", 
            border=True
        )
        
    with c2:
        # Energy is aggregated by scenario, so we take the mean of these sums
        avg_yearly_gen = weather_df.group_by('scenario').agg(pl.col('energy').sum())['energy'].mean()
        st.metric(
            label="Average Yearly Generation", 
            value=f"{avg_yearly_gen/1000:.1f} GWh", 
            border=True
        )


def render_plots(fairprice_df, weather_df):
    """Render the dashboard plots."""
    fig = plt.figure(figsize=(10, 8))
    
    # 1. Histogram of Yearly Average Price
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(fairprice_df['fairprice'], bins=10, color='skyblue', edgecolor='black')
    ax1.set_title('Yearly Average Fair Price Distribution')
    ax1.set_xlabel('Average Price (EUR/MWh)')
    ax1.set_ylabel('Frequency')
    
    # 2. Avg s100 vs Avg Price
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(fairprice_df['s100'], fairprice_df['fairprice'], color='orange', alpha=0.6)
    ax2.set_title('Wind Speed vs Fair Price')
    ax2.set_xlabel('Average s100')
    ax2.set_ylabel('Average Price (EUR/MWh)')
    
    # 3. Monthly Energy Profiles
    ax3 = plt.subplot(2, 1, 2)
    weather_monthly = weather_df.group_by(['month', 'scenario']).agg(pl.col('energy').sum()).sort('month')
    
    for s in weather_monthly['scenario'].unique().to_list():
        df_s = weather_monthly.filter(pl.col('scenario') == s).sort('month')
        ax3.plot(df_s['month'], df_s['energy'], color='gray', alpha=0.2)
    
    # Highlight the mean profile
    mean_profile = weather_monthly.group_by('month').agg(pl.col('energy').mean()).sort('month')
    ax3.plot(mean_profile['month'], mean_profile['energy'], color='red', linewidth=2, label='Mean Profile')
    
    ax3.set_title('Monthly Energy Profiles (All Scenarios)')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Total Energy (MWh)')
    ax3.legend()
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------

def main():
    ensure_data_loaded()
    init_page()
    capacity = render_sidebar()
    
    tab_tool, tab_method = st.tabs(["Pricing Tool", "Methodology"])
    
    with tab_tool:
        metrics_columns = st.columns(3)

        col_map, col_viz = st.columns([1, 1])
        
        with col_map:
            m = create_base_map()
            map_data = st_folium(
                m,
                width=700,
                height=500,
                key="main_map",
                returned_objects=["last_clicked"],
            )
        
        # Clicked location
        lat, lng = 56.0, 10.0 # Default location
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
        
        # Grid resolution snapping (0.25 deg)
        lat_res = round(lat / 0.25) * 0.25
        lng_res = round(lng / 0.25) * 0.25
        
        # Load and Process Data
        # Using placeholder data if no click yet? streamlit rerun handles this.
        with st.spinner("Loading scenarios and weather data..."):
            price_df = load_price_scenario()
            weather_df = load_point_weather(lat_res, lng_res)
            
            # Application Logic: Windspeed to Energy
            weather_df = weather_df.with_columns(
                energy = normalized_logistic(pl.col('s100'), saturation=35, steep=-10) * capacity
            )
            
            # Compute PPA stats
            fairprice_df = compute_fairprice_stats(price_df, weather_df)
        

        # Visualizations
        with col_viz:
            render_plots(fairprice_df, weather_df)
            
        with st.expander("Show Raw Data"):
            st.subheader("Fair Price Statistics by Scenario")
            st.dataframe(fairprice_df)
            st.subheader("Monthly Weather/Energy Summary")
            st.dataframe(weather_df.head(100))

    render_metrics(fairprice_df, weather_df, metrics_columns)


    with tab_method:
        st.header("Methodology")
        st.markdown(r"""
        ### PPA Fair Price Calculation
        The headline PPA fair price ($p^*$) is obtained in **two steps**:
        a per-scenario fair price is computed first, and the reported value
        is the **average of those per-scenario fair prices** across all
        weather scenarios.

        **Step 1 — Fair price per scenario.** For each scenario $s$ (i.e. each
        historical weather year used to drive prices and generation), the
        fair price is the volume-weighted average of the daily price, with
        the daily energy generation as the weight:

        $$
        p^*_s = \frac{\sum_d \left( p_{s,d} \cdot w_{s,d} \right)}{\sum_d w_{s,d}}
        $$

        Where:
        - $p_{s,d}$ is the modeled day-ahead price for scenario $s$ on day $d$.
        - $w_{s,d}$ is the energy generation for scenario $s$ on day $d$ at
          the clicked grid point.

        **Step 2 — Average across scenarios.** The final fair price displayed
        in the dashboard is the simple mean of the per-scenario fair prices:

        $$
        p^* = \frac{1}{N_s} \sum_{s=1}^{N_s} p^*_s
        $$

        where $N_s$ is the number of weather scenarios. This makes $p^*$ an
        estimate of the *expected* PPA fair price under the historical
        distribution of weather years.

        ### Energy Generation Model
        Wind speed at 100m is converted to energy using a normalized logistic curve:
        - **Saturation:** 35 m/s
        - **Steepness:** -10

        ### Price Scenario Construction
        Each scenario represents a counterfactual answer to the question:
        *"What would today's prices look like if the weather of year $Y$ were replayed?"*

        The procedure has two stages:

        **1. Fit a daily day-ahead price model.** Using historical DK_1 day-ahead
        prices (2018–2024) joined with ERA5 weather, an `LpRegressor` is fitted
        with the additive structure:

        $$
        \hat{p}_d = \mathrm{cs}(\text{ordinal\_day}) + f(\text{weekday}) + f(\text{year}) + \beta \cdot s_{100,d}
        $$

        where $\mathrm{cs}$ is a cyclic spline on day-of-year, $f(\cdot)$ are
        categorical factors, and $s_{100,d}$ is the daily spatial-mean wind
        speed at 100 m over Denmark.

        **2. Generate one scenario per historical weather year.** For each year
        $Y$ in the ERA5 archive, the daily features $(\text{ordinal\_day},
        \text{weekday}, s_{100,d})$ are taken from that year's actual weather,
        but the calendar `year` feature is **forced to the baseline year 2024**.
        This freezes the year-level price effect at the most recent observed
        level so that scenarios differ *only* through weather, not through
        underlying price drift.

        The fitted model then predicts a full daily price series for each
        scenario, which the dashboard joins with the clicked-location energy
        profile to produce the distribution of fair prices shown above.
        """)

if __name__ == "__main__":
    main()
