import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Outbreak Investigator",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Courier Prime', monospace;
    background-color: #0b0e13;
    color: #c8d6b9;
}

.stApp {
    background-color: #0b0e13;
}

h1, h2, h3 {
    font-family: 'Share Tech Mono', monospace;
    color: #7fba5a;
    letter-spacing: 0.08em;
}

.top-bar {
    background: #0f1318;
    border-bottom: 1px solid #7fba5a33;
    padding: 1.2rem 2rem 0.8rem;
    margin-bottom: 1.5rem;
}
.top-bar h1 {
    font-size: 1.4rem;
    margin: 0;
    color: #7fba5a;
}
.top-bar .sub {
    font-size: 0.72rem;
    color: #4a5e38;
    letter-spacing: 0.18em;
    margin-top: 2px;
}

.metric-card {
    background: #0f1318;
    border: 1px solid #7fba5a33;
    border-left: 3px solid #7fba5a;
    padding: 0.9rem 1.1rem;
    border-radius: 2px;
    margin-bottom: 0.6rem;
}
.metric-card .label {
    font-size: 0.65rem;
    color: #4a5e38;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.metric-card .value {
    font-size: 1.6rem;
    color: #b5e48c;
    font-family: 'Share Tech Mono', monospace;
}
.metric-card .delta {
    font-size: 0.7rem;
    color: #7fba5a99;
}

.poi-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #1e2a17;
    font-size: 0.8rem;
}
.poi-row:last-child { border-bottom: none; }
.poi-badge {
    background: #1e2a17;
    border: 1px solid #7fba5a44;
    border-radius: 2px;
    padding: 1px 6px;
    font-size: 0.65rem;
    color: #7fba5a;
    letter-spacing: 0.1em;
}

.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: #4a5e38;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e2a17;
}

div[data-testid="stSlider"] > label {
    color: #7fba5a !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
}

div[data-testid="stCheckbox"] label {
    color: #c8d6b9 !important;
    font-size: 0.8rem !important;
}

.stSlider > div > div > div > div {
    background: #7fba5a !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <h1>⬡ OUTBREAK INVESTIGATOR // WUHAN 2019</h1>
  <div class="sub">EPIDEMIOLOGICAL SOURCE TRACING · SYNTHETIC CASE DATA · RESTRICTED</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Simulation Parameters</div>', unsafe_allow_html=True)

    total_cases = st.slider("Total cases", 100, 1000, 500, 50)
    cluster_pct = st.slider("Cluster % (near source)", 10, 90, 70, 5)
    spread_sigma = st.slider("Cluster spread (km)", 1, 20, 5) * 0.001

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Map Options</div>', unsafe_allow_html=True)
    show_heatmap   = st.checkbox("Heatmap", value=True)
    show_pois      = st.checkbox("Points of Interest", value=True)
    show_source    = st.checkbox("Reveal true source", value=False)

    heatmap_radius = st.slider("Heat radius", 5, 25, 12)
    heatmap_blur   = st.slider("Heat blur", 5, 30, 15)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Map Style</div>', unsafe_allow_html=True)
    tile_choice = st.selectbox(
        "Tile layer",
        ["cartodbdark_matter", "cartodbpositron", "openstreetmap"],
        index=0,
    )

# ── Data generation ────────────────────────────────────────────────────────────
market_lat, market_lon = 30.6195, 114.2577

cluster_count = int(total_cases * cluster_pct / 100)
noise_count   = total_cases - cluster_count

np.random.seed(42)
cluster_lats = np.random.normal(market_lat, spread_sigma, cluster_count)
cluster_lons = np.random.normal(market_lon, spread_sigma, cluster_count)
noise_lats   = np.random.uniform(30.50, 30.70, noise_count)
noise_lons   = np.random.uniform(114.20, 114.40, noise_count)

df_cases = pd.DataFrame({
    'case_id': range(total_cases),
    'lat': np.concatenate([cluster_lats, noise_lats]),
    'lon': np.concatenate([cluster_lons, noise_lons]),
    'type': ['clustered'] * cluster_count + ['scattered'] * noise_count,
})

df_pois = pd.DataFrame({
    'name': ['Wuhan International Plaza', 'Huanan Seafood Market', 'Hankou Railway Station', 'Wuhan CDC'],
    'lat':  [30.584,   30.6195, 30.618, 30.612],
    'lon':  [114.271,  114.2577, 114.250, 114.265],
    'true_source': [False, True, False, False],
})

# ── Layout ─────────────────────────────────────────────────────────────────────
left_col, map_col = st.columns([1, 3])

# ── Left panel ─────────────────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-label">Case Statistics</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Total Cases</div>
        <div class="value">{total_cases:,}</div>
    </div>
    <div class="metric-card">
        <div class="label">Clustered (source-linked)</div>
        <div class="value">{cluster_count:,}</div>
        <div class="delta">{cluster_pct}% of total</div>
    </div>
    <div class="metric-card">
        <div class="label">Scattered (community)</div>
        <div class="value">{noise_count:,}</div>
        <div class="delta">{100 - cluster_pct}% of total</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Points of Interest</div>', unsafe_allow_html=True)
    for _, poi in df_pois.iterrows():
        badge = "SOURCE" if (poi['true_source'] and show_source) else "SUSPECT"
        badge_color = "#b5e48c" if (poi['true_source'] and show_source) else "#7fba5a"
        st.markdown(f"""
        <div class="poi-row">
            <span style="font-size:0.78rem">{poi['name']}</span>
            <span class="poi-badge" style="color:{badge_color}">{badge}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Investigation Note</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#4a5e38; line-height:1.6em">
    Each marker represents a candidate origin site.
    Heatmap intensity reflects case density.
    Toggle "Reveal true source" to check your hypothesis.
    <br><br>
    <em>All case data is synthetic.</em>
    </div>
    """, unsafe_allow_html=True)

# ── Map ─────────────────────────────────────────────────────────────────────────
with map_col:
    m = folium.Map(location=[30.61, 114.28], zoom_start=13, tiles=tile_choice)

    if show_heatmap:
        heat_data = df_cases[['lat', 'lon']].values.tolist()
        HeatMap(heat_data, radius=heatmap_radius, blur=heatmap_blur).add_to(m)

    if show_pois:
        for _, poi in df_pois.iterrows():
            if poi['true_source'] and show_source:
                icon_color, icon_name = 'red', 'exclamation-sign'
                label = f"★ TRUE SOURCE: {poi['name']}"
            else:
                icon_color, icon_name = 'black', 'question-sign'
                label = poi['name']

            folium.Marker(
                location=[poi['lat'], poi['lon']],
                popup=folium.Popup(label, max_width=220),
                tooltip=label,
                icon=folium.Icon(color=icon_color, icon=icon_name),
            ).add_to(m)

    st_folium(m, width="100%", height=560, returned_objects=[])