import streamlit as st
import folium
from streamlit_folium import st_folium
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import itur
import astropy.units as u

# ==========================================
# 1. CONSTANTS & PHYSICAL PARAMETERS
# ==========================================
LAMBDA_NM = 1550             
GAS_ATTEN_DB_KM = 0.15       
C_N2_GROUND = 1e-13          

RX_SENSITIVITY = {
    1: -32.0,
    10: -24.0,
    25: -19.0,
    100: -13.0
}

AVAILABILITIES = [99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999]

# ==========================================
# 2. DATA RETRIEVAL FUNCTIONS
# ==========================================

@st.cache_data(show_spinner=False)
def get_closest_airport(lat, lon):
    url = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    try:
        df = pd.read_csv(url)
        valid_airports = df[
            (df['type'].isin(['medium_airport', 'large_airport'])) & 
            (df['ident'].str.len() == 4)
        ].copy()
        
        R = 6371.0 
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2, lon2 = np.radians(valid_airports['latitude_deg']), np.radians(valid_airports['longitude_deg'])
        
        dphi = lat2 - lat1
        dlambda = lon2 - lon1
        a = np.sin(dphi/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda/2)**2
        valid_airports['distance_km'] = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        closest = valid_airports.loc[valid_airports['distance_km'].idxmin()]
        return closest['ident'], closest['name'], closest['distance_km']
    except Exception as e:
        return "LLBG", "Fallback Airport (Error fetching DB)", 0.0

@st.cache_data(show_spinner=False)
def fetch_metar_visibility(icao_code, years_back=3):
    end_year = datetime.now().year
    start_year = end_year - years_back
    
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={icao_code.upper()}&data=vsby&year1={start_year}&month1=1&day1=1"
        f"&year2={end_year}&month2=1&day2=1&tz=Etc/UTC&format=onlycomma&latlon=no"
        f"&missing=M&trace=T&direct=no&report_type=1&report_type=2"
    )
    
    try:
        df = pd.read_csv(url, na_values=['M'])
        if df.empty or 'vsby' not in df.columns:
            return None
        df['vsby_km'] = pd.to_numeric(df['vsby'], errors='coerce') * 1.60934
        df = df.dropna(subset=['vsby_km'])
        return df['vsby_km'].values
    except Exception as e:
        return None

def get_itu_rain_rate(lat, lon, availability):
    p = 100.0 - availability 
    R = itur.models.itu837.rainfall_rate(lat, lon, p)
    return float(R.value)

def get_percentile_visibility(vis_array, availability):
    if vis_array is None or len(vis_array) == 0:
        return 0.0
    p_outage = 100.0 - availability
    return np.percentile(vis_array, p_outage)

def get_elevation_profile(lat1, lon1, lat2, lon2, num_points=50):
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in zip(lats, lons)]
    try:
        res = requests.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": locations}, timeout=5)
        if res.status_code == 200:
            elevations = [r['elevation'] for r in res.json()['results']]
            return np.array(elevations), lats, lons
    except:
        pass
    return np.zeros(num_points), lats, lons

# ==========================================
# 3. OPTICAL PHYSICS FUNCTIONS
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((phi2-phi1)/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin((math.radians(lon2-lon1))/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calc_geo_loss(d_km, d_tx_m, d_rx_m, theta_mrad):
    if d_km <= 0: return 0
    d_m = d_km * 1000
    beam_diameter = d_tx_m + (theta_mrad / 1000.0) * d_m
    if d_rx_m >= beam_diameter: return 0.0 
    return 20 * math.log10(d_rx_m / beam_diameter) 

def calc_scintillation_margin(d_km, h_m, p_outage=1e-5):
    if d_km <= 0: return 0
    cn2 = C_N2_GROUND * (max(h_m, 1.0))**(-4/3)
    k = 2 * math.pi / (LAMBDA_NM * 1e-9)
    sigma_R2 = 1.23 * cn2 * (k**(7/6)) * ((d_km * 1000)**(11/6))
    return 4.343 * (math.sqrt(-2 * math.log(p_outage)) * math.sqrt(sigma_R2))

def calc_rain_loss(rate_mmhr, d_km):
    return 1.076 * (rate_mmhr ** 0.67) * d_km

def calc_fog_loss(vis_km, d_km):
    if vis_km <= 0: return 999.0
    if vis_km > 50: q = 1.6
    elif vis_km > 6: q = 1.3
    elif vis_km > 1: q = 0.16 * vis_km + 0.34
    elif vis_km > 0.5: q = vis_km - 0.5
    else: q = 0
    return (3.91 / vis_km) * ((LAMBDA_NM / 550) ** -q) * d_km

# ==========================================
# 4. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="FSO Link Planner (Pro)", layout="wide")
st.title("FSO Link Planner: ITU-R & METAR Integration")

if 'site_a' not in st.session_state: st.session_state.site_a = (32.0945, 34.9566)
if 'site_b' not in st.session_state: st.session_state.site_b = (32.1100, 34.9700)
if 'click_mode' not in st.session_state: st.session_state.click_mode = 'A'

st.sidebar.header("1. Location Data")
lat_a = st.sidebar.number_input("Site A Lat", value=st.session_state.site_a[0], format="%.5f")
lon_a = st.sidebar.number_input("Site A Lon", value=st.session_state.site_a[1], format="%.5f")
lat_b = st.sidebar.number_input("Site B Lat", value=st.session_state.site_b[0], format="%.5f")
lon_b = st.sidebar.number_input("Site B Lon", value=st.session_state.site_b[1], format="%.5f")
st.session_state.site_a, st.session_state.site_b = (lat_a, lon_a), (lat_b, lon_b)

mid_lat, mid_lon = (lat_a + lat_b)/2, (lon_a + lon_b)/2

st.sidebar.subheader("Local Weather Station")
with st.spinner("Locating nearest METAR station..."):
    icao_code, airport_name, dist_to_airport = get_closest_airport(mid_lat, mid_lon)

st.sidebar.success(f"**Nearest Airport:** {icao_code}\n\n*{airport_name}*\n\nDist: {dist_to_airport:.1f} km")

st.sidebar.header("2. Hardware Specs")
tx_power = st.sidebar.number_input("Avg Tx Power (dBm)", value=10.0, step=1.0)
tx_apt = st.sidebar.number_input("Tx Aperture (m)", value=0.05, step=0.01)

# --- SMART AUTO-ADAPTING DIVERGENCE LOGIC ---
min_div_rad = 1.22 * (LAMBDA_NM * 1e-9) / tx_apt
min_div_mrad = float(min_div_rad * 1000)

auto_divergence = st.sidebar.checkbox("Auto-calculate Optimal Divergence", value=True)

if auto_divergence:
    # Lock to the physical limit. Changes automatically as tx_apt changes.
    divergence_mrad = min_div_mrad
    st.sidebar.info(f"🔒 Locked to diffraction limit: **{divergence_mrad:.3f} mrad**")
else:
    # Unlock for manual input, but physically block numbers below the limit
    divergence_mrad = st.sidebar.number_input(
        "Manual Beam Divergence (mrad)", 
        min_value=min_div_mrad,
        value=max(0.1, min_div_mrad), 
        step=0.01,
        help=f"Minimum allowed is {min_div_mrad:.3f} mrad based on the {tx_apt}m lens."
    )
# ---------------------------------------------

rx_apt = st.sidebar.number_input("Rx Aperture (m)", value=0.20, step=0.01)
height_m = st.sidebar.number_input("Tower Height AGL (m)", value=25.0, step=5.0)
capacity = st.sidebar.selectbox("Capacity (Gbps)", options=[1, 10, 25, 100], index=1)
modulation = st.sidebar.selectbox("Modulation", options=["OOK (Continuous Wave)", "Femtosecond (SCL)"])

# Interactive Map
col1, col2, _ = st.columns([1, 1, 4])
if col1.button("Set Site A on Click", type="primary" if st.session_state.click_mode == 'A' else "secondary"):
    st.session_state.click_mode = 'A'
if col2.button("Set Site B on Click", type="primary" if st.session_state.click_mode == 'B' else "secondary"):
    st.session_state.click_mode = 'B'

m = folium.Map(location=[mid_lat, mid_lon], zoom
