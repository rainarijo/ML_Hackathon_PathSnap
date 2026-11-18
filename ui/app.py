
import requests
import streamlit as st
import folium

from streamlit_folium import st_folium


# ==========================================
# CONFIGURATION
# ==========================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PathSnap | Smart Map Matching",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #e2e8f0;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background-color: #111c32;
    border-right: 1px solid #263653;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e40af, #075985);
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #31558a;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #cbdffb;
    font-size: 16px;
    margin-top: 9px;
}

.eyebrow {
    color: #67e8f9;
    font-size: 11px;
    letter-spacing: 3px;
    font-weight: 700;
    margin-bottom: 10px;
}

.section-heading {
    color: #f1f5f9;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 16px;
}

.metric-card {
    background: #14213a;
    border: 1px solid #2b3c5c;
    border-radius: 14px;
    padding: 21px;
    min-height: 125px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 12px;
}

.metric-value {
    color: #67e8f9;
    font-size: 30px;
    font-weight: 750;
}

.metric-description {
    color: #64748b;
    font-size: 11px;
    margin-top: 7px;
}

.status-pill {
    display: inline-block;
    background: #064e3b;
    color: #6ee7b7;
    border: 1px solid #047857;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
}

.legend-box {
    background: #14213a;
    border: 1px solid #2b3c5c;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 15px;
}

.legend-item {
    color: #cbd5e1;
    font-size: 14px;
    padding: 5px 0;
}

.legend-line {
    display: inline-block;
    width: 28px;
    height: 4px;
    border-radius: 5px;
    margin-right: 9px;
    vertical-align: middle;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px;
    border-top: 1px solid #263653;
    margin-top: 35px;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 700;
    min-height: 45px;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #1d4ed8, #0e7490);
    border: none;
}

[data-testid="stMetric"] {
    background: #14213a;
    border: 1px solid #2b3c5c;
    padding: 18px;
    border-radius: 13px;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    color: #67e8f9;
}

hr {
    border-color: #263653;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 📍 PathSnap")
    st.caption("SMART GPS MAP MATCHING")

    st.divider()

    st.markdown("### Project Overview")

    st.write(
        "PathSnap transforms raw GPS trajectories "
        "into road-network routes using intelligent "
        "map-matching techniques."
    )

    st.divider()

    st.markdown("### System Status")

    try:
        health = requests.get(API_URL, timeout=5)
        health.raise_for_status()

        st.markdown(
            '<span class="status-pill">● API ONLINE</span>',
            unsafe_allow_html=True
        )

    except requests.RequestException:
        st.error("API OFFLINE")
        st.stop()

    st.divider()

    st.markdown("### Technology Stack")

    st.markdown("""
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    - **Mapping:** Folium
    - **Processing:** Python
    """)

    st.divider()

    st.caption("PathSnap | Hackathon Project")


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">GEOSPATIAL INTELLIGENCE PLATFORM</div>
    <h1>PathSnap</h1>
    <p>
        From raw GPS observations to intelligent road-network
        trajectories.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRIPS
# ==========================================

try:
    response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )
    response.raise_for_status()

    trips = response.json()["trips"]

except requests.RequestException as error:
    st.error(f"Could not retrieve trips: {error}")
    st.stop()


# ==========================================
# TRIP SELECTION
# ==========================================

st.markdown(
    '<div class="section-heading">01 / Select Trajectory</div>',
    unsafe_allow_html=True
)

select_col, info_col = st.columns([3, 1])

with select_col:

    selected_trip = st.selectbox(
        "Choose an evaluator trip",
        trips
    )

with info_col:

    st.metric("Available Trajectories", len(trips))


# Clear old results when the selected trip changes.
if st.session_state.get("last_selected_trip") != selected_trip:

    st.session_state.pop("prediction", None)

    st.session_state["last_selected_trip"] = selected_trip


predict_col, _ = st.columns([1, 3])

with predict_col:

    predict_clicked = st.button(
        "✦  Generate Route Prediction",
        type="primary",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_clicked:

    with st.spinner(
        "Analysing GPS observations and matching road segments..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json={"trip_id": selected_trip},
                timeout=180
            )

            response.raise_for_status()

            st.session_state["prediction"] = response.json()

        except requests.RequestException as error:

            st.error(f"Prediction failed: {error}")


# ==========================================
# DISPLAY RESULTS
# ==========================================

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    st.divider()

    st.markdown(
        f'<div class="section-heading">'
        f'02 / Analysis Results &nbsp; '
        f'<span style="color:#67e8f9;">{result["trip_id"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------
    # MAIN METRICS
    # --------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BASELINE ACCURACY</div>
            <div class="metric-value">
                {result['accuracy_percent']}%
            </div>
            <div class="metric-description">
                Exact reference arc agreement
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">GPS OBSERVATIONS</div>
            <div class="metric-value">
                {result['total_points']:,}
            </div>
            <div class="metric-description">
                Total trajectory points
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CORRECT PREDICTIONS</div>
            <div class="metric-value">
                {result['correct_points']:,}
            </div>
            <div class="metric-description">
                Exact predicted arc matches
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------
    # SECONDARY METRICS
    # --------------------------------------

    c4, c5 = st.columns(2)

    with c4:

        st.metric(
            "Candidate Coverage",
            f"{result['candidate_coverage_percent']}%"
        )

    with c5:

        st.metric(
            "Median Label Distance",
            f"{result['median_label_distance_m']} m"
        )

    st.caption(
        "These are temporary nearest-candidate baseline results. "
        "Final ML/HMM predictions will be integrated separately."
    )

    st.divider()

    # ======================================
    # ROUTE VISUALIZATION
    # ======================================

    st.markdown(
        '<div class="section-heading">03 / Route Visualization</div>',
        unsafe_allow_html=True
    )

    # Clean legend using separate Streamlit columns.
    legend_col1, legend_col2 = st.columns(2)

    with legend_col1:

        st.markdown(
            '<span style="color:#38bdf8;">━━━━</span> '
            'Original GPS trajectory',
            unsafe_allow_html=True
        )

    with legend_col2:

        st.markdown(
            '<span style="color:#fb7185;">━━━━</span> '
            'Predicted road segments',
            unsafe_allow_html=True
        )

    gps_points = result["gps_points"]

    predicted_arcs = result["predicted_arcs"]

    if gps_points:

        gps_coordinates = [
            [p["lat"], p["lon"]]
            for p in gps_points
        ]

        center_lat = sum(
            p["lat"] for p in gps_points
        ) / len(gps_points)

        center_lon = sum(
            p["lon"] for p in gps_points
        ) / len(gps_points)

        # OpenStreetMap does not require a CartoDB API key.
        route_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles="OpenStreetMap"
        )

        # ----------------------------------
        # ORIGINAL GPS TRAJECTORY
        # ----------------------------------

        folium.PolyLine(
            gps_coordinates,
            color="#2563eb",
            weight=4,
            opacity=0.9,
            tooltip="Original GPS trajectory"
        ).add_to(route_map)

        # ----------------------------------
        # PREDICTED ROAD SEGMENTS
        # ----------------------------------

        for arc in predicted_arcs:

            coordinates = [
                [point[1], point[0]]
                for point in arc["coordinates"]
            ]

            folium.PolyLine(
                coordinates,
                color="#e11d48",
                weight=4,
                opacity=0.95,
                tooltip=f"Predicted Arc {arc['arc_id']}"
            ).add_to(route_map)

        # ----------------------------------
        # START MARKER
        # ----------------------------------

        folium.Marker(
            gps_coordinates[0],
            tooltip="Starting Point",
            icon=folium.Icon(
                color="green",
                icon="play"
            )
        ).add_to(route_map)

        # ----------------------------------
        # END MARKER
        # ----------------------------------

        folium.Marker(
            gps_coordinates[-1],
            tooltip="Ending Point",
            icon=folium.Icon(
                color="red",
                icon="stop"
            )
        ).add_to(route_map)

        # Fit map to GPS trajectory.
        route_map.fit_bounds(gps_coordinates)

        # Display map.
        st_folium(
            route_map,
            width=None,
            height=650,
            returned_objects=[]
        )

    else:

        st.warning(
            "No GPS coordinates found for this trajectory."
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    PATHSNAP &nbsp; | &nbsp;
    GEOSPATIAL INTELLIGENCE &nbsp; | &nbsp;
    GPS MAP MATCHING
</div>
""", unsafe_allow_html=True)


# ==========================================
# EMPTY STATE
# ==========================================

if "prediction" not in st.session_state:

    st.info(
        "Choose an evaluator trajectory and generate a prediction "
        "to explore its route and analysis results."
    )
