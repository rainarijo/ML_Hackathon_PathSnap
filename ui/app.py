
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# ==================================================
# CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PathSnap | GPS Map Matching",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main application background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #0b1020 !important;
    color: #f1f5f9 !important;
}

/* Fix white header patch */
header[data-testid="stHeader"],
[data-testid="stHeader"],
.stAppHeader {
    background: #0b1020 !important;
    color: #f1f5f9 !important;
}

/* Toolbar */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    background: #0b1020 !important;
}

/* Top-right controls */
[data-testid="stToolbar"] button,
[data-testid="stHeader"] button {
    color: #f1f5f9 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111e35 !important;
}

[data-testid="stSidebar"] > div {
    background-color: #111e35 !important;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* Main headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #f1f5f9 !important;
}

/* Main text */
.stApp p,
.stApp label,
.stApp li {
    color: #e2e8f0;
}

/* Captions */
[data-testid="stCaptionContainer"],
.stCaption {
    color: #94a3b8 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #15243e !important;
    border: 1px solid #2b405f !important;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #91a8c9 !important;
}

[data-testid="stMetricValue"] {
    color: #55e5f4 !important;
}

/* Select box */
[data-baseweb="select"] > div {
    background-color: #15243e !important;
    border: 1px solid #38516f !important;
    color: #ffffff !important;
}

[data-baseweb="select"] * {
    color: #ffffff !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #15243e !important;
}

[data-baseweb="menu"] * {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2860f5, #079bb5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    border: 1px solid #55e5f4 !important;
    color: #ffffff !important;
}

/* Dividers */
hr {
    border-color: #24344f !important;
}

/* Links */
a {
    color: #55e5f4 !important;
}

/* Success and error messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Remove unnecessary top spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* Map container */
iframe {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# API FUNCTIONS
# ==================================================

def check_api():
    try:
        response = requests.get(API_URL, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_trips():
    response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )
    response.raise_for_status()
    return response.json().get("trips", [])


def predict_trip(trip_id):
    response = requests.post(
        f"{API_URL}/predict",
        json={"trip_id": trip_id},
        timeout=300
    )
    response.raise_for_status()
    return response.json()


# ==================================================
# SIDEBAR
# ==================================================

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

    if check_api():
        st.success("● API ONLINE")
    else:
        st.error("● API OFFLINE")

    st.divider()

    st.caption("PathSnap | LightGBM + HMM")


# ==================================================
# MAIN DASHBOARD
# ==================================================

st.markdown("# PathSnap Dashboard")

st.caption(
    "GPS trajectory analysis and intelligent road map matching"
)


# ==================================================
# FETCH AVAILABLE TRIPS
# ==================================================

try:
    trips = get_trips()

    if not trips:
        st.warning("No trajectories found in the dataset.")
        st.stop()

except requests.RequestException as error:
    st.error(f"Could not connect to the PathSnap API: {error}")
    st.stop()


# ==================================================
# TRIP SELECTION
# ==================================================

col1, col2 = st.columns([3, 1])

with col1:

    selected_trip = st.selectbox(
        "Select GPS Trajectory",
        trips,
        format_func=lambda x: str(x)
    )

with col2:

    st.metric(
        "Available Trajectories",
        len(trips)
    )


# Clear stale results when a different trip is selected
if (
    "prediction_result" in st.session_state
    and st.session_state.get("last_selected_trip") != selected_trip
):
    del st.session_state["prediction_result"]


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.markdown("### Generate Route Prediction")

if st.button(
    "✨ Generate Route Prediction",
    use_container_width=False
):

    with st.spinner(
        "Running LightGBM + HMM route prediction..."
    ):

        try:

            result = predict_trip(selected_trip)

            st.session_state["prediction_result"] = result
            st.session_state["last_selected_trip"] = selected_trip

        except requests.RequestException as error:

            st.error(f"Prediction failed: {error}")
            st.stop()


# ==================================================
# DISPLAY RESULTS
# ==================================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.markdown(
        f"""
        ## 02 / Analysis Results
        <span style="color:#55e5f4">
        {result['trip_id']}
        </span>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ML + HMM Accuracy",
        f"{result['accuracy_percent']:.2f}%"
    )

    col2.metric(
        "GPS Observations",
        f"{result['total_points']:,}"
    )

    col3.metric(
        "Correct Predictions",
        f"{result['correct_points']:,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Candidate Coverage",
        f"{result['candidate_coverage_percent']:.2f}%"
    )

    if result["median_label_distance_m"] is not None:

        col5.metric(
            "Median Label Distance",
            f"{result['median_label_distance_m']:.2f} m"
        )

    else:

        col5.metric(
            "Median Label Distance",
            "N/A"
        )

    col6.metric(
        "Route Continuity",
        f"{result['route_continuity'] * 100:.2f}%"
    )

    st.caption(
        f"Prediction Method: {result['prediction_method']} "
        f"| Coordinate System: EPSG:{result['epsg']}"
    )

    st.info(
        "Route predictions are generated using LightGBM "
        "classification and HMM-based Viterbi sequence decoding."
    )

    # ==================================================
    # MAP VISUALIZATION
    # ==================================================

    st.divider()

    st.markdown("## 03 / Matched Route Visualization")

    gps_points = result["gps_points"]
    predicted_arcs = result["predicted_arcs"]

    if not gps_points:

        st.warning("No GPS points available to display.")

    else:

        # Map center
        center_lat = sum(
            point["lat"] for point in gps_points
        ) / len(gps_points)

        center_lon = sum(
            point["lon"] for point in gps_points
        ) / len(gps_points)

        route_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="OpenStreetMap"
        )

        # --------------------------------------------------
        # RAW GPS TRAJECTORY
        # --------------------------------------------------

        gps_coordinates = [
            [point["lat"], point["lon"]]
            for point in gps_points
        ]

        folium.PolyLine(
            gps_coordinates,
            color="#2878ff",
            weight=3,
            opacity=0.85,
            tooltip="Raw GPS Trajectory"
        ).add_to(route_map)

        # --------------------------------------------------
        # PREDICTED ROAD SEGMENTS
        # --------------------------------------------------

        for arc in predicted_arcs:

            coordinates = arc.get("coordinates", [])

            if len(coordinates) == 2:

                # API coordinates: [longitude, latitude]
                segment = [
                    [
                        coordinates[0][1],
                        coordinates[0][0]
                    ],
                    [
                        coordinates[1][1],
                        coordinates[1][0]
                    ]
                ]

                folium.PolyLine(
                    segment,
                    color="#ff3155",
                    weight=5,
                    opacity=0.9,
                    tooltip=f"Predicted Road Arc: {arc['arc_id']}"
                ).add_to(route_map)

        # --------------------------------------------------
        # START MARKER
        # --------------------------------------------------

        folium.Marker(
            gps_coordinates[0],
            tooltip="Starting Point",
            icon=folium.Icon(
                color="green",
                icon="play"
            )
        ).add_to(route_map)

        # --------------------------------------------------
        # END MARKER
        # --------------------------------------------------

        folium.Marker(
            gps_coordinates[-1],
            tooltip="Ending Point",
            icon=folium.Icon(
                color="red",
                icon="stop"
            )
        ).add_to(route_map)

        # Fit map to GPS trajectory
        route_map.fit_bounds(gps_coordinates)

        st_folium(
            route_map,
            use_container_width=True,
            height=550,
            returned_objects=[]
        )

        # --------------------------------------------------
        # MAP LEGEND
        # --------------------------------------------------

        st.markdown("""
        **Map Legend**

        🔵 Blue — Raw GPS trajectory

        🔴 Red — Predicted road segments

        🟢 Green marker — Starting point

        🔴 Red marker — Ending point
        """)

else:

    st.info(
        "Select a trajectory and click Generate Route Prediction "
        "to begin map matching."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        padding:10px;
        font-size:13px;
    ">
        PathSnap | Intelligent GPS Map Matching
        <br>
        Powered by LightGBM + Hidden Markov Model
    </div>
    """,
    unsafe_allow_html=True
)
