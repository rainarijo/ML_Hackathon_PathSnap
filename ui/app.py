
import requests
import streamlit as st
import folium

from streamlit_folium import st_folium


# -------------------------------
# CONFIGURATION
# -------------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PathSnap",
    page_icon="📍",
    layout="wide"
)


# -------------------------------
# PAGE HEADER
# -------------------------------

st.title("📍 PathSnap")
st.subheader("GPS Trajectory Map Matching")

st.write(
    "Select a trip to predict its road route "
    "and evaluate the map-matching result."
)


# -------------------------------
# LOAD AVAILABLE TRIPS
# -------------------------------

try:
    response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )
    response.raise_for_status()

    trips = response.json()["trips"]

except requests.RequestException as error:
    st.error(f"Cannot connect to PathSnap API: {error}")
    st.stop()


# -------------------------------
# TRIP SELECTION
# -------------------------------

selected_trip = st.selectbox(
    "Select a trip",
    trips
)


# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("Predict Route", type="primary"):

    with st.spinner("Processing GPS trajectory..."):

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"trip_id": selected_trip},
                timeout=180
            )

            response.raise_for_status()
            result = response.json()

            st.session_state["prediction"] = result

        except requests.RequestException as error:
            st.error(f"Prediction failed: {error}")


# -------------------------------
# DISPLAY PREDICTION
# -------------------------------

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    st.divider()

    st.header(f"Results for Trip {result['trip_id']}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{result['accuracy_percent']}%"
    )

    col2.metric(
        "GPS Points",
        result["total_points"]
    )

    col3.metric(
        "Correct Predictions",
        result["correct_points"]
    )

    st.subheader("Route Visualization")

    gps_points = result["gps_points"]
    predicted_arcs = result["predicted_arcs"]

    if gps_points:

        # Centre map around the GPS trajectory.
        center_lat = sum(p["lat"] for p in gps_points) / len(gps_points)
        center_lon = sum(p["lon"] for p in gps_points) / len(gps_points)

        route_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14
        )

        # Original GPS trajectory.
        gps_coordinates = [
            [p["lat"], p["lon"]]
            for p in gps_points
        ]

        folium.PolyLine(
            gps_coordinates,
            color="blue",
            weight=3,
            opacity=0.7,
            tooltip="Original GPS trajectory"
        ).add_to(route_map)

        # Predicted road segments.
        for arc in predicted_arcs:

            coordinates = [
                [point[1], point[0]]
                for point in arc["coordinates"]
            ]

            folium.PolyLine(
                coordinates,
                color="red",
                weight=4,
                opacity=0.8,
                tooltip=f"Predicted Arc {arc['arc_id']}"
            ).add_to(route_map)

        # Starting and ending points.
        folium.Marker(
            gps_coordinates[0],
            tooltip="Start",
            icon=folium.Icon(color="green")
        ).add_to(route_map)

        folium.Marker(
            gps_coordinates[-1],
            tooltip="End",
            icon=folium.Icon(color="red")
        ).add_to(route_map)

        st_folium(
            route_map,
            width=None,
            height=600,
            returned_objects=[]
        )

    else:
        st.warning("No GPS points available for this trip.")

    st.caption(
        f"Prediction method: {result['prediction_method']}"
    )
