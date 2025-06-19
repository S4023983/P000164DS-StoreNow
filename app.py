#This is a Streamlit app that integrates multiple functionalities for analyzing rooftops.

# library imports
import streamlit as st
from PIL import Image
import os

# Importing necessary functions from the existing scripts.
try:
    from image_classification_estimator import (
        fetch_satellite_view,
        classify_roof,
        get_best_panel_orientation,
        azimuth_to_direction,
        GOOGLE_API_KEY
    )
    from roof_obstruction_estimator import run_pipeline
except ImportError as e:
    st.error(f"Failed to import necessary modules. Make sure all script files are in the same directory. Error: {e}")
    st.stop()

# Set up the Streamlit app configuration
st.set_page_config(page_title="Roof AI Dashboard", layout="wide")
st.title("🏠 Integrated Rooftop Analysis Dashboard")

# --- User Input for coordinates ---
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = "52.43464, 9.73011"  # Default sample coordinates

st.session_state.coordinates = st.text_input(
    "Enter coordinates (latitude, longitude):",
    st.session_state.coordinates,
    help="Example: 52.43464, 9.73011"
)
# --- User Input for roof angle ---
roof_angle = st.slider(
    "Confirm Roof Pitch/Tilt (degrees) for accurate area calculation:", 
    min_value=0, max_value=60, value=25,
    help="Adjust this slider to match the roof's actual tilt."
)

if st.button("Analyze Location"):
    try:
        # Parse user coordinates input; Checking for valid coordinate format otherwise show error
        try:
            lat_str, lon_str = st.session_state.coordinates.split(",")
            lat = float(lat_str.strip())
            lng = float(lon_str.strip())
        except ValueError:
            st.error("Invalid coordinate format! Use: latitude, longitude (e.g. 52.43464, 9.73011)")
            st.stop()
        # Fetches Satellite Image from Google Static Maps API
        with st.spinner("Fetching satellite image..."):
            img_bytes = fetch_satellite_view(lat, lng, GOOGLE_API_KEY)
            st.success(f"Successfully retrieved satellite image for: {lat:.6f}, {lng:.6f}")

        #Uses the classification model to predict roof type and displays the predicted roof type (e.g., "Flat Roof", "Gabled Roof", etc.)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Satellite View & Roof Classification")
            st.image(img_bytes, caption="Satellite View", use_container_width=True)
            roof_type = classify_roof(img_bytes)
            st.metric(label="Predicted Roof Type", value=roof_type)


        # Calls get_best_panel_orientation() function that uses Google Solar API and extracts the important metrics and displays them.

        with col2:
            st.subheader("Solar API & Orientation Insights")
            try:
                tilt, azimuth, area = get_best_panel_orientation(f"{lat},{lng}")  # pass coordinates string
                direction = azimuth_to_direction(azimuth)
                st.metric(label="Optimal Panel Tilt", value=f"{tilt:.1f}°")
                st.metric(label="Best Panel Direction", value=f"{direction} ({azimuth:.1f}°)")
                st.info(f"**Google Solar API Estimated Usable Area:** `{area:.1f} m²`")
            except Exception as e:
                st.warning(f"Could not retrieve Google Solar API data. Error: {e}")


        # Calls the main pipeline run_pipeline() which downloads the image, segments the roof area (DeepLabV3+), 
        # runs SAM model to detect obstructions, calculates adjusted roof area, obstruction area, usable area.
        st.divider()

        st.subheader("Deep Learning Obstruction & Area Analysis")
        with st.spinner("Running segmentation models..."):
            analysis_results = run_pipeline(lat, lng, roof_angle)

        # Display the final annotated image where roof boundary (green) and obstructions (blue) are marked.
        if analysis_results:
            st.success("Segmentation analysis complete!")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Total Adjusted Roof Area", f"{analysis_results['adjusted_roof_area']:.2f} m²")
                st.metric("Detected Obstruction Area", f"{analysis_results['obstruction_area']:.2f} m²")
                st.metric("Final Calculated Usable Area", f"{analysis_results['usable_area']:.2f} m²")

            with res_col2:
                if os.path.exists(analysis_results['output_image_path']):
                    final_image = Image.open(analysis_results['output_image_path'])
                    st.image(final_image, caption="Annotated Roof with Obstructions", use_container_width=True)
                else:
                    st.error("Could not load the final annotated image.")
        else:
            st.error("The roof obstruction analysis failed. The model may not have detected a valid roof in the image.")

    except Exception as e:
        st.error(f"An unexpected error occurred during the analysis: {e}")






