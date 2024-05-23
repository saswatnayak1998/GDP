import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from scipy.interpolate import make_interp_spline, interp1d
import numpy as np
from io import BytesIO
import tempfile
import requests
from PIL import Image, UnidentifiedImageError
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Function to load data
def load_data():
    df = pd.read_csv("gdp_per_capita.csv")
    df.columns = df.columns.str.strip()  # Clean up column names by stripping any extra spaces
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    df_long = pd.melt(df, id_vars=['Country Name', 'Code'], var_name='Year', value_name='GDP per Capita')
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long

# Function to load flag images using Flags API by the first two letters of the country code
def load_flag_image(country_code):
    country_code = country_code[:2].upper()  # Use only the first two letters of the country code and capitalize them
    url = f"https://flagsapi.com/{country_code}/flat/64.png"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            img = Image.open(BytesIO(response.content))
            return img
        except UnidentifiedImageError:
            st.error(f"Could not identify image for country code: {country_code}")
            return None
    else:
        st.error(f"Failed to fetch flag for country code: {country_code}")
        return None

# Streamlit app
st.title("GDP per Capita Animation with Country Flags")

st.markdown("Refer to [Flags API](https://flagsapi.com/) for country codes.")

# Load data
df_long = load_data()
countries = df_long['Country Name'].unique()

# Placeholder for video path

# Select countries using multiselect
selected_countries = st.multiselect("Select countries to add:", countries)

# Initialize a dictionary to store selected countries and their codes
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = {}

# Form to input country codes
with st.form(key='add_country_form'):
    for country in selected_countries:
        if country not in st.session_state.selected_countries:
            country_code = st.text_input(f"Enter code for {country}:", key=country)
            submit_button = st.form_submit_button(label='Add Country')
            if submit_button and country_code:
                st.session_state.selected_countries[country] = country_code

# Display selected countries and their codes
if st.session_state.selected_countries:
    st.write("Selected countries and codes:", st.session_state.selected_countries)

# Input for video file name
video_file_name = st.text_input("Enter the name for the saved video file (without extension):", "gdp_animation")

# Function to create the animation
def create_animation(selected_countries):
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 10))  # Change figure size for vertical format
    country_flags = {country: load_flag_image(code) for country, code in selected_countries.items()}

    def animate(year):
        ax.clear()
        ax.tick_params(axis='both', colors='black', labelsize=14)  # Set tick labels color and size
        min_y, max_y = float('inf'), float('-inf')
        for country, code in selected_countries.items():
            data = df_long[(df_long['Country Name'] == country) & (df_long['Year'] <= year)]
            if not data.empty:
                # Interpolate data for smoother curves
                years = data['Year']
                gdp = data['GDP per Capita']
                if len(years) > 3:  # Use spline interpolation if there are enough data points
                    years_new = np.linspace(years.min(), years.max(), 300)
                    spl = make_interp_spline(years, gdp, k=3)  # Smoothing spline of degree 3
                    gdp_smooth = spl(years_new)
                    sns.lineplot(x=years_new, y=gdp_smooth, label=country, ax=ax)
                elif len(years) > 1:  # Use linear interpolation if there are enough data points
                    years_new = np.linspace(years.min(), years.max(), 300)
                    f = interp1d(years, gdp, kind='linear')
                    gdp_smooth = f(years_new)
                    sns.lineplot(x=years_new, y=gdp_smooth, label=country, ax=ax)
                else:  # Plot the points directly if there are fewer than two data points
                    sns.lineplot(x=years, y=gdp, label=country, ax=ax)
                
                min_y = min(min_y, gdp.min())
                max_y = max(max_y, gdp.max())
                
                # Add flag image at the end of the curve
                x, y = years.values[-1], gdp.values[-1]
                flag_img = country_flags[country]
                if flag_img:
                    imagebox = OffsetImage(flag_img, zoom=0.3)  # Increase zoom level for larger flags
                    ab = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=(0.5, 0.5))
                    ax.add_artist(ab)
        ax.set_title(f"GDP per Capita Over Time - Up to {year}", fontsize=24, color='black', weight='bold')
        ax.set_xlabel('Year', fontsize=20, color='black', weight='bold')
        ax.set_ylabel('GDP per Capita', fontsize=20, color='black', weight='bold')
        ax.set_xlim(df_long['Year'].min(), df_long['Year'].max())
        if min_y < max_y:
            ax.set_ylim(min_y, max_y)
        ax.legend().set_visible(False)  # Remove legend
    
    ani = animation.FuncAnimation(fig, animate, frames=range(df_long['Year'].min(), df_long['Year'].max() + 1), repeat=False, interval=200)
    return ani

if st.button("Generate Animation") and st.session_state.selected_countries:
    ani = create_animation(st.session_state.selected_countries)
    
    # Save the animation to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpfile:
        ani.save(tmpfile.name, writer='ffmpeg', dpi=80)
        tmpfile.seek(0)
        
        # Read the temporary file into a BytesIO object
        video = BytesIO(tmpfile.read())
        
        # Provide download button
        st.download_button(label="Download Video", data=video, file_name=f"{video_file_name}.mp4")

    st.video(video)
st.write("Example Video")
example_video_path = "gdp_ind.mp4"  # Replace with the actual path to your example video
st.video(example_video_path)

