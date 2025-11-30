import streamlit as st
import pandas as pd
import requests
from PIL import Image
import io

@st.cache_data
def load_data():
    df = pd.read_csv('profiles.csv')
    # Fix any column name issues if needed, e.g., 'perefence' to 'preference'
    if 'perefence' in df.columns:
        df = df.rename(columns={'perefence': 'preference'})
    return df

def display_profile(profile):
    col1, col2 = st.columns(2)
    with col1:
        try:
            response = requests.get(profile['profilePicture'], timeout=10)
            img = Image.open(io.BytesIO(response.content))
            st.image(img, width=200)
        except:
            st.image("https://via.placeholder.com/200", width=200)  # Fallback
    with col2:
        st.metric("Name", profile['name'])
        st.metric("Age", profile['age'])
        st.metric("Occupation", profile['occupation'])
        st.write(f"**Gender:** {profile['gender']}")
        st.write(f"**Height:** {profile['height']}")
        st.write(f"**Education:** {profile['education']}")
        st.write(f"**Star Sign:** {profile['starSign']}")
        st.write("**Interests:**")
        for interest in profile['interests'].split(', '):
            st.write(f"- {interest}")
    st.write("**Bio:**")
    st.write(profile['bio'])
    st.write("**Looking For:**")
    st.write(profile['lookingFor'])

def main():
    st.title("Profiles Demo App")
    df = load_data()
    
    # Display all profiles
    for idx, row in df.iterrows():
        st.subheader(f"Profile {idx + 1}")
        display_profile(row)
        st.divider()

if __name__ == "__main__":
    main()
