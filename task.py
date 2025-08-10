import pickle
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Load the preprocessor and model
try:
    with open('preprocessor.pkl', 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    with open('xgboost_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please ensure both 'xgboost_model.pkl' and 'preprocessor.pkl' exist.")
    st.stop()

# Define options for dropdown menus
locations = ["India", "United States", "Germany", "Indonesia", "Brazil", "Japan", "Mexico", "Philippines", "Pakistan", "Vietnam"]
platforms = ["YouTube", "Facebook", "Instagram", "TikTok"]
video_categories = ["Gaming", "Comedy", "Vlogs", "Entertainment", "ASMR", "Trends", "Pranks", "Life Hacks", "Jokes/Memes"]
engagement_levels = ["high", "moderate", "less"]
frequencies = ["Morning", "Afternoon", "Evening", "Night"]
watch_reasons = ["Entertainment", "Procrastination", "Boredom", "Habit"]
watch_times = [
    "8:00 AM", "9:15 AM", "3:45 PM", "5:00 PM", "6:05 PM", "7:25 PM", "8:30 PM", "9:00 PM", "10:15 PM", "11:30 PM"
]

def main():
    # Custom CSS styling
    st.markdown(
        """
        <style>
            body {
                background-color: #D3D3D3;  /* Light grey background */
                color: #FF4500;  /* Orange font color */
            }
            .title {
                color: #FF4500;
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }
            .sub-header {
                color: #FF4500;
                font-size: 24px;
                margin-bottom: 15px;
            }
            .input-section {
                background-color: #FFFFFF;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .prediction {
                color: #FF4500;
                font-size: 24px;
                text-align: center;
                font-weight: bold;
                margin-top: 20px;
                padding: 10px;
                background-color: #F5F5F5;
                border: 2px solid #FF4500;
                border-radius: 8px;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                font-size: 14px;
                color: #FF4500;
                font-style: italic;
            }
            .sidebar {
                font-size: 16px;
                color: #FF4500;
                background-color: #FFFFFF;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the model name in the sidebar
    st.sidebar.markdown("### Model Used")
    st.sidebar.markdown("XGBoost Classifier 🤖")

    # Page title
    st.markdown("<div class='title'>Addiction Level Predictor 🤖</div>", unsafe_allow_html=True)

    # Collect user inputs
    st.markdown("<div class='sub-header'>Enter Your Details 📝</div>", unsafe_allow_html=True)

    with st.expander("Personal Information"):
        age = st.number_input("Age 🧑", min_value=1, max_value=100, step=1, value=25)
        satisfaction = st.slider("Satisfaction (1 to 10) 😊", min_value=1.0, max_value=10.0, step=0.1, value=5.0)

    with st.expander("Preferences"):
        gender = st.selectbox("Gender 🚻", ["Male", "Female", "Other"])
        location = st.selectbox("Location 🌍", locations)
        platform = st.selectbox("Platform 📱", platforms)
        video_category = st.selectbox("Video Category 🎥", video_categories)
        engagement = st.selectbox("Engagement Level 🤝", engagement_levels)
        frequency = st.selectbox("Frequency ⏱", frequencies)
        watch_reason = st.selectbox("Watch Reason 💭", watch_reasons)
        watch_time = st.selectbox("Watch Time ⏰", watch_times)

    if st.button("Predict Addiction Level 🚀"):
        try:
            # Create a DataFrame from user inputs
            input_data = pd.DataFrame({
                "Age": [age],
                "Satisfaction": [satisfaction],
                "Gender": [gender],
                "Location": [location],
                "Platform": [platform],
                "Video Category": [video_category],
                "Engagement": [engagement],
                "Frequency": [frequency],
                "Watch Reason": [watch_reason],
                "Watch Time": [watch_time]
            })

            # Ensure all columns match the preprocessor requirements
            required_columns = preprocessor.get_feature_names_out()
            for col in required_columns:
                if col not in input_data.columns:
                    input_data[col] = 0  # Add missing columns with default values

            # Preprocess the input data
            input_transformed = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(input_transformed)

            # Decode prediction
            addiction_levels = ["low", "moderate", "high", "no addiction", "extreme"]  # Match your model's encoding
            if 0 <= int(prediction[0]) < len(addiction_levels):
                predicted_level = addiction_levels[int(prediction[0])]
                st.markdown(f"<div class='prediction'>Predicted Addiction Level: {predicted_level} 🔮</div>", unsafe_allow_html=True)
            else:
                st.error("Unexpected prediction result. Please check the model's output.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

    # Footer
    st.markdown("<div class='footer'>Developed by Tirth and Sparsh 💻</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()