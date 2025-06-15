import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the trained model
model = load_model('har_cnn_lstm_model.h5')

# Set page title
st.title("Human Activity Recognition (HAR) App")
st.write("Upload your sensor CSV data and get activity predictions.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a test CSV file", type=["csv"])

# Load LabelEncoder (must match training order)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS'])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    # Drop subject and activity if present
    if 'subject' in data.columns:
        data = data.drop(['subject'], axis=1)
    if 'Activity' in data.columns:
        data = data.drop(['Activity'], axis=1)

    # Reshape for model (samples, time_steps, features)
    reshaped_data = data.values.reshape(data.shape[0], data.shape[1], 1)

    # Predict
    y_pred_probs = model.predict(reshaped_data)
    y_preds = np.argmax(y_pred_probs, axis=1)
    predicted_labels = label_encoder.inverse_transform(y_preds)

    # Show results
    st.subheader("Predicted Activities")
    st.write(predicted_labels)

    # Add to original data if needed
    result_df = data.copy()
    result_df['Predicted_Activity'] = predicted_labels
    st.write(result_df.head())

    # Downloadable results
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "har_predictions.csv", "text/csv")
