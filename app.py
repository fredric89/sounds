import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
import os
from pathlib import Path

# Class mapping
CLASS_LABELS = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MFCC = 40
MAX_PAD_LEN = 174

@st.cache_resource
def load_model():
    """Load the Keras model"""
    model_path = "urban_sound_classifier (1).h5"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please make sure it's in the same directory.")
        return None
    return tf.keras.models.load_model(model_path)

def extract_features(audio_data, sr):
    """Extract MFCC features from audio"""
    try:
        # Ensure audio is the right length
        if len(audio_data) > sr * DURATION:
            audio_data = audio_data[:sr * DURATION]
        else:
            # Pad audio if too short
            pad_len = sr * DURATION - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_len), mode='constant')
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC)
        
        # Pad or truncate to ensure consistent shape
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        
        # Add channel dimension
        mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        return mfccs
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def predict_sound(model, features):
    """Make prediction using the model"""
    prediction = model.predict(features, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    return predicted_class, confidence, prediction[0]

def main():
    st.set_page_config(
        page_title="Urban Sound Classifier",
        page_icon="🔊",
        layout="centered"
    )
    
    st.title("🎵 Urban Sound Classifier")
    st.markdown("""
    Upload an audio file to classify it into one of 10 urban sound categories:
    """)
    
    # Display class labels
    with st.expander("View all classification categories"):
        cols = st.columns(2)
        for i, (class_id, label) in enumerate(CLASS_LABELS.items()):
            cols[i % 2].write(f"**{class_id}:** {label}")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file...",
        type=["wav", "mp3", "ogg", "flac"],
        help="Supported formats: WAV, MP3, OGG, FLAC"
    )
    
    if uploaded_file is not None:
        # Create a temporary file
        with st.spinner("Processing audio..."):
            # Save uploaded file temporarily
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Load audio
                audio_data, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
                
                # Display audio player
                st.audio(temp_path)
                
                # Extract features
                features = extract_features(audio_data, sr)
                
                if features is not None:
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_sound(model, features)
                    predicted_label = CLASS_LABELS[predicted_class]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Main prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Class", predicted_label)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Progress bar
                    st.markdown("### Confidence Level")
                    st.progress(float(confidence))
                    
                    # Top 3 predictions
                    st.markdown("### Top 3 Predictions")
                    top_3_indices = np.argsort(all_predictions)[-3:][::-1]
                    
                    for i, idx in enumerate(top_3_indices, 1):
                        label = CLASS_LABELS[idx]
                        conf = all_predictions[idx]
                        st.write(f"{i}. **{label}** - {conf:.2%}")
                        st.progress(float(conf))
                    
                    # Visualize MFCCs (optional)
                    with st.expander("View MFCC Features"):
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 4))
                        mfccs_display = features[0, :, :, 0]  # Remove batch and channel dims
                        img = librosa.display.specshow(mfccs_display, sr=sr, x_axis='time', ax=ax)
                        ax.set_title('MFCC Features')
                        plt.colorbar(img, ax=ax)
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Instructions
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app classifies urban sounds using a deep learning model.
        
        **Model:** CNN trained on MFCC features
        
        **Input:** 4-second audio clips
        
        **Classes:** 10 urban sound categories
        
        **How to use:**
        1. Upload an audio file
        2. Wait for processing
        3. View predictions
        
        **Tips:**
        - Use clear audio recordings
        - 4-second clips work best
        - Avoid background noise when possible
        """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit & TensorFlow")

if __name__ == "__main__":
    main()
