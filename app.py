import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Class mapping
CLASS_LABELS = {
    0: "air_conditioner", 1: "car_horn", 2: "children_playing",
    3: "dog_bark", 4: "drilling", 5: "engine_idling",
    6: "gun_shot", 7: "jackhammer", 8: "siren", 9: "street_music"
}

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 40
MAX_PAD_LEN = 174

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Keras model with error handling"""
    try:
        model_path = "urban_sound_classifier (1).h5"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: `{model_path}`")
            st.stop()
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

def extract_features(audio_data, sr):
    """Extract MFCC features"""
    try:
        # Trim or pad audio
        target_len = sr * DURATION
        if len(audio_data) > target_len:
            audio_data = audio_data[:target_len]
        else:
            audio_data = np.pad(audio_data, (0, target_len - len(audio_data)), mode='constant')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC)
        
        # Pad/truncate
        if mfccs.shape[1] < MAX_PAD_LEN:
            mfccs = np.pad(mfccs, ((0, 0), (0, MAX_PAD_LEN - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        
        return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None

def main():
    st.set_page_config(page_title="Urban Sound Classifier", page_icon="🔊")
    st.title("🎵 Urban Sound Classifier")
    
    # Load model
    model = load_model()
    st.success("✅ Model loaded successfully!")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Classes:** 10 urban sounds  
        **Model:** CNN (700KB)  
        **Input:** 4-second audio clips  
        **Features:** MFCC
        """)
        
        with st.expander("View all categories"):
            for i, label in CLASS_LABELS.items():
                st.write(f"**{i}:** {label}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload audio file (WAV, MP3, OGG, FLAC)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    
    if uploaded_file:
        with st.spinner("Processing audio..."):
            temp_path = "temp_audio.wav"
            try:
                # Save temp file
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                
                # Load and display audio
                audio_data, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
                st.audio(temp_path)
                
                # Predict
                features = extract_features(audio_data, sr)
                if features is not None:
                    prediction = model.predict(features, verbose=0)
                    pred_class = np.argmax(prediction[0])
                    confidence = prediction[0][pred_class]
                    
                    # Results
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", CLASS_LABELS[pred_class])
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    st.progress(float(confidence))
                    
                    # Top 3
                    st.subheader("Top 3 Predictions")
                    top3 = np.argsort(prediction[0])[-3:][::-1]
                    for i, idx in enumerate(top3, 1):
                        st.write(f"{i}. **{CLASS_LABELS[idx]}** - {prediction[0][idx]:.1%}")
            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()
