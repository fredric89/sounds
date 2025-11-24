import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

CLASS_LABELS = {
    0: "air_conditioner", 1: "car_horn", 2: "children_playing",
    3: "dog_bark", 4: "drilling", 5: "engine_idling",
    6: "gun_shot", 7: "jackhammer", 8: "siren", 9: "street_music"
}

SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 40

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model and return it with expected input shape"""
    try:
        model_path = "urban_sound_classifier (1).h5"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: `{model_path}`")
            st.stop()
        
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path)
        
        # Get expected input shape
        input_shape = model.input_shape
        st.sidebar.subheader("🔍 Model Input Shape")
        st.sidebar.code(f"{input_shape}")
        
        return model, input_shape
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

def extract_features(audio_data, sr, target_shape):
    """Extract features matching EXACT model input shape"""
    try:
        # Trim/pad audio to 4 seconds
        target_len = sr * DURATION
        audio_data = audio_data[:target_len] if len(audio_data) > target_len else np.pad(
            audio_data, (0, target_len - len(audio_data)), mode='constant'
        )
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC)
        
        # Get target dimensions from model shape
        # Handle both (batch, height, width, channels) and (batch, timesteps, features)
        if len(target_shape) == 4:
            _, target_h, target_w, target_c = target_shape
            # Resize MFCC to match exactly
            from skimage.transform import resize
            mfccs_resized = resize(mfccs, (target_h, target_w), preserve_range=True)
            features = mfccs_resized.reshape(1, target_h, target_w, target_c)
            
        elif len(target_shape) == 3:
            _, target_t, target_f = target_shape
            # For 3D input (time, features)
            if mfccs.shape[1] < target_t:
                mfccs = np.pad(mfccs, ((0, 0), (0, target_t - mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[:, :target_t]
            features = mfccs.T.reshape(1, target_t, target_f)  # Transpose to (time, features)
        
        else:
            st.error(f"Unsupported input shape: {target_shape}")
            return None
        
        return features.astype(np.float32)
        
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def main():
    st.set_page_config(page_title="Urban Sound Classifier", page_icon="🔊")
    st.title("🎵 Urban Sound Classifier")
    
    # Load model and get its expected shape
    model, input_shape = load_model()
    st.success("✅ Model loaded!")
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""**Model:** CNN (700KB)  
        **Input:** 4-second audio clips  
        **Features:** MFCC""")
        
        with st.expander("📋 Categories"):
            for i, label in CLASS_LABELS.items():
                st.write(f"**{i}:** {label}")
    
    uploaded_file = st.file_uploader(
        "Upload audio file (WAV, MP3, OGG, FLAC)",
        type=["wav", "mp3", "ogg", "flac"]
    )
    
    if uploaded_file:
        temp_path = "temp_audio.wav"
        try:
            with st.spinner("Processing audio..."):
                # Save uploaded file
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load audio
                audio_data, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
                st.audio(temp_path)
                
                # Extract features with correct shape
                features = extract_features(audio_data, sr, input_shape)
                
                if features is not None:
                    st.sidebar.subheader("📊 Feature Shape")
                    st.sidebar.code(f"{features.shape}")
                    
                    # Predict
                    prediction = model.predict(features, verbose=0)
                    pred_class = np.argmax(prediction[0])
                    confidence = prediction[0][pred_class]
                    
                    # Display results
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🎯 Prediction", CLASS_LABELS[pred_class])
                    with col2:
                        st.metric("💯 Confidence", f"{confidence:.1%}")
                    
                    st.progress(float(confidence))
                    
                    # Top 3 predictions
                    st.subheader("🏆 Top 3 Predictions")
                    top3 = np.argsort(prediction[0])[-3:][::-1]
                    for i, idx in enumerate(top3, 1):
                        st.write(f"{i}. **{CLASS_LABELS[idx]}** - {prediction[0][idx]:.1%}")
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()
