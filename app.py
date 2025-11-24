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
MAX_PAD_LEN = 174

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model and inspect input shape"""
    try:
        model_path = "urban_sound_classifier (1).h5"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: `{model_path}`")
            st.stop()
        
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path)
        
        # INSPECT MODEL INPUT SHAPE
        input_shape = model.input_shape
        st.sidebar.subheader("🔍 Model Info")
        st.sidebar.code(f"Expected Input Shape: {input_shape}")
        
        return model, input_shape
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

def extract_features(audio_data, sr, expected_shape):
    """Extract features matching model's expected shape"""
    try:
        target_len = sr * DURATION
        if len(audio_data) > target_len:
            audio_data = audio_data[:target_len]
        else:
            audio_data = np.pad(audio_data, (0, target_len - len(audio_data)), mode='constant')
        
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC)
        
        if mfccs.shape[1] < MAX_PAD_LEN:
            mfccs = np.pad(mfccs, ((0, 0), (0, MAX_PAD_LEN - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        
        # ADAPT TO MODEL'S EXPECTED SHAPE
        if len(expected_shape) == 4:  # (batch, height, width, channels)
            h, w, c = expected_shape[1], expected_shape[2], expected_shape[3]
            features = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
            
            # If model expects different dimensions, transpose
            if h == mfccs.shape[1] and w == mfccs.shape[0]:
                features = features.transpose(0, 2, 1, 3)  # Swap height/width
        
        elif len(expected_shape) == 3:  # (batch, timesteps, features)
            features = mfccs.reshape(1, mfccs.shape[1], mfccs.shape[0])
        
        else:
            st.error(f"Unexpected model input shape: {expected_shape}")
            st.stop()
        
        return features.astype(np.float32)
        
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def main():
    st.set_page_config(page_title="Urban Sound Classifier", page_icon="🔊")
    st.title("🎵 Urban Sound Classifier")
    
    # Load model and get expected shape
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
    
    uploaded_file = st.file_uploader("Upload audio (WAV, MP3, OGG, FLAC)", 
                                     type=["wav", "mp3", "ogg", "flac"])
    
    if uploaded_file:
        temp_path = "temp_audio.wav"
        try:
            with st.spinner("Processing..."):
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                audio_data, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
                st.audio(temp_path)
                
                # Extract with correct shape
                features = extract_features(audio_data, sr, input_shape)
                
                if features is not None:
                    st.sidebar.code(f"Feature Shape: {features.shape}")
                    
                    prediction = model.predict(features, verbose=0)
                    pred_class = np.argmax(prediction[0])
                    confidence = prediction[0][pred_class]
                    
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🎯 Prediction", CLASS_LABELS[pred_class])
                    with col2:
                        st.metric("💯 Confidence", f"{confidence:.1%}")
                    
                    st.progress(float(confidence))
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()
