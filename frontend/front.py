"""
Streamlit Interface for Facial Emotion Recognition
Allows real-time emotion detection via webcam or video upload
Includes dashboard with emotion statistics
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from pathlib import Path
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .emotion-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤮',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😮'
}
EMOTION_COLORS = {
    'angry': '#e74c3c',
    'disgust': '#8e44ad',
    'fear': '#e67e22',
    'happy': '#f1c40f',
    'neutral': '#95a5a6',
    'sad': '#3498db',
    'surprise': '#1abc9c'
}

# ==================== LOAD MODEL & DATA ====================
@st.cache_resource
def load_model():
    """Load trained emotion recognition model"""
    try:
        # Try to load from saved models directory
        model_path = Path('../saved_models/best_model.h5')
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.warning("Model not found. Please train a model first in 02_baseline_cnn.ipynb or 03_advanced_model.ipynb")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_face_cascade():
    """Load face detection cascade classifier"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

# ==================== IMAGE PROCESSING ====================
def preprocess_face(face_roi):
    """Preprocess face for model prediction"""
    # Resize to 48x48
    face_resized = cv2.resize(face_roi, (48, 48))
    # Convert to grayscale if not already
    if len(face_resized.shape) == 3:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    # Normalize to [0, 1]
    face_normalized = face_resized.astype('float32') / 255.0
    # Add batch and channel dimensions
    face_input = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=-1)
    return face_input, face_resized

def detect_emotions_in_frame(frame, face_cascade, model):
    """Detect emotions in a frame"""
    if model is None:
        return frame, []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions_detected = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess
        face_input, _ = preprocess_face(face_roi)
        
        # Predict
        predictions = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(predictions[0][emotion_idx]) * 100
        
        emotions_detected.append({
            'emotion': emotion,
            'confidence': confidence,
            'bbox': (x, y, w, h)
        })
        
        # Draw bounding box and emotion label
        color = tuple(int(EMOTION_COLORS[emotion][i:i+2], 16) for i in (5, 3, 1))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame, emotions_detected

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.markdown("## 🧠 Emotion Recognition System")
page = st.sidebar.radio(
    "Select Mode:",
    ["📹 Webcam Detection", "📂 Upload Video", "📊 Dashboard", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Facial Emotion Recognition** using Deep Learning\n\n"
    "7 Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise"
)

# Load model and cascade
model = load_model()
face_cascade = load_face_cascade()

# Initialize session state for storing results
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# ==================== PAGE: WEBCAM DETECTION ====================
if page == "📹 Webcam Detection":
    st.markdown("""
    <div class="emotion-header">
        <h1>🎥 Real-Time Emotion Detection</h1>
        <p>Use your webcam to detect emotions in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure a trained model is saved in ../saved_models/best_model.h5")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Live Camera Feed")
            camera_placeholder = st.empty()
            frame_count = st.empty()
            
        with col2:
            st.markdown("### Statistics")
            emotion_counter_placeholder = st.empty()
            top_emotion_placeholder = st.empty()
            confidence_placeholder = st.empty()
        
        st.markdown("---")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            duration = st.slider("Recording Duration (seconds)", 5, 120, 30)
        with col2:
            start_button = st.button("▶️ Start Detection", key="webcam_start")
        with col3:
            confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50)
        
        if start_button:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot open webcam. Please check camera permissions.")
            else:
                st.success("Webcam started! Press 'Stop' to finish.")
                
                frame_count_val = 0
                all_emotions = []
                stop_button = st.button("⏹️ Stop Detection")
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Flip for selfie view
                    frame = cv2.flip(frame, 1)
                    
                    # Detect emotions
                    frame_processed, emotions_detected = detect_emotions_in_frame(frame, face_cascade, model)
                    
                    # Filter by confidence threshold
                    emotions_detected = [e for e in emotions_detected if e['confidence'] >= confidence_threshold]
                    all_emotions.extend([e['emotion'] for e in emotions_detected])
                    frame_count_val += 1
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, use_column_width=True)
                    
                    # Update statistics
                    frame_count.metric("Frames Processed", frame_count_val)
                    
                    if all_emotions:
                        emotion_counts = Counter(all_emotions)
                        top_emotion = emotion_counts.most_common(1)[0]
                        
                        with emotion_counter_placeholder.container():
                            for emotion, count in emotion_counts.most_common():
                                st.write(f"{EMOTION_EMOJIS[emotion]} {emotion.capitalize()}: {count}")
                        
                        top_emotion_placeholder.metric(
                            "Most Frequent",
                            f"{EMOTION_EMOJIS[top_emotion[0]]} {top_emotion[0].capitalize()}",
                            top_emotion[1]
                        )
                
                cap.release()
                
                # Store results
                st.session_state.emotion_history.extend(all_emotions)
                st.success("✅ Detection Complete!")
                
                if all_emotions:
                    st.markdown("---")
                    st.markdown("### 📈 Summary")
                    summary_df = pd.DataFrame({
                        'Emotion': list(emotion_counts.keys()),
                        'Count': list(emotion_counts.values()),
                        'Percentage': [f"{(count/len(all_emotions))*100:.1f}%" for count in emotion_counts.values()]
                    }).sort_values('Count', ascending=False)
                    st.dataframe(summary_df, use_container_width=True)

# ==================== PAGE: VIDEO UPLOAD ====================
elif page == "📂 Upload Video":
    st.markdown("""
    <div class="emotion-header">
        <h1>📹 Analyze Uploaded Video</h1>
        <p>Upload a video file to detect emotions frame by frame</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure a trained model is saved.")
    else:
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded file
            video_path = f"temp_video_{datetime.now().timestamp()}.mp4"
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            st.success(f"Video uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Processing Video...")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            with col2:
                st.markdown("### Settings")
                confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50, key="video_conf")
                process_every_n_frames = st.slider("Process Every N Frames", 1, 10, 2)
            
            st.markdown("---")
            
            # Process video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            all_emotions = []
            processed_frames = []
            frame_idx = 0
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    if frame_idx % process_every_n_frames == 0:
                        frame_processed, emotions_detected = detect_emotions_in_frame(
                            frame, face_cascade, model
                        )
                        
                        emotions_detected = [e for e in emotions_detected if e['confidence'] >= confidence_threshold]
                        all_emotions.extend([e['emotion'] for e in emotions_detected])
                        processed_frames.append(frame_processed)
                    
                    frame_idx += 1
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {frame_idx}/{total_frames} frames")
                
                cap.release()
                
                st.success("✅ Video Processing Complete!")
                
                # Display results
                if all_emotions:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Emotion Distribution")
                        emotion_counts = Counter(all_emotions)
                        
                        # Pie chart
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_counts.keys()],
                            values=list(emotion_counts.values()),
                            marker=dict(colors=[EMOTION_COLORS[e][1:] for e in emotion_counts.keys()])
                        )])
                        fig_pie.update_layout(height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📈 Emotion Timeline")
                        # Create timeline
                        frames_per_emotion = len(all_emotions) // len(emotion_counts) if emotion_counts else 1
                        timeline_data = []
                        for i, emotion in enumerate(all_emotions):
                            timeline_data.append({'Frame': i, 'Emotion': emotion})
                        
                        timeline_df = pd.DataFrame(timeline_data)
                        if not timeline_df.empty:
                            fig_line = px.histogram(
                                timeline_df,
                                x='Frame',
                                color='Emotion',
                                nbins=50,
                                color_discrete_map={e: EMOTION_COLORS[e][1:] for e in EMOTION_LABELS}
                            )
                            fig_line.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig_line, use_container_width=True)
                    
                    # Summary table
                    st.markdown("### 📋 Summary Statistics")
                    summary_df = pd.DataFrame({
                        'Emotion': list(emotion_counts.keys()),
                        'Count': list(emotion_counts.values()),
                        'Percentage': [f"{(count/len(all_emotions))*100:.1f}%" for count in emotion_counts.values()]
                    }).sort_values('Count', ascending=False)
                    
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Store results
                    st.session_state.emotion_history.extend(all_emotions)
                else:
                    st.warning("No emotions detected in the video. Try lowering the confidence threshold.")
            
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                # Clean up temp file
                import os
                if os.path.exists(video_path):
                    os.remove(video_path)

# ==================== PAGE: DASHBOARD ====================
elif page == "📊 Dashboard":
    st.markdown("""
    <div class="emotion-header">
        <h1>📊 Emotion Detection Dashboard</h1>
        <p>Overall statistics from all detections</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.emotion_history:
        st.info("👉 No emotion detections yet. Start by using Webcam Detection or Upload Video!")
    else:
        emotion_counts = Counter(st.session_state.emotion_history)
        total_detections = len(st.session_state.emotion_history)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", total_detections)
        
        with col2:
            top_emotion = emotion_counts.most_common(1)[0]
            st.metric(
                "Most Frequent",
                f"{EMOTION_EMOJIS[top_emotion[0]]} {top_emotion[0].capitalize()}",
                f"{(top_emotion[1]/total_detections)*100:.1f}%"
            )
        
        with col3:
            unique_emotions = len(emotion_counts)
            st.metric("Unique Emotions", unique_emotions)
        
        with col4:
            st.metric("Emotion Diversity", f"{(unique_emotions/7)*100:.1f}%")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥧 Emotion Distribution")
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_counts.keys()],
                values=list(emotion_counts.values()),
                marker=dict(colors=[EMOTION_COLORS[e][1:] for e in emotion_counts.keys()])
            )])
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Emotion Counts")
            fig_bar = go.Figure(data=[go.Bar(
                x=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_counts.keys()],
                y=list(emotion_counts.values()),
                marker=dict(color=[EMOTION_COLORS[e][1:] for e in emotion_counts.keys()])
            )])
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed table
        st.markdown("---")
        st.markdown("### 📋 Detailed Statistics")
        stats_df = pd.DataFrame({
            'Emotion': list(emotion_counts.keys()),
            'Count': list(emotion_counts.values()),
            'Percentage': [f"{(count/total_detections)*100:.1f}%" for count in emotion_counts.values()],
            'Emoji': [EMOTION_EMOJIS[e] for e in emotion_counts.keys()]
        }).sort_values('Count', ascending=False)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Clear history button
        st.markdown("---")
        if st.button("🗑️ Clear History"):
            st.session_state.emotion_history = []
            st.rerun()

# ==================== PAGE: ABOUT ====================
else:  # About page
    st.markdown("""
    <div class="emotion-header">
        <h1>ℹ️ About This Application</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🧠 Facial Emotion Recognition System
        
        This application uses a Deep Learning CNN model to detect facial emotions in real-time.
        
        #### 🎯 Features
        - **Real-time Webcam Detection**: Stream emotions detected from your webcam
        - **Video Analysis**: Upload and analyze video files for emotion detection
        - **Dashboard**: View comprehensive statistics of all detections
        - **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
        
        #### 🔬 Technical Details
        
        **Model Architecture:**
        - Convolutional Neural Network (CNN)
        - Input: 48×48 grayscale images
        - Preprocessing: Normalization to [0,1] range
        - Optimization: Adam optimizer
        - Loss: Sparse categorical cross-entropy
        
        **Face Detection:**
        - Haar Cascade Classifier
        - Real-time detection and tracking
        
        **Dataset:**
        - FER2013 Dataset: 35,887 images
        - 7 emotion classes
        - Class balancing via weighted loss
        
        ---
        
        ### 📚 How to Use
        
        1. **Webcam Detection**
           - Click the webcam tab
           - Adjust confidence threshold and recording duration
           - Click "Start Detection" to begin
           - Watch real-time emotion detection
        
        2. **Upload Video**
           - Upload an MP4, AVI, MOV, or MKV file
           - Adjust processing parameters
           - View emotion distribution and timeline
        
        3. **Dashboard**
           - View overall statistics from all detections
           - Track emotion frequency and patterns
           - Export data for analysis
        
        ---
        
        ### 🔧 Troubleshooting
        
        **Webcam not working?**
        - Check camera permissions in system settings
        - Try refreshing the browser
        - Ensure decent lighting for better detection
        
        **Low detection accuracy?**
        - Improve lighting conditions
        - Ensure face is clearly visible
        - Lower confidence threshold
        - Try multiple angles
        
        **Model not loading?**
        - Train a model using 02_baseline_cnn.ipynb or 03_advanced_model.ipynb
        - Save model to `../saved_models/best_model.h5`
        - Restart the application
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Emotion Classes
        
        """)
        for emotion in EMOTION_LABELS:
            st.write(f"{EMOTION_EMOJIS[emotion]} **{emotion.capitalize()}**")
        
        st.markdown("""
        ---
        
        ### 🎓 Project Info
        
        **Framework:** Streamlit  
        **ML Framework:** TensorFlow/Keras  
        **Language:** Python 3.8+  
        **License:** MIT  
        
        **Dataset:** FER2013  
        **Model:** CNN  
        **Accuracy:** [Train your model to see]
        
        ---
        
        ### 🔗 Resources
        
        - [FER2013 Dataset](https://www.kaggle.com/datasets/deadskull7/fer2013)
        - [Streamlit Docs](https://docs.streamlit.io)
        - [TensorFlow Docs](https://www.tensorflow.org)
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
    <p>Built with ❤️ for Facial Emotion Recognition | GL4 INSAT Project</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px; padding: 10px;">
<p>Facial Emotion Recognition System | TensorFlow + Streamlit | 2024-2025</p>
</div>
""", unsafe_allow_html=True)
