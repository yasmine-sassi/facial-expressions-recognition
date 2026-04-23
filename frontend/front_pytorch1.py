"""
Streamlit Interface for Facial Emotion Recognition (PyTorch Version)
Real-time emotion detection via webcam or video upload
Includes dashboard with emotion statistics
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import time

# Add src to path for PyTorch models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ==================== CONSTANTS ====================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
FACE_SIZE = 48
DEFAULT_CONFIDENCE_THRESHOLD = 40
DEFAULT_FPS_LIMIT = 15
MAX_FAILED_FRAMES = 10
CAMERA_WARMUP_FRAMES = 5
VIDEO_CONFIDENCE_THRESHOLD = 50
VIDEO_PROCESS_EVERY_N_FRAMES = 2

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

# ==================== EMOTION CONSTANTS ====================
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
def load_model() -> tuple[torch.nn.Module | None, torch.device | None]:
    """Load trained PyTorch emotion recognition model.
    
    Returns:
        tuple: (model, device) or (None, None) if loading failed
    """
    try:
        from src.pytorch_models import get_model
        # Detect available device (GPU if CUDA is available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = "GPU (CUDA)" if device.type == "cuda" else "CPU"
        st.sidebar.info(f"⚠️ Running on {device_name}")
        
        # Try multiple possible paths for the model
        script_dir = Path(__file__).parent.parent
        model_dir = script_dir / 'notebooks' / 'saved_models'
        model_file = model_dir / 'MediumCNN_best.pth'
        
        possible_paths = [
            model_file,
            Path('../notebooks/saved_models/MediumCNN_best.pth'),
            Path('./notebooks/saved_models/MediumCNN_best.pth'),
            Path('notebooks/saved_models/MediumCNN_best.pth'),
        ]
        
        model_path = next((p for p in possible_paths if p.exists()), None)
        
        if not model_path:
            paths_str = '\n'.join([f"  - {p}" for p in possible_paths])
            st.warning(f"⚠️ PyTorch model not found! Tried paths:\n{paths_str}")
            return None, None
        
        # Load main model
        try:
            model = get_model('medium_cnn', num_classes=7, device=device)
            model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
            model.eval()
            st.sidebar.success(f"✅ Model loaded successfully from {model_path.name}")
            return model, device
        except (RuntimeError, KeyError) as e:
            st.error(f"Error loading model: {type(e).__name__}: {e}")
            return None, None
    
    except ImportError as e:
        st.error(f"Error importing model module: {e}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        return None, None

@st.cache_resource
def load_face_cascade() -> cv2.CascadeClassifier:
    """Load and validate face detection cascade classifier.
    
    Returns:
        cv2.CascadeClassifier: Loaded cascade classifier
        
    Raises:
        RuntimeError: If cascade classifier fails to load
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")
    return cascade

# ==================== IMAGE PROCESSING ====================
def preprocess_face(face_roi: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Preprocess face for model prediction.
    
    Args:
        face_roi: Face region of interest as numpy array
        
    Returns:
        tuple: (preprocessed_tensor, resized_face) or (None, None) if face_roi is empty
    """
    # Validate input
    if face_roi.size == 0:
        return None, None
    
    # Resize to model input size
    face_resized = cv2.resize(face_roi, (FACE_SIZE, FACE_SIZE))
    # Convert to grayscale if not already
    if len(face_resized.shape) == 3:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    # Normalize to [0, 1]
    face_normalized = face_resized.astype('float32') / 255.0
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W) for NCHW format (PyTorch convention)
    face_input = face_normalized[np.newaxis, np.newaxis, :, :]  # (1, 1, 48, 48)
    return face_input, face_resized

def detect_emotions_in_frame(frame: np.ndarray, face_cascade: cv2.CascadeClassifier, 
                                model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, list[dict]]:
    """Detect emotions in a frame using PyTorch model.
    
    Args:
        frame: Input frame as BGR numpy array
        face_cascade: OpenCV cascade classifier for face detection
        model: PyTorch emotion classification model
        device: Device to run inference on (cpu or cuda)
        
    Returns:
        tuple: (annotated_frame, emotions_detected_list)
    """
    if model is None or face_cascade is None or device is None:
        return frame, []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    emotions_detected = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess
        face_input, _ = preprocess_face(face_roi)
        
        # Skip if preprocessing failed
        if face_input is None:
            continue
        
        # Predict with PyTorch — tensor is already in NCHW format (1, 1, 48, 48)
        face_tensor = torch.from_numpy(face_input).float().to(device)
        with torch.no_grad():
            predictions = torch.softmax(model(face_tensor), dim=1).cpu().numpy()
        
        emotion_idx = np.argmax(predictions[0])
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(predictions[0][emotion_idx]) * 100
        
        emotions_detected.append({
            'emotion': emotion,
            'confidence': confidence,
            'bbox': (x, y, w, h)
        })
        
        # Draw bounding box and emotion label
        # Note: Color format is BGR (blue, green, red) - hex color indices (5,3,1) convert to BGR order
        color = tuple(int(EMOTION_COLORS[emotion][i:i+2], 16) for i in (5, 3, 1))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame, emotions_detected

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.markdown("## 🧠 Emotion Recognition System")
st.sidebar.markdown("---")

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
model, device = load_model()

try:
    face_cascade = load_face_cascade()
except RuntimeError as e:
    st.error(f"Failed to load face cascade: {e}")
    face_cascade = None

# Initialize session state for storing results
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'stop_webcam' not in st.session_state:
    st.session_state.stop_webcam = False

# ==================== PAGE: WEBCAM DETECTION ====================
if page == "📹 Webcam Detection":
    st.markdown("""
    <div class="emotion-header">
        <h1>🎥 Real-Time Emotion Detection</h1>
        <p>Live emotion detection from your webcam</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure the trained PyTorch model is saved at ../notebooks/saved_models/MediumCNN_best.pth")
    else:
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, DEFAULT_CONFIDENCE_THRESHOLD)
        with col2:
            fps_limit = st.slider("FPS (frames per second)", 1, 30, DEFAULT_FPS_LIMIT)
        with col3:
            show_stats = st.checkbox("Show Statistics", value=True)
        
        st.markdown("---")
        
        col_start, col_stop = st.columns(2)
        with col_start:
            start_detection = st.button("▶️ Start Detection", type="primary")
        with col_stop:
            if st.button("⏹️ Stop Detection"):
                st.session_state.stop_webcam = True
        
        # Create placeholders for live updates
        frame_placeholder = st.empty()
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            frame_count_placeholder = st.empty()
        with stats_col2:
            top_emotion_placeholder = st.empty()
        with stats_col3:
            gpu_status_placeholder = st.empty()
        
        emotion_stats_placeholder = st.empty()
        
        # Device status - display actual device being used
        if device is not None:
            device_status = "GPU (CUDA)" if device.type == "cuda" else "CPU"
        else:
            device_status = "Unknown"
        gpu_status_placeholder.metric("Device", device_status)
        
        # Start live camera capture
        cap = cv2.VideoCapture(0)
        
        # Configure camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Please check camera permissions.")
        else:
            st.info("✅ Webcam prête. Cliquez sur **Start Detection** pour commencer.")
            
            if not start_detection:
                cap.release()
            else:
                st.session_state.stop_webcam = False
                frame_count_val = 0
                all_emotions = []
                failed_frames = 0
                
                # Warm up camera (skip first few frames)
                for _ in range(CAMERA_WARMUP_FRAMES):
                    cap.read()
                
                try:
                    while True:
                        # Check stop button
                        if st.session_state.get('stop_webcam', False):
                            st.session_state.stop_webcam = False
                            st.info("⏹️ Detection stopped.")
                            break
                        
                        ret, frame = cap.read()
                    
                        # Handle frame read errors
                        if not ret or frame is None:
                            failed_frames += 1
                            if failed_frames > MAX_FAILED_FRAMES:
                                st.error("❌ Webcam disconnected or unavailable. Please refresh the page.")
                                break
                            continue
                        
                        failed_frames = 0  # Reset on successful read
                        
                        # Flip for selfie view
                        frame = cv2.flip(frame, 1)
                        
                        # Detect emotions
                        frame_processed, emotions_detected = detect_emotions_in_frame(
                            frame, face_cascade, model, device
                        )
                        
                        # Filter by confidence threshold
                        emotions_detected = [e for e in emotions_detected if e['confidence'] >= confidence_threshold]
                        all_emotions.extend([e['emotion'] for e in emotions_detected])
                        frame_count_val += 1
                        
                        # Display frame
                        frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, width=640)
                        
                        # Update metrics
                        frame_count_placeholder.metric("Frames", frame_count_val)
                        
                        if all_emotions and show_stats:
                            emotion_counts = Counter(all_emotions)
                            top_emotion = emotion_counts.most_common(1)[0]
                            
                            top_emotion_placeholder.metric(
                                "Most Detected",
                                f"{EMOTION_EMOJIS.get(top_emotion[0], '')} {top_emotion[0].capitalize()}",
                                top_emotion[1]
                            )
                            
                            # Display emotion distribution
                            with emotion_stats_placeholder.container():
                                st.markdown("### 📊 Emotion Distribution")
                                cols = st.columns(len(emotion_counts))
                                for col, (emotion, count) in zip(cols, emotion_counts.most_common()):
                                    with col:
                                        st.metric(
                                            f"{EMOTION_EMOJIS.get(emotion, '')} {emotion.capitalize()}",
                                            count,
                                            f"{(count/len(all_emotions))*100:.1f}%"
                                        )
                        
                        # Control frame rate
                        time.sleep(1.0 / fps_limit)
                finally:
                    cap.release()
                    st.session_state.emotion_history.extend(all_emotions)
                    if device is not None and device.type == 'cuda':
                        torch.cuda.empty_cache()

# ==================== PAGE: VIDEO UPLOAD ====================
elif page == "📂 Upload Video":
    st.markdown("""
    <div class="emotion-header">
        <h1>📹 Analyze Uploaded Video</h1>
        <p>Upload a video file to detect emotions frame by frame</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure the trained PyTorch model is saved at ../saved_models/MediumCNN_best.pth")
    else:
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded file safely using tempfile
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                video_path = tmp.name
            
            st.success(f"Video uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Processing Video...")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            with col2:
                st.markdown("### Settings")
                confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, VIDEO_CONFIDENCE_THRESHOLD, key="video_conf")
                process_every_n_frames = st.slider("Process Every N Frames", 1, 10, VIDEO_PROCESS_EVERY_N_FRAMES)
            
            st.markdown("---")
            
            # Process video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            all_emotions = []  # list of {'emotion': str, 'frame': int}
            processed_frames = []
            frame_idx = 0
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    if frame_idx % process_every_n_frames == 0:
                        frame_processed, emotions_detected = detect_emotions_in_frame(
                            frame, face_cascade, model, device
                        )
                        
                        emotions_detected = [e for e in emotions_detected if e['confidence'] >= confidence_threshold]
                        for e in emotions_detected:
                            all_emotions.append({'emotion': e['emotion'], 'frame': frame_idx})
                        processed_frames.append(frame_processed)
                    
                    frame_idx += 1
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {frame_idx}/{total_frames} frames")
                
                cap.release()
                if device is not None and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                st.success("✅ Video Processing Complete!")
                
                # Display results
                emotion_labels_only = [e['emotion'] for e in all_emotions]
                if emotion_labels_only:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Emotion Distribution")
                        emotion_counts = Counter(emotion_labels_only)
                        
                        # Pie chart
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_counts.keys()],
                            values=list(emotion_counts.values()),
                            marker=dict(colors=[EMOTION_COLORS[e] for e in emotion_counts.keys()])
                        )])
                        fig_pie.update_layout(height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📈 Emotion Timeline")
                        timeline_df = pd.DataFrame(all_emotions)  # cols: 'emotion', 'frame'
                        if not timeline_df.empty:
                            fig_line = px.histogram(
                                timeline_df,
                                x='frame',
                                color='emotion',
                                nbins=50,
                                color_discrete_map={e: EMOTION_COLORS[e] for e in EMOTION_LABELS},
                                labels={'frame': 'Frame Number', 'emotion': 'Emotion'}
                            )
                            fig_line.update_layout(height=400, showlegend=True)
                            st.plotly_chart(fig_line, use_container_width=True)
                    
                    # Summary table
                    st.markdown("### 📋 Summary Statistics")
                    summary_df = pd.DataFrame({
                        'Emotion': list(emotion_counts.keys()),
                        'Count': list(emotion_counts.values()),
                        'Percentage': [f"{(count/len(emotion_labels_only))*100:.1f}%" for count in emotion_counts.values()]
                    }).sort_values('Count', ascending=False)
                    
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Store results
                    st.session_state.emotion_history.extend(emotion_labels_only)
                else:
                    st.warning("No emotions detected in the video. Try lowering the confidence threshold.")
            
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                # Clean up temp file
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
            if emotion_counts:
                top_emotion = emotion_counts.most_common(1)[0]
                st.metric(
                    "Most Frequent",
                    f"{EMOTION_EMOJIS[top_emotion[0]]} {top_emotion[0].capitalize()}",
                    f"{(top_emotion[1]/total_detections)*100:.1f}%"
                )
            else:
                st.metric("Most Frequent", "N/A", "0%")
        
        with col3:
            unique_emotions = len(emotion_counts)
            st.metric("Unique Emotions", unique_emotions)
        
        with col4:
            st.metric("Emotion Diversity", f"{(unique_emotions/7)*100:.1f}%")
        
        st.markdown("---")
        
        # Visualizations
        if emotion_counts:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🥧 Emotion Distribution")
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_counts.keys()],
                    values=list(emotion_counts.values()),
                    marker=dict(colors=[EMOTION_COLORS[e] for e in emotion_counts.keys()])
                )])
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Emotion Counts")
                fig_bar = go.Figure(data=[go.Bar(
                    x=[f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in emotion_counts.keys()],
                    y=list(emotion_counts.values()),
                    marker=dict(color=[EMOTION_COLORS[e] for e in emotion_counts.keys()])
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
        else:
            st.warning("No emotion data available to display charts.")
        
        # Clear history button
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
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
        st.markdown(f"""
        ### 🧠 Facial Emotion Recognition System
        
        This application uses a Deep Learning CNN model to detect facial emotions in real-time.
        
        **Framework:** PyTorch with GPU acceleration
        
        #### 🎯 Features
        - **Real-time Webcam Detection**: Stream emotions detected from your webcam
        - **Video Analysis**: Upload and analyze video files for emotion detection
        - **Dashboard**: View comprehensive statistics of all detections
        - **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
        - **GPU Acceleration**: Optimized for NVIDIA GPUs
        
        #### 🔬 Technical Details
        
        **PyTorch Models Available:**
        - Baseline CNN: 2.7M parameters
        - Advanced CNN: 7.2M parameters
        - ResNet: 4M+ parameters
        
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
        
        **GPU Support:**
        - PyTorch: Full GPU acceleration (NVIDIA GTX/RTX)
        - Training: ~6-9x faster than CPU
        - Inference: ~15-30x faster than CPU
        
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
        - Train models using PyTorch notebooks:
          - 04_pytorch_baseline.ipynb
          - 05_pytorch_advanced.ipynb
        - Save model to `../saved_models/MediumCNN_best.pth`
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
        
        **Frameworks:** 
        - PyTorch 2.1.0
        - Streamlit
        
        **Language:** Python 3.10+  
        **License:** MIT  
        
        **Dataset:** FER2013 (35,887 images)
        **GPU:** NVIDIA RTX 3050 (4GB)
        
        ---
        
        ### 🔗 Resources
        
        - [FER2013 Dataset](https://www.kaggle.com/datasets/deadskull7/fer2013)
        - [PyTorch Docs](https://pytorch.org)
        - [Streamlit Docs](https://docs.streamlit.io)
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 20px;">
    <p>Built with ❤️ for Facial Emotion Recognition | PyTorch + Streamlit | 2025-2026</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #888; font-size: 12px; padding: 10px;">
<p>Facial Emotion Recognition System | PyTorch | 2025-2026</p>
</div>
""", unsafe_allow_html=True)
