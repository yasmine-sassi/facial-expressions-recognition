"""Prediction module for emotion recognition."""

import numpy as np
import cv2
from tensorflow.keras.models import load_model


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 165, 255),  # Orange
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 0),      # Green
    'Neutral': (200, 200, 200),# Gray
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 255)  # Yellow
}


class EmotionPredictor:
    """Class for emotion prediction from images."""
    
    def __init__(self, model_path):
        """Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved trained model
        """
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def preprocess_face(self, face_image):
        """Preprocess face image for prediction.
        
        Args:
            face_image: Face image (BGR)
        
        Returns:
            Preprocessed image ready for model
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        gray = cv2.resize(gray, (48, 48))
        
        # Normalize
        gray = gray.astype('float32') / 255.0
        
        # Add channel dimension
        gray = np.expand_dims(gray, axis=-1)
        
        return gray
    
    def predict_emotion(self, face_image):
        """Predict emotion from face image.
        
        Args:
            face_image: Face image (BGR)
        
        Returns:
            Tuple of (emotion_label, confidence, probabilities)
        """
        preprocessed = self.preprocess_face(face_image)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        probabilities = self.model.predict(preprocessed, verbose=0)[0]
        emotion_idx = np.argmax(probabilities)
        emotion_label = EMOTION_LABELS[emotion_idx]
        confidence = probabilities[emotion_idx]
        
        return emotion_label, confidence, probabilities
    
    def predict_video(self, video_source=0, show_fps=True):
        """Predict emotions from video stream.
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
            show_fps: Whether to display FPS counter
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        import time
        fps_start = time.time()
        fps_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Predict emotion for each face
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion, confidence, _ = self.predict_emotion(face_roi)
                
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw FPS
            if show_fps:
                fps_counter += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1:
                    fps = fps_counter / elapsed
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    fps_counter = 0
                    fps_start = time.time()
            
            # Display
            cv2.imshow('Emotion Recognition', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
