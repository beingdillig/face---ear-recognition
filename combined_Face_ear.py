import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class MultimodalBiometricSystem:
    def __init__(self, ear_model_path: str):
        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device.upper()}")
        
        # Face components
        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=60,
            thresholds=[0.7, 0.7, 0.8],
            post_process=False,
            device=self.device
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Ear components
        self.ear_detector = YOLO(ear_model_path)
        self.ear_encoder = self._init_ear_encoder()
        
        # Database
        self.known_face_embeddings = []
        self.known_ear_embeddings = []
        self.known_names = []
        self.load_database()
        
        # Thresholds (adjust based on your validation)
        self.FACE_THRESHOLD = 0.65    # Minimum similarity for face recognition
        self.EAR_THRESHOLD = 0.70     # Minimum similarity for ear recognition
        self.CONFIRMATION_THRESHOLD = 0.85  # Combined threshold
        
        # Confidence levels
        self.FACE_CONFIDENCE = 0.70   # Confidence when only face matches
        self.EAR_CONFIDENCE = 0.75    # Confidence when only ear matches
        self.COMBINED_CONFIDENCE = 0.98  # Confidence when both match

    def _init_ear_encoder(self):
        """Initialize ear feature extractor using ResNet50"""
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final layer
        model.eval().to(self.device)
        
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        return {'model': model, 'preprocess': preprocess}

    def load_database(self):
        """Load registered biometrics from database"""
        if not os.path.exists("biometric_database"):
            os.makedirs("biometric_database")
            
        self.known_face_embeddings = []
        self.known_ear_embeddings = []
        self.known_names = []
        
        for filename in os.listdir("biometric_database"):
            if filename.endswith('.pkl'):
                with open(os.path.join("biometric_database", filename), 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_embeddings.append(data['face_embedding'])
                    self.known_ear_embeddings.append(data['ear_embedding'])
                    self.known_names.append(data['name'])

    def register_person(self, name: str):
        """Fixed registration with proper array handling and error checking"""
        cap = None
        try:
            # Try multiple camera indexes
            for cam_idx in [0, 1, 2]:
                cap = cv2.VideoCapture(cam_idx)
                if cap.isOpened():
                    print(f"Using camera index {cam_idx}")
                    break
            else:
                print("Error: Could not open any camera")
                return False

            angles = [
                {"name": "Frontal face", "samples": [], "target": 7},
                {"name": "Left profile", "samples": [], "target": 7},
                {"name": "Right profile", "samples": [], "target": 7},
                {"name": "Upward angle", "samples": [], "target": 7},
                {"name": "Downward angle", "samples": [], "target": 7}
            ]
            current_angle = 0
            ear_samples = []
            
            print(f"\n=== Registering {name} ===")
            print("Press 'c' to capture, 'n' for next angle, 'q' to quit")

            while current_angle < len(angles):
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to grab frame")
                    continue

                # Display instructions
                display_frame = frame.copy()
                status_text = f"{angles[current_angle]['name']} [{len(angles[current_angle]['samples'])}/{angles[current_angle]['target']}]"
                cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press: c-capture | n-next | q-quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                # Process detections
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection with proper return handling
                face_result = self._process_face(rgb_frame, return_box=True)
                face_box = face_result[1] if face_result else None
                face_embedding = face_result[0] if face_result else None
                
                # Ear detection with proper return handling
                ear_result = self._process_ears(rgb_frame, return_boxes=True)
                ear_boxes = ear_result[1] if ear_result else []
                ear_embeddings = ear_result[0] if ear_result else []
                
                # Draw detections
                if face_box is not None:
                    x1, y1, x2, y2 = map(int, face_box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                for box in ear_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                cv2.imshow(f"Registering {name}", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Registration cancelled")
                    break
                    
                elif key == ord('c'):
                    if face_embedding is not None and len(angles[current_angle]['samples']) < angles[current_angle]['target']:
                        angles[current_angle]['samples'].append(face_embedding)
                        print(f"Captured {angles[current_angle]['name']} sample {len(angles[current_angle]['samples'])}")
                    
                    if ear_embeddings:
                        ear_samples.extend(ear_embeddings)
                        print(f"Captured {len(ear_embeddings)} ear samples")
                
                elif key == ord('n'):
                    if len(angles[current_angle]['samples']) >= angles[current_angle]['target']:
                        current_angle += 1
                        if current_angle < len(angles):
                            print(f"Now capturing: {angles[current_angle]['name']}")
                    else:
                        needed = angles[current_angle]['target'] - len(angles[current_angle]['samples'])
                        print(f"Need {needed} more {angles[current_angle]['name']} samples")

        except Exception as e:
            print(f"Registration error: {str(e)}")
            return False
        
        finally:
            if cap and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            
            # Combine all face samples
            all_face_samples = []
            for angle in angles:
                all_face_samples.extend(angle['samples'])
            
            if len(all_face_samples) >= 10 and len(ear_samples) >= 10:
                self._save_to_database(name, all_face_samples, ear_samples)
                print(f"Successfully registered {name}!")
                return True
            else:
                print(f"Failed to register. Got {len(all_face_samples)} face and {len(ear_samples)} ear samples")
                return False
        
    def recognize_person(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Recognize person using multimodal biometrics"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame = frame.copy()
        results = []
        
        # Process face
        face_embedding, face_box = self._process_face(rgb_frame, return_box=True)
        
        # Process ears
        ear_embeddings, ear_boxes = self._process_ears(rgb_frame, return_boxes=True)
        
        # Multimodal matching
        if face_embedding is not None or ear_embeddings:
            match_result = self._multimodal_match(face_embedding, ear_embeddings)
            if match_result:
                results.append(match_result)
        
        # Visualize results
        output_frame = self._draw_recognition_results(output_frame, face_box, ear_boxes, results)
        return output_frame, results

    def _process_face(self, rgb_frame: np.ndarray, return_box: bool = False):
        """Detect and extract face features"""
        try:
            boxes, probs = self.mtcnn.detect(rgb_frame)
            if boxes is None or len(boxes) == 0:
                return (None, None) if return_box else None
                
            best_idx = np.argmax(probs)
            if probs[best_idx] < 0.8:  # Confidence threshold
                return (None, None) if return_box else None
                
            x1, y1, x2, y2 = map(int, boxes[best_idx])
            face_region = rgb_frame[y1:y2, x1:x2]
            
            # Get face tensor
            face_tensor = self.mtcnn(face_region)
            if face_tensor is None:
                return (None, None) if return_box else None
                
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
                
            # Extract embedding
            embedding = self.facenet(face_tensor.to(self.device))
            embedding = embedding.detach().cpu().numpy().flatten()
            
            return (embedding, (x1, y1, x2, y2)) if return_box else embedding
            
        except Exception as e:
            print(f"Face processing error: {e}")
            return (None, None) if return_box else None

    def _process_ears(self, rgb_frame: np.ndarray, return_boxes: bool = False):
        """Detect and extract ear features"""
        try:
            ear_results = self.ear_detector(rgb_frame, verbose=False)
            embeddings = []
            boxes = []
            
            for box in ear_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ear_img = rgb_frame[y1:y2, x1:x2]
                
                if ear_img.size == 0:
                    continue
                    
                # Extract ear features
                embedding = self.encode_ear(ear_img)
                if embedding is not None:
                    embeddings.append(embedding)
                    boxes.append((x1, y1, x2, y2))
            
            return (embeddings, boxes) if return_boxes else embeddings
            
        except Exception as e:
            print(f"Ear processing error: {e}")
            return ([], []) if return_boxes else []

    def encode_ear(self, ear_img: np.ndarray) -> Optional[np.ndarray]:
        """Convert ear image to feature vector"""
        try:
            input_tensor = self.ear_encoder['preprocess'](ear_img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.ear_encoder['model'](input_batch)
            return features.squeeze().cpu().numpy()
        except:
            return None

    def _multimodal_match(self, face_embedding: Optional[np.ndarray], 
                         ear_embeddings: List[np.ndarray]) -> Optional[Dict]:
        """Fuse face and ear recognition results with confidence boosting"""
        face_match = self._match_face(face_embedding) if face_embedding is not None else None
        ear_match = self._match_ear(ear_embeddings) if ear_embeddings else None
        
        # Case 1: Both modalities agree
        if face_match and ear_match and face_match['name'] == ear_match['name']:
            return {
                'name': face_match['name'],
                'confidence': self.COMBINED_CONFIDENCE,
                'modality': 'face+ear',
                'face_similarity': face_match['similarity'],
                'ear_similarity': ear_match['similarity']
            }
        
        # Case 2: Only face matches
        elif face_match:
            return {
                'name': face_match['name'],
                'confidence': self.FACE_CONFIDENCE,
                'modality': 'face',
                'face_similarity': face_match['similarity'],
                'ear_similarity': None
            }
        
        # Case 3: Only ear matches
        elif ear_match:
            return {
                'name': ear_match['name'],
                'confidence': self.EAR_CONFIDENCE,
                'modality': 'ear',
                'face_similarity': None,
                'ear_similarity': ear_match['similarity']
            }
        
        return None

    def _match_face(self, embedding: np.ndarray) -> Optional[Dict]:
        """Match face against database"""
        if not self.known_face_embeddings:
            return None
            
        similarities = cosine_similarity([embedding], self.known_face_embeddings)[0]
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        
        if max_sim > self.FACE_THRESHOLD:
            return {
                'name': self.known_names[max_idx],
                'similarity': max_sim
            }
        return None

    def _match_ear(self, embeddings: List[np.ndarray]) -> Optional[Dict]:
        """Match ears against database"""
        if not self.known_ear_embeddings:
            return None
            
        # Compare each ear with all known ears
        best_sim = -1
        best_idx = -1
        
        for ear_embedding in embeddings:
            similarities = cosine_similarity([ear_embedding], self.known_ear_embeddings)[0]
            current_max = np.max(similarities)
            if current_max > best_sim:
                best_sim = current_max
                best_idx = np.argmax(similarities)
        
        if best_sim > self.EAR_THRESHOLD:
            return {
                'name': self.known_names[best_idx],
                'similarity': best_sim
            }
        return None

    def _save_to_database(self, name: str, face_samples: List[np.ndarray], 
                         ear_samples: List[np.ndarray]):
        """Save averaged embeddings to database"""
        avg_face = np.mean(face_samples, axis=0)
        avg_ear = np.mean(ear_samples, axis=0)
        
        data = {
            'name': name,
            'face_embedding': avg_face,
            'ear_embedding': avg_ear,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'face_samples': len(face_samples),
            'ear_samples': len(ear_samples)
        }
        
        filename = f"biometric_database/{name}_{data['timestamp']}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        self.load_database()  # Refresh database

    def _draw_recognition_results(self, frame: np.ndarray, 
                                face_box: Optional[Tuple], ear_boxes: List[Tuple], 
                                results: List[Dict]) -> np.ndarray:
        """Draw recognition results on frame"""
        display = frame.copy()
        
        # Draw face box (green)
        if face_box:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ear boxes (blue)
        for box in ear_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw recognition info
        for result in results:
            # Determine color based on confidence
            if result['confidence'] >= 0.9:
                color = (0, 255, 0)  # Green for high confidence
            elif result['confidence'] > 0.7:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 0, 255)  # Red for low
                
            text = f"{result['name']} {result['confidence']*100:.0f}% ({result['modality']})"
            
            # Position text above face box if available, else top of frame
            if face_box:
                x1, y1, _, _ = face_box
                y = max(20, y1 - 10)
            else:
                x1, y = 20, 30
                
            cv2.putText(display, text, (x1, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return display

def main():
    # Initialize system with your ear detection model
    system = MultimodalBiometricSystem("ear_detection_yolo.pt")  # Path to your YOLOv8 ear model
    
    while True:
        print("\n==== Multimodal Biometric System ====")
        print("1. Register New Person")
        print("2. Start Recognition")
        print("3. View Database")
        print("4. Exit")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            name = input("Enter name to register: ").strip()
            if name:
                if system.register_person(name):
                    print(f"Successfully registered {name}!")
                else:
                    print(f"Failed to register {name}")
            else:
                print("Invalid name")
        
        elif choice == '2':
            cap = cv2.VideoCapture(0)
            print("Starting recognition... Press Q to quit")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame, results = system.recognize_person(frame)
                cv2.imshow("Multimodal Recognition", frame)
                
                # Print results to console
                for result in results:
                    print(f"Recognized: {result['name']} ({result['confidence']*100:.0f}% confidence)")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '3':
            print("\nRegistered Persons:")
            for name in system.known_names:
                print(f"- {name}")
            print(f"\nTotal: {len(system.known_names)} persons")
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()