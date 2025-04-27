# Multimodal Biometric Recognition System (Face + Ear)

This project implements a **real-time multimodal biometric authentication system** using **face** and **ear recognition** to maximize identification accuracy.  
It integrates deep learning models like **MTCNN**, **Facenet**, **YOLO**, and **ResNet50**.

---

## ✨ Features

- **Face Detection** using MTCNN
- **Face Embedding** extraction using InceptionResnetV1 (pretrained on VGGFace2)
- **Ear Detection** using YOLOv8
- **Ear Embedding** extraction using ResNet50
- **Multimodal Authentication** (Face + Ear fusion)
- **Registration System** with guided multi-angle capturing (frontal, left, right, up, down)
- **Live Recognition** with real-time camera input
- **Database Management** (automatic saving and loading of embeddings)
- **Confidence-Based Decisions** (face only, ear only, combined)

---


## 🔥 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- facenet-pytorch
- torchvision
- ultralytics
- mediapipe
- scikit-learn
- numpy
- pickle

Install all dependencies using:

```bash
pip install opencv-python torch torchvision facenet-pytorch ultralytics mediapipe scikit-learn numpy
```
## 🛠️ How it Works

### Initialization: 
  - Loads face detection (MTCNN) and face recognition (Facenet) models.
  - Loads YOLO model for ear detection (you must provide the trained model .pt file).
  - Loads ResNet50 for ear feature extraction.
  - Loads existing user biometric data from the database.

###  Registration
  - Capture face and ear samples at multiple angles.
  - Minimum 35 face samples and 10 ear samples are required.
  - Saves the embeddings and user name into a .pkl file for future matching.

### Recognition
  - Detects face and ears in live camera feed.
  - Extracts embeddings and matches against the database.
  - Computes cosine similarity for both face and ear.
  - Combines matching scores to increase recognition reliability
  - Draws bounding boxes with recognized names and confidence scores.

---


### 🧠 Important Notes
   **Ear detection model** (.pt file) must be downlaoded and kept in the same folder as code.
  
  System automatically handles device **(CPU/GPU)** selection.

  Minimum number of samples ensures **Robust Recognition**.

  **Database** folder will be created automatically if it doesn't exist.

---

## 🎯 Accuracy Estimation of Face and Ear Biometric System

### ✨ Theoretical Accuracy

When combining Face Recognition and Ear Recognition in a multimodal biometric system, the theoretical identification rates show significant improvement due to the fusion of two independent biometric modalities.

| System                     | Typical Accuracy (Identification Rate) |
|----------------------------|--------------------------------------|
| Face Recognition (Alone)   | 95% – 98%                             |
| Ear Recognition (Alone)    | 85% – 92%                             |
| **Face + Ear (Fusion)** | **> 99% theoretically** |

**Reason:**

Combining two independent biometrics (face + ear) drastically reduces the chances of misidentification due to the inherent redundancy and complementary information present in these modalities. If one modality faces challenges (e.g., poor lighting affecting face recognition), the other (ear recognition) might still provide reliable information.

**Fusion formula (basic theory):**

The probability of a combined error in a simplified scenario where the errors are independent can be approximated by multiplying the individual error probabilities:

```
P(error_combined) ≈ P(face_error) × P(ear_error)
```
**Example:**

If the error rate for face recognition is 2% and the error rate for ear recognition is 8%:

```
Combined error = 0.02 × 0.08 = 0.0016 → 0.16% error
Accuracy ≈ 100% - 0.16% = 99.84%
```

### 🧪 Practical Accuracy

Real-world testing, however, introduces various factors that can affect the ideal theoretical accuracy. Practical results typically show slightly lower but still significantly high accuracy for the combined system.

| Mode                             | Expected Practical Accuracy |
|----------------------------------|-----------------------------|
| Face Only (Good Lighting, Frontal) | 96% – 98%                   |
| Ear Only (Clear Profile Shot)     | 85% – 90%                   |
| **Face + Ear Combined** | **98% – 99.5%** |


**✅ Good conditions for optimal accuracy:**

* Controlled lighting environment
* Faces presented frontally or with slight turns
* Ears fully visible and not obstructed by hair or accessories

### 📋 Conditions Affecting Accuracy

Several factors can influence the performance and accuracy of both individual and combined biometric systems:

| Factor                        | Impact                                                                    |
|-------------------------------|---------------------------------------------------------------------------|
| Lighting                      | Poor lighting significantly reduces both face and ear detection accuracy.    |
| Pose                          | Extreme head rotations (>45°) primarily affect face detection reliability. |
| Occlusion                     | Hair covering the ear or face masks substantially lower recognition quality. |
| Camera Quality                | Low-resolution cameras degrade the clarity of facial and ear features.      |
| Ear Detection Model           | A weak or inaccurate ear detection model reduces the overall system reliability. |
| Angle Variety during Registration | Capturing multiple face angles (left, right, up, down) improves robustness to pose variations. |
| Face or Ear Missing           | The system gracefully falls back to single-modal recognition, leading to a slight reduction in confidence. |

### 🔥 Summary

| Mode                 | Ideal Accuracy | Real-World Accuracy |
|----------------------|----------------|---------------------|
| Face Only            | 95–98%         | 95–97%             |
| Ear Only             | 85–92%         | 85–90%             |
| **Combined (Face+Ear)** | **>99%** | **98–99.5%** |

**Key Takeaways:**

* ✅ When both face and ear are detected properly under good conditions, misidentification is extremely rare in a combined system.
* ✅ If only the face is reliably visible, the system can still maintain a relatively high accuracy of approximately 95–97%.
* ✅ If only the ear is clearly visible, the system can still provide a reasonable level of accuracy in the range of 85–90%.

---
## 📸 Sample Output
  - **Face and Ear** boxes drawn in real-time
  - **Name** with confidence score displayed
  - Fallback to face-only or ear-only if the other is missing

---

#### ✍️ Author
Developed by **Aman Kumar Dwiwedi**
