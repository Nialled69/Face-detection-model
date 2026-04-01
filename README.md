# Face-Based Biometric Authentication for Secure Banking

This project implements a secure, face-based biometric authentication system designed for banking transactions. It utilizes deep learning for facial feature extraction and verification, combined with an active liveness detection mechanism to prevent spoofing attacks.

<p></p>
<p></p>

## 🥇Features

* **Deep Learning Authentication**: Leverages the **ArcFace** model (ResNet50 backbone) for high-precision facial feature extraction (512-dimensional embeddings).
* **Active Liveness Detection**: Implements a defense mechanism against spoofing (photo/video playback attacks) using **Eye-blink dynamics** via MediaPipe Face Mesh.
* **High Performance**: Demonstrated a verification accuracy of **99.83%** under experimental conditions.
* **Microservice Design**: Built as a Python-based microservice for seamless integration into banking infrastructures.

## 🥈Technical Stack
The system operates on a multi-stage pipeline to ensure both security and speed:

1.  **Face Acquisition**: Captures real-time frames from a camera feed.
2.  **Liveness Verification**: Calculates the **Eye Aspect Ratio (EAR)** to detect natural blinking patterns.
3.  **Feature Extraction**: Generates unique facial signatures using ArcFace.
4.  **Matching**: Compares live signatures against the database using **Cosine Similarity**.
5.  **Decision**: Grants or denies access based on a predefined similarity threshold.

## 🥉Work Documentation
The methodology and experimental results of this project are detailed here :

> **[Face-Based Biometric Authentication for Secure Banking Transactions Using Deep Learning](https://drive.google.com/file/d/1HpnP9MCIGQlpDHITaS2Lb34Fw0ZUdM0l/view?usp=drive_link)**

## 🈴Prerequisites

* Python 3.x
* OpenCV
* MediaPipe
* TensorFlow / PyTorch (depending on your ArcFace implementation)
* NumPy

## 🤹System Workflow

a) Face Capture: Real-time acquisition via camera.


b) Liveness Detection: Verification of a live user through eye-blink detection.


c) Feature Extraction: Processing the face through ArcFace to create a unique embedding.


d) Identity Matching: Comparing the live embedding with stored data using Cosine Similarity.


e) Transaction Authorization: Approving or rejecting the transaction based on the similarity score.

## 👥 Authors

* **Soumyadeep Basu**
* **Sarthik Dasgupta**
* **Ankit Das**
* **Sourasish Biswas**
