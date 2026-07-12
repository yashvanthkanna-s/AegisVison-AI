# AegisVision AI: Egocentric Fall Detection System

## Project Title
AegisVision AI

## Team Details
*   **Team Name:** Team Aegis
*   **Hackathon:** HackZen 2026 OPEN CHALLENGE

## Problem Statement
Falls are one of the leading causes of serious injury among elderly individuals. Existing fall detection systems rely heavily on fixed CCTV cameras (which have blind spots), or wearable sensors/smartwatches (which must be consistently worn and charged). There is a critical need for an intelligent, portable, vision-based system capable of recognizing probable falls from the user's own perspective.

## Objective
To develop a Computer Vision-based healthcare MVP that analyzes first-person (egocentric) videos to detect sudden movement patterns consistent with a fall, and to intelligently determine if the user has recovered or if an emergency response is needed.

## Proposed Solution
AegisVision AI uses the camera as a sensor. By employing dense Optical Flow (Farneback), it tracks motion vectors between frames to identify sudden, high-intensity downward motion (a fall). Post-impact, it monitors the motion graph to determine if the individual has resumed movement (Recovered) or remains still (Emergency Alert Recommended).

## Technologies Used
*   **Programming Language:** Python 3.10+
*   **Frontend Dashboard:** Streamlit
*   **Computer Vision:** OpenCV (cv2)
*   **Data Processing & Analytics:** NumPy, Pandas
*   **Data Visualization:** Matplotlib
*   **Version Control:** Git & GitHub

## Dataset
*   **Name:** EGOFALLS (Representative sample used for hackathon MVP threshold calibration).
*   *Note: Video samples used in the demonstration showcase everyday walking and sudden egocentric falls.*

## Methodology / Model Architecture
This project deliberately avoids "black-box" deep learning models in favor of explainable, real-time classical Computer Vision.

1.  **Video Reader:** Extracts frames sequentially from the uploaded video.
2.  **Optical Flow (`cv2.calcOpticalFlowFarneback`):** Calculates dense motion vectors across the entire visual field to output an average Motion Magnitude.
3.  **Motion Analyzer:** Maintains a rolling history of motion magnitude. A sudden, massive spike (above a defined threshold) indicates a rapid orientation shift (potential fall).
4.  **Decision Engine:** State machine logic.
    *   `Normal Activity` -> (Spike Detected) -> `Fall Detected`
    *   Waits for a 3-second post-fall window. If average motion is near zero, transitions to `Emergency Alert`. If motion resumes, transitions to `Recovered`.
5.  **Visualization:** Streamlit dashboard renders the annotated frames, real-time Matplotlib motion graph, and event timeline.

## Installation & Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <YOUR-GITHUB-REPO-URL>
    cd AegisVision-AI
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage Instructions

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
2.  **Interact with the Dashboard:**
    *   Open your browser to `http://localhost:8501`.
    *   Use the sidebar slider to adjust the "Fall Magnitude Threshold" (useful if your test video has a very high or very low framerate).
    *   Upload an egocentric video (MP4 format).
    *   Click "Start Analysis".
    *   Watch the frame-by-frame analysis, the real-time motion intensity graph, and the Decision Engine status updates.

## Results and Outputs
*   **Lightweight processing:** Capable of running without GPU acceleration.
*   **Explainable UI:** The motion graph directly shows *why* the system triggered an alert (a visible spike followed by a flatline).
*   **Actionable States:** Differentiates between a fall with a recovery (no ambulance needed) and a severe fall (ambulance needed).

## Future Scope
*   **Smart Glasses Integration:** Porting the logic to run on AR/Smart Glasses hardware.
*   **Edge AI Optimization:** Converting the Python pipeline to C++ for lower latency on wearable edge devices.
*   **Caregiver Notifications:** Integrating Twilio or similar APIs to send SMS alerts when an `Emergency Alert` is triggered.

## References
*   OpenCV Documentation for Optical Flow: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
*   EGOFALLS Dataset: (Link/Citation to the academic source if applicable).
