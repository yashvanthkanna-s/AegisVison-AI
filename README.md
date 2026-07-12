# AegisVision AI

## Project Title
AegisVision AI - Egocentric Fall Detection System

## Team Details
**Team Name:** Team Aegis
**Event:** HackZen 2026 OPEN CHALLENGE

---

## 1. The Problem Statement
Falls are a leading cause of injury for elderly individuals. When we looked at existing fall detection solutions, we noticed a few major flaws:
* **CCTV Cameras:** They have blind spots, only work in specific rooms, and raise privacy concerns.
* **Smartwatches/Wearables:** These rely heavily on accelerometers and are often forgotten on the nightstand or have dead batteries. 

We realized there is a need for a portable, vision-based system that travels *with* the user. 

## 2. Objective
Our goal for this 24-hour hackathon was to build a working prototype (MVP) that analyzes first-person (egocentric) video to not only detect if a fall has occurred, but to intelligently decide if the person actually needs emergency assistance.

## 3. Our Approach & Proposed Solution
Instead of relying on heavy, "black-box" deep learning models (like YOLO) which can be slow and hard to explain, we decided to use **the camera itself as the sensor**. 

We approached the solution by asking: *What does a fall look like from the first-person perspective?*
When a person falls, the camera's view of the floor rushes upward rapidly. We realized we could track this massive shift in pixels using classical Computer Vision. 

By analyzing the motion vectors of the video frame-by-frame, our system looks for a massive spike in movement. More importantly, we built a **Decision Engine** to prevent false alarms. If a fall is detected, the system waits 3 seconds. If the person gets back up, it marks them as "Recovered". If they lie completely still, it triggers an "Emergency Alert".

## 4. Methodology / Model Architecture
We broke the architecture down into modular steps:
1. **Video Reader:** Extracts the video frame-by-frame for real-time processing.
2. **Optical Flow (Farneback):** Calculates dense motion vectors across the entire screen. This gives us a raw "Motion Magnitude" score.
3. **Motion Analyzer:** Maintains a rolling history of the user's movement speed. It flags sudden, intense spikes as potential falls.
4. **Decision Engine:** A custom state machine that manages the logic:
   - `Normal Activity`
   - `Fall Detected` (Spike identified)
   - `Recovered` (Motion resumes after the fall)
   - `Emergency Alert` (Absolute stillness after the fall)
5. **Dashboard:** A Streamlit web app that provides a live motion graph and explains the engine's real-time decisions to the caregiver.

## 5. Technologies Used
* **Language:** Python
* **Frontend UI:** Streamlit
* **Computer Vision:** OpenCV (cv2)
* **Data Processing:** NumPy
* **Visualization:** Matplotlib

## 6. Dataset
* **Name:** EGOFALLS 
* **Note:** To quickly test and tune our thresholds during the 24-hour hackathon, we used synthetic first-person video generations (walking, recovering, and emergency scenarios) inspired by the EGOFALLS dataset parameters.

---

## 7. Installation & Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/yashvanthkanna-s/AegisVison-AI.git
   cd AegisVision-AI
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## 8. Usage Instructions
1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to `http://localhost:8501`.
3. Use the sidebar sliders to tune the AI's sensitivity (Motion Threshold, Recovery Time).
4. Upload an egocentric `.mp4` video (you can use the test videos located in the `dataset/` folder).
5. Click **Start Analysis** and watch the real-time motion graph and Decision Engine status board.

## 9. Results and Outputs
Our MVP successfully differentiates between normal walking, a recovered fall (e.g., tripping but standing back up), and a severe emergency fall (e.g., losing consciousness). Because it uses Optical Flow rather than heavy object detection, it runs very efficiently on standard CPUs without requiring expensive GPUs. The Streamlit dashboard successfully makes the AI's decision process completely transparent and explainable.

## 10. Future Scope
* **Wearable Integration:** Deploying the code onto AR Smart Glasses or dedicated body-worn cameras.
* **Edge Optimization:** Converting the Python logic into C++ to run entirely offline on low-power edge devices.
* **Caregiver API:** Adding Twilio integration to automatically text a family member if the `Emergency Alert` state is triggered.

## 11. References
* EGOFALLS Dataset: Xueyi-Wang/EGOFALLS (GitHub)
* OpenCV Optical Flow Documentation: docs.opencv.org
