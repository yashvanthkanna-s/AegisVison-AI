# AegisVision AI
**Egocentric Fall Detection & Emergency Alert System**

## Team Details
* **Team Name:** Stack Attack
* **Event:** HackZen 2026 Open Challenge

## Problem Statement
Falls are one of the leading causes of serious injuries among elderly people, especially those living alone. Most existing fall detection systems have some practical limitations.

* **CCTV-based systems** only work where cameras are installed. They also have blind spots and raise privacy concerns.
* **Smartwatches and wearable sensors** depend on batteries and are only useful if the person is actually wearing them.

We wanted to explore a different approach by using a first-person camera as the main source of information. Since the camera moves with the user, it can monitor falls in different environments without depending on fixed cameras.

## Objective
The goal of this project is to build a lightweight Computer Vision application that can analyze first-person (egocentric) videos and identify possible fall events.

Instead of only detecting a fall, the system also checks whether the person gets back up or remains still before recommending an emergency response.

This project was developed as a Minimum Viable Product (MVP) for the HackZen 2026 Open Challenge.

## Example Scenario
An elderly person named **Kumaran** is walking at home while wearing a body camera or smart glasses. He accidentally slips and falls.

AegisVision AI detects the sudden camera movement and marks it as a **Possible Fall**. It then monitors the next few seconds.

1. If **Kumaran** gets up and starts moving again, the system displays **"Fall Detected – User Recovered."**
2. If **Kumaran** remains motionless, the system displays **"Emergency Alert Recommended."**

This helps distinguish between a minor fall and a situation where immediate assistance may be needed.

## Our Approach
We started by asking a simple question:
*What does a fall look like from the person's own perspective?*

When someone wearing a body camera or smart glasses falls, the camera suddenly moves downward, rotates quickly, and often becomes still afterward.

Instead of training a large deep learning model, we decided to use Optical Flow from OpenCV to measure camera movement between frames. This gives us an estimate of how much the camera is moving and whether that movement matches the pattern of a fall.

To reduce false alarms, we added a simple Decision Engine.
* If movement resumes after a few seconds, the event is marked as **Recovered**.
* If almost no movement is detected after the fall, the system recommends an **Emergency Alert**.

This approach keeps the system lightweight, explainable, and suitable for running on a normal laptop.

## Methodology
The project is divided into a few simple modules.

* **Video Reader:** Reads the uploaded video frame by frame and passes it to the processing pipeline.
* **Optical Flow:** Uses OpenCV's Farneback Optical Flow algorithm to calculate motion between consecutive frames and generate a motion score.
* **Motion Analysis:** Tracks the motion score over time and looks for sudden spikes that may indicate a fall.
* **Decision Engine:** The Decision Engine manages four possible states:
  1. Normal Activity
  2. Fall Detected
  3. User Recovered
  4. Emergency Alert Recommended

  Instead of triggering an alert immediately, the system waits for a short period to check whether movement resumes.

* **Dashboard:** The application is built using Streamlit and provides a simple dashboard where users can:
  * Upload a video
  * View the processed output
  * Monitor the motion graph
  * See the current system status
  * View the final recommendation

## Technologies Used
* Python
* Streamlit
* OpenCV
* NumPy
* Matplotlib

## Dataset
The project is designed for first-person (egocentric) videos.

During development, we referred to the **EGOFALLS dataset** to understand the characteristics of egocentric fall scenarios.

For this hackathon prototype, the system was tested using representative first-person videos that simulate:
* Normal walking
* Fall followed by recovery
* Fall followed by no movement

These videos helped us tune the motion thresholds and validate the workflow within the available hackathon time.

## Installation
Clone the repository.
```bash
git clone https://github.com/yashvanthkanna-s/AegisVison-AI.git
cd AegisVision-AI
```

(Optional) Create a virtual environment.
```bash
python -m venv venv
```

Activate it.
Windows:
```powershell
venv\Scripts\activate
```

Install the required packages.
```bash
pip install -r requirements.txt
```

## Running the Project
Start the Streamlit application.
```bash
streamlit run app.py
```

Open the application in your browser.
Upload a first-person video and click **Start Analysis**. The system's Adaptive AI will automatically calculate a Z-score baseline for the environment—no manual slider tuning required!

The dashboard will display:
* Motion graph
* Current activity status
* Fall detection result
* Recovery status
* Emergency recommendation

## Results
The prototype was able to distinguish between different motion scenarios during testing.

It correctly demonstrated:
* Normal walking without unnecessary alerts
* Fall followed by recovery
* Fall followed by prolonged inactivity, resulting in an emergency recommendation

Since the project mainly relies on Optical Flow instead of computationally heavy object detection models, it performs well on a standard CPU without requiring a dedicated GPU.

The motion graph and status indicators also make the system's decision process easy to understand.

## Future Scope
There are several ways this project can be extended in the future.
* Live webcam support
* Smart glasses integration
* Body-worn camera deployment
* SMS or caregiver notifications
* Edge-device optimization
* Evaluation on larger real-world egocentric datasets

## References
* EGOFALLS Dataset
* OpenCV Documentation
* Streamlit Documentation
* NumPy Documentation
