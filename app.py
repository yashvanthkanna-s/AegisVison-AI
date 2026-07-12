import streamlit as st
import cv2
import tempfile
import os
import time
import matplotlib.pyplot as plt

from utils.video_reader import VideoReader
from utils.optical_flow import OpticalFlowAnalyzer
from utils.motion_analysis import MotionAnalyzer
from utils.decision_engine import DecisionEngine
from utils.visualization import draw_flow, generate_motion_graph

if os.path.exists("logo.png"):
    st.set_page_config(page_title="AegisVision AI", layout="wide", page_icon="logo.png")
else:
    st.set_page_config(page_title="AegisVision AI", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

header_html = """
<div style="padding-top: 15px; margin-bottom: 25px;">
    <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0; line-height: 1.1;">
        <span style="background: -webkit-linear-gradient(45deg, #00D2FF, #3A7BD5); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AegisVision AI</span>
        <br>
        <span style="font-size: 2.0rem; color: #D0D0D0; font-weight: 500;">Egocentric Fall Detection</span>
    </h1>
    <p style="font-size: 1.2rem; color: #888888; margin-top: 10px;">
        Analyzing first-person video to detect probable falls using <b>camera motion (Optical Flow)</b>.
    </p>
</div>
"""

if os.path.exists("logo.png"):
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("logo.png", width=110)
    with col2:
        st.markdown(header_html, unsafe_allow_html=True)
else:
    st.markdown(header_html, unsafe_allow_html=True)

with st.expander("System Instructions", expanded=False):
    st.markdown('''
    Upload an egocentric video sequence below and initiate the analysis. 
    
    AegisVision AI will automatically process the optical flow motion vectors in real-time, providing a transparent and explainable assessment of the user's status.
    ''')


# Sidebar for controls
st.sidebar.header("System Settings")
st.sidebar.success("🧠 **Adaptive AI: Active**")
st.sidebar.markdown('''
AegisVision AI is currently running in fully adaptive mode. 

It calculates a dynamic **Z-Score baseline** for the user's standard movement, meaning it automatically adjusts to different video environments without requiring manual slider tuning.
''')

# We no longer need the sliders, but we can pass default configs to the engine if it asks
fall_threshold = None
recovery_time = 3
emergency_threshold = None


uploaded_file = st.file_uploader("Upload an Egocentric Video (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])

st.write("---")

# UI Layout (Always visible so the dashboard doesn't look empty)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Analysis")
    video_placeholder = st.empty()
    video_placeholder.info("Upload a video and click 'Start Analysis' to view the camera feed.")
    
with col2:
    st.subheader("Current Status")
    status_placeholder = st.empty()
    status_placeholder.info("Awaiting video feed...")
    confidence_placeholder = st.empty()
    
    st.subheader("Motion Intensity")
    graph_placeholder = st.empty()
    graph_placeholder.info("Graph will appear during analysis.")
    
    st.subheader("Analysis Summary")
    summary_placeholder = st.empty()
    summary_placeholder.info("Summary will be generated upon completion.")


if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close() # CRITICAL FOR WINDOWS: Close the file handle before OpenCV opens it!
    
    video_path = tfile.name
    
    if st.button("Start Analysis", type="primary", use_container_width=True):
        # Clear the initial info boxes
        video_placeholder.empty()
        status_placeholder.empty()
        graph_placeholder.empty()
        summary_placeholder.empty()
        
        # Initialize modules
        reader = VideoReader(video_path)
        info = reader.get_info()
        fps = info['fps']
        
        flow_analyzer = OpticalFlowAnalyzer()
        motion_analyzer = MotionAnalyzer(fps=fps)
        motion_analyzer.fall_magnitude_threshold = fall_threshold
        decision_engine = DecisionEngine(fps=fps)
        decision_engine.recovery_check_window = int(fps * recovery_time)
        decision_engine.emergency_threshold = emergency_threshold
        
        all_magnitudes = []
        has_fallen = False
        has_emergency = False
        
        # Process video frame-by-frame
        for frame_idx, frame in enumerate(reader.read_frames()):
            # 1. Optical Flow
            flow_data = flow_analyzer.process_frame(frame)
            
            # 2. Motion Analysis
            motion_results = motion_analyzer.analyze(flow_data)
            all_magnitudes.append(flow_data['magnitude'])
            
            # 3. Decision Engine
            decision = decision_engine.update(motion_results)
            state = decision['state']
            conf = decision['confidence']
            
            # 4. Visualization Update (update UI every N frames to save time, or every frame)
            # Drawing flow vectors on the frame
            vis_frame = draw_flow(frame, flow_data['flow'])
            # Convert BGR to RGB for Streamlit
            vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            
            video_placeholder.image(vis_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update Status Card (Smart Explainable UI)
            if state == "Normal Activity":
                status_html = "### Normal Activity\nWaiting for sudden motion events..."
                status_placeholder.success(status_html)
            elif state == "Fall Detected":
                status_html = "### Possible Fall Detected\n- **Sudden Motion** (Spike above threshold)\n- Analyzing post-fall movement..."
                status_placeholder.warning(status_html)
            elif state == "Emergency Alert":
                status_html = "### Emergency Assistance Recommended\n- **Sudden Motion** (Fall detected)\n- **Motion Stopped** (Post-fall motion extremely low)\n- **Recovery Not Detected**"
                status_placeholder.error(status_html)
            elif state == "Recovered":
                status_html = "### User Recovered\n- **Sudden Motion** (Fall detected)\n- **Movement Resumed** (Post-fall motion is healthy)"
                status_placeholder.info(status_html)
            
            confidence_placeholder.progress(conf / 100.0)
            confidence_placeholder.caption(f"Confidence: {conf}%")
            
            # Update Graph every 5 frames to reduce UI lag
            if frame_idx % 5 == 0:
                fig = generate_motion_graph(all_magnitudes[-100:]) # Show last 100 frames
                graph_placeholder.pyplot(fig)
                plt.close(fig) # Prevent memory leak
                
            # Track states for final summary
            if state == "Fall Detected":
                has_fallen = True
            elif state == "Emergency Alert":
                has_emergency = True
                
            summary_placeholder.info("⏳ Analyzing video stream in real-time... generating final report.")
            
        # Generate Final Summary
        if has_emergency:
            final_summary = "**🚨 CRITICAL INCIDENT:** AegisVision AI detected a severe fall. The user exhibited zero recovery motion during the observation window, indicating potential unconsciousness or severe injury. **Immediate medical dispatch is recommended.**"
            summary_placeholder.error(final_summary)
        elif has_fallen:
            final_summary = "**⚠️ MINOR INCIDENT:** AegisVision AI detected a sudden fall, but the user exhibited healthy movement shortly after. This was likely a trip or a soft fall. No emergency response is required at this time."
            summary_placeholder.warning(final_summary)
        else:
            final_summary = "**✅ NO INCIDENTS:** The user maintained normal activity levels throughout the observation period. No sudden motion spikes or fall signatures were detected."
            summary_placeholder.success(final_summary)
            
        max_mag = max(all_magnitudes) if all_magnitudes else 0
        reader.release()
        os.unlink(video_path)
        st.success(f"Analysis Complete! (Diagnostic Info: The peak motion spike during this video was {max_mag:.1f})")
