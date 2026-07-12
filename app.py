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

st.set_page_config(page_title="AegisVision AI", layout="wide", page_icon="🛡️")

st.title("🛡️ AegisVision AI: Egocentric Fall Detection")
st.markdown("Analyzing first-person video to detect probable falls using camera motion (Optical Flow).")

# Sidebar for controls
st.sidebar.header("Settings")
st.sidebar.markdown("Use these sliders to tune the algorithms live.")
fall_threshold = st.sidebar.slider("Motion Spike Threshold", min_value=1.0, max_value=20.0, value=8.0, step=0.5, help="Minimum sudden motion to trigger a fall.")
recovery_time = st.sidebar.slider("Recovery Check Window (sec)", min_value=1, max_value=10, value=3, help="How long to wait after a fall before checking for recovery.")
emergency_threshold = st.sidebar.slider("Emergency Motion Threshold", min_value=0.1, max_value=5.0, value=1.5, step=0.1, help="If post-fall motion is below this, trigger emergency.")


uploaded_file = st.file_uploader("Upload an Egocentric Video (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    video_path = tfile.name
    
    if st.button("Start Analysis"):
        st.write("---")
        
        # UI Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Video Analysis")
            video_placeholder = st.empty()
            
        with col2:
            st.subheader("Current Status")
            status_placeholder = st.empty()
            confidence_placeholder = st.empty()
            
            st.subheader("Motion Intensity")
            graph_placeholder = st.empty()
            
            st.subheader("Event Timeline")
            timeline_placeholder = st.empty()
            
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
        timeline = []
        
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
                status_html = "### ✅ Normal Activity\nWaiting for sudden motion events..."
                status_color = "green"
            elif state == "Fall Detected":
                status_html = "### ⚠️ Possible Fall Detected\n- 🚨 **Sudden Motion** (Spike above threshold)\n- ⏳ Analyzing post-fall movement..."
                status_color = "orange"
            elif state == "Emergency Alert":
                status_html = "### 🚨 Emergency Assistance Recommended\n- 🚨 **Sudden Motion** (Fall detected)\n- 🛑 **Motion Stopped** (Post-fall motion extremely low)\n- ❌ **Recovery Not Detected**"
                status_color = "red"
            elif state == "Recovered":
                status_html = "### 🔄 User Recovered\n- 🚨 **Sudden Motion** (Fall detected)\n- 🏃 **Movement Resumed** (Post-fall motion is healthy)"
                status_color = "blue"
                
            status_placeholder.info(status_html)
            
            confidence_placeholder.progress(conf / 100.0)
            confidence_placeholder.caption(f"Confidence: {conf}%")
            
            # Update Graph every 5 frames to reduce UI lag
            if frame_idx % 5 == 0:
                fig = generate_motion_graph(all_magnitudes[-100:]) # Show last 100 frames
                graph_placeholder.pyplot(fig)
                plt.close(fig) # Prevent memory leak
                
            # Update Timeline if state changes significantly
            time_sec = frame_idx / fps
            time_str = time.strftime('%M:%S', time.gmtime(time_sec))
            
            if motion_results['is_sudden_motion']:
                timeline.append(f"{time_str} - Sudden Motion Detected (Mag: {flow_data['magnitude']:.1f})")
            if state == "Emergency Alert" and f"{time_str} - Emergency Alert Triggered" not in timeline:
                timeline.append(f"{time_str} - Emergency Alert Triggered")
            
            # Keep timeline concise
            if len(timeline) > 5:
                timeline = timeline[-5:]
            
            timeline_html = "<br>".join(timeline)
            timeline_placeholder.markdown(f"```text\n{timeline_html}\n```")
            
        reader.release()
        os.unlink(video_path)
        st.success("Analysis Complete!")
