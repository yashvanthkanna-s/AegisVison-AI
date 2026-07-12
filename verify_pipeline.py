import os
from utils.video_reader import VideoReader
from utils.optical_flow import OpticalFlowAnalyzer
from utils.motion_analysis import MotionAnalyzer
from utils.decision_engine import DecisionEngine

def test_pipeline(video_path):
    print(f"\n{'='*50}")
    print(f"Testing Video: {video_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
        
    reader = VideoReader(video_path)
    info = reader.get_info()
    fps = info['fps']
    print(f"Video Info: FPS={fps}, Frames={info['total_frames']}")
    
    flow_analyzer = OpticalFlowAnalyzer()
    motion_analyzer = MotionAnalyzer(fps=fps)
    decision_engine = DecisionEngine(fps=fps)
    
    # We'll print state changes
    last_state = None
    
    for frame_idx, frame in enumerate(reader.read_frames()):
        flow_data = flow_analyzer.process_frame(frame)
        motion_results = motion_analyzer.analyze(flow_data)
        decision = decision_engine.update(motion_results)
        
        current_state = decision['state']
        
        if frame_idx % fps == 0:  # Print once a second (approx) to show it's working
            print(f"Frame {frame_idx:03d} | Mag: {flow_data['magnitude']:5.1f} | Y-Mov: {flow_data['y_movement']:5.1f} | State: {current_state}")
            
        # Or print immediately if state changes
        if current_state != last_state:
            print(f"*** STATE CHANGE at Frame {frame_idx}: {last_state} -> {current_state} (Mag: {flow_data['magnitude']:.1f})")
            last_state = current_state
            
    print(f"Final State for {os.path.basename(video_path)}: {current_state}")
    reader.release()

if __name__ == "__main__":
    test_pipeline("dataset/test_normal.mp4")
    test_pipeline("dataset/test_recovery.mp4")
    test_pipeline("dataset/test_emergency.mp4")
