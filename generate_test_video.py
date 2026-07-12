import cv2
import numpy as np
import os

def create_synthetic_video(filename, scenario, fps=30, duration=8):
    """
    scenario: 'normal', 'recovery', 'emergency'
    Generates a synthetic egocentric video to test the pipeline.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    # Create a base image (e.g., a room with a floor line and some features)
    base_img = np.zeros((height*3, width*3, 3), dtype=np.uint8)
    # Draw grid/features so optical flow has something to track
    for y in range(0, height*3, 40):
        cv2.line(base_img, (0, y), (width*3, y), (200, 200, 200), 2)
    for x in range(0, width*3, 40):
        cv2.line(base_img, (x, 0), (x, height*3), (200, 200, 200), 2)
        
    # Center crop coordinates
    cx, cy = width, height
    
    for i in range(total_frames):
        # 0-3 seconds: Walking (slight bobbing)
        if i < fps * 3:
            y_offset = int(np.sin(i * 0.5) * 5)
            x_offset = int(np.cos(i * 0.25) * 3)
        # 3-4 seconds: The Fall
        elif i >= fps * 3 and i < fps * 4:
            if scenario in ['recovery', 'emergency']:
                # Massive upward shift of the visual field (camera falls down)
                fall_progress = (i - fps * 3) / fps
                y_offset = int(fall_progress * height * 0.8) # Move view up heavily
                x_offset = 0
            else:
                # Normal walking continues
                y_offset = int(np.sin(i * 0.5) * 5)
                x_offset = int(np.cos(i * 0.25) * 3)
        # 4-8 seconds: Post-fall
        else:
            if scenario == 'recovery':
                # Resume walking/movement
                y_offset = int(np.sin(i * 0.5) * 5) + int(height * 0.8)
                x_offset = int(np.cos(i * 0.25) * 3)
            elif scenario == 'emergency':
                # Complete stillness
                y_offset = int(height * 0.8)
                x_offset = 0
            else:
                # Normal walking continues
                y_offset = int(np.sin(i * 0.5) * 5)
                x_offset = int(np.cos(i * 0.25) * 3)
                
        # Crop the base image based on offset to simulate camera movement
        frame = base_img[cy + y_offset : cy + y_offset + height, cx + x_offset : cx + x_offset + width]
        # Ensure exact size
        frame = cv2.resize(frame, (width, height))
        
        # Add some text to know what frame we are looking at visually
        cv2.putText(frame, f"Frame: {i} | Scenario: {scenario}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Generated {filename}")

if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    create_synthetic_video('dataset/test_normal.mp4', 'normal')
    create_synthetic_video('dataset/test_recovery.mp4', 'recovery')
    create_synthetic_video('dataset/test_emergency.mp4', 'emergency')
