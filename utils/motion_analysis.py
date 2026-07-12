from collections import deque
import numpy as np

class MotionAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        # Keep a short history of magnitudes to detect spikes
        # A 1-second window
        self.history_len = int(fps)
        self.magnitudes = deque(maxlen=self.history_len)
        self.y_movements = deque(maxlen=self.history_len)
        
        # Thresholds (can be tuned later)
        self.fall_magnitude_threshold = 8.0  # Spikes above this mean sudden movement
        
    def analyze(self, flow_data):
        """
        Analyzes current flow data against history to detect sudden movements.
        """
        mag = flow_data['magnitude']
        y_mov = flow_data['y_movement']
        
        self.magnitudes.append(mag)
        self.y_movements.append(y_mov)
        
        is_sudden_motion = False
        is_downward_fall = False
        
        if len(self.magnitudes) > self.fps // 2: # Wait for at least half a second of data
            # Check if current magnitude is a spike compared to the mean of the history
            historical_mean = np.mean(list(self.magnitudes)[:-1]) if len(self.magnitudes) > 1 else mag
            
            # If current motion is significantly higher than historical and above absolute threshold
            if mag > self.fall_magnitude_threshold and mag > historical_mean * 2.5:
                is_sudden_motion = True
                
            # For egocentric falls, falling forward/down means visual field moves UP
            # So y_movement (OpenCV y goes down) becomes negative. 
            # Or if they fall backward, visual field moves DOWN (y_movement positive).
            # We'll consider any intense Y-shift combined with magnitude spike as a potential fall.
            if is_sudden_motion and abs(y_mov) > 2.0:
                is_downward_fall = True
                
        return {
            "is_sudden_motion": is_sudden_motion,
            "is_possible_fall": is_downward_fall,
            "current_magnitude": mag
        }
