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
        
        if len(self.magnitudes) > self.fps // 2: 
            # Get historical data excluding the current frame
            recent_mags = list(self.magnitudes)[:-1]
            historical_mean = float(np.mean(recent_mags))
            historical_std = float(np.std(recent_mags))
            
            # ADAPTIVE AI LOGIC: 
            # A fall is a massive anomaly. We trigger if motion is 4 standard deviations 
            # above the baseline, with a hard floor of 2.5 to prevent micro-jitters from triggering it.
            adaptive_threshold = max(2.5, historical_mean + (4 * historical_std))
            
            if mag > adaptive_threshold:
                is_sudden_motion = True
                is_downward_fall = True
                
        return {
            "is_sudden_motion": is_sudden_motion,
            "is_possible_fall": is_downward_fall,
            "current_magnitude": mag,
            "baseline_mean": np.mean(self.magnitudes) if len(self.magnitudes) > 0 else 0
        }
