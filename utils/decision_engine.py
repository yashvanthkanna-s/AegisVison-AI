class DecisionEngine:
    def __init__(self, fps=30):
        self.fps = fps
        self.state = "Normal Activity"
        
        # Timers (in frames)
        self.frames_since_fall = 0
        self.recovery_check_window = int(fps * 3) # 3 seconds to check for recovery
        
        # Track motion during recovery window
        self.post_fall_magnitudes = []
        
        # Confidence score (0 to 100)
        self.confidence = 100 
        
    def update(self, motion_data):
        """
        Updates the state machine based on the latest motion data.
        Returns the current state and confidence.
        """
        is_fall = motion_data['is_possible_fall']
        current_mag = motion_data['current_magnitude']
        
        # State Transitions
        if self.state == "Normal Activity":
            if is_fall:
                self.state = "Fall Detected"
                self.frames_since_fall = 0
                self.post_fall_magnitudes = []
                self.confidence = 80 # High confidence a fall happened due to spike
                # Capture their normal baseline right before the fall
                self.pre_fall_baseline = motion_data.get('baseline_mean', 1.0)
                
        elif self.state == "Fall Detected":
            self.frames_since_fall += 1
            self.post_fall_magnitudes.append(current_mag)
            
            # Wait for the recovery window to pass
            if self.frames_since_fall >= self.recovery_check_window:
                recent_magnitudes = self.post_fall_magnitudes[-int(self.fps):]
                avg_post_movement = sum(recent_magnitudes) / len(recent_magnitudes) if recent_magnitudes else 0
                
                # ADAPTIVE RECOVERY LOGIC:
                # If their post-fall motion is extremely low (< 1.0) OR less than 40% of what 
                # their normal activity was before the fall, they are not recovering.
                if avg_post_movement < 1.0 or avg_post_movement < (self.pre_fall_baseline * 0.4):
                    self.state = "Emergency Alert"
                    self.confidence = 95
                else:
                    self.state = "Recovered"
                    self.confidence = 90
                    
        elif self.state in ["Recovered", "Emergency Alert"]:
            # Could reset to Normal after some time, or wait for manual reset
            # For this MVP, we can keep it in the final state or let it naturally reset if they start walking normally again.
            if self.state == "Recovered" and current_mag > 3.0:
                # If they are moving around well, go back to normal
                self.state = "Normal Activity"
                self.confidence = 100
                
        return {
            "state": self.state,
            "confidence": self.confidence
        }
