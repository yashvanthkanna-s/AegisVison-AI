import cv2
import numpy as np

class OpticalFlowAnalyzer:
    def __init__(self):
        self.prev_gray = None

    def process_frame(self, frame):
        """
        Calculates dense optical flow using Farneback method.
        Returns the flow array, magnitude, and the visualization image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If it's the first frame, just store it and return zero motion
        if self.prev_gray is None:
            self.prev_gray = gray
            h, w = gray.shape
            return {
                "flow": np.zeros((h, w, 2), dtype=np.float32),
                "magnitude": 0.0,
                "direction": 0.0, # Mean angle
                "y_movement": 0.0 # Vertical movement
            }

        # Calculate Dense Optical Flow (Farneback)
        # Parameters: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        self.prev_gray = gray

        # Compute magnitude and angle of 2D vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # We are particularly interested in downward movement of the camera.
        # If the camera falls *down*, the visual field moves *up* in the frame.
        # In OpenCV image coordinates, y increases downwards.
        # So flow[..., 1] is the movement in Y. 
        # Mean of flow in Y:
        y_movement = np.mean(flow[..., 1])
        
        # Average magnitude across the whole frame
        avg_magnitude = np.mean(mag)
        avg_angle = np.mean(ang)

        return {
            "flow": flow,
            "magnitude": float(avg_magnitude),
            "direction": float(avg_angle),
            "y_movement": float(y_movement)
        }
