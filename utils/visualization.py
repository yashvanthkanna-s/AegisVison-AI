import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_flow(img, flow, step=16):
    """
    Draws the optical flow vectors on an image.
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def generate_motion_graph(magnitudes):
    """
    Creates a matplotlib figure showing the motion intensity over time.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Plotting motion magnitude
    ax.plot(magnitudes, color='cyan', linewidth=2)
    ax.fill_between(range(len(magnitudes)), magnitudes, color='cyan', alpha=0.3)
    
    ax.set_title("Motion Intensity (Camera Movement)", color='white')
    ax.set_xlabel("Frames", color='lightgray')
    ax.set_ylabel("Magnitude", color='lightgray')
    
    # Styling for dark mode (fits well with Streamlit dark themes)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='lightgray')
    for spine in ax.spines.values():
        spine.set_color('gray')
        
    plt.tight_layout()
    return fig
