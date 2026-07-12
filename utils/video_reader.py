import cv2

class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Avoid division by zero if fps is not read correctly
        if self.fps == 0 or self.fps != self.fps:
            self.fps = 30.0

    def get_info(self):
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_sec": self.total_frames / self.fps if self.fps else 0
        }

    def read_frames(self):
        """Yields frames one by one."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            # 640 width keeps aspect ratio
            aspect_ratio = self.height / self.width
            new_width = 640
            new_height = int(new_width * aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
            
            yield frame

    def release(self):
        if self.cap:
            self.cap.release()
