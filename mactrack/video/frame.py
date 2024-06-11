class VideoFrames:
    def __init__(self):
        self.frames = []
        self.frames_v = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_frame_v(self, frame):
        self.frames_v.append(frame)