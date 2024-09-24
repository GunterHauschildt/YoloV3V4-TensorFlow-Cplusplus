from numpy.typing import NDArray
import cv2 as cv


class VideoStream:
    def __init__(self, path: str):
        self._name = path
        self._video_capture = cv.VideoCapture(path)
        self._x = int(self._video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self._y = int(self._video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._video_capture.get(cv.CAP_PROP_FPS)
        self._frames = int(self._video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        self._frame_num = 0

    def is_open(self) -> bool:
        return self._video_capture.isOpened()

    def size_xy(self) -> tuple[int, int]:
        return self._x, self._y

    def fps(self) -> float:
        return self._fps

    def mspf(self) -> float:
        return round(1000. / self._fps)

    def next_frame(self) -> tuple[NDArray, int] or None:
        success, frame = self._video_capture.read()
        if not success:
            return None
        else:
            frame_num = self._frame_num
            self._frame_num += 1
            return frame, frame_num

    def total_frames(self) -> int:
        return self._frames

    def reset(self, frame_num: int = 0) -> tuple[NDArray, int] or None:
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = self._video_capture.read()
        if not success:
            return None
        else:
            self._video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_num)
            return frame, frame_num

    def fast_forward(self, seconds: float) -> tuple[NDArray, int] or None:
        frames = seconds * self._fps
        self._frame_num += frames
        return self.reset(self._frame_num)

    def rewind(self, seconds: float) -> tuple[NDArray, int] or None:
        frames = seconds * self._fps
        self._frame_num -= frames
        self._frame_num = max(self._frame_num, 0)
        return self.reset(self._frame_num)

    def name(self) -> str:
        return self._name

    def __repr__(self):
        return ", ".join([self._name, self._x, self._y])

    def __str__(self):
        self.__repr__()

