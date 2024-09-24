import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from numpy.typing import NDArray
from utils.video_stream import VideoStream
from yolov3v4_tf import (
    get_yolo,
    YoloV3, YoloV4, YoloV3V4Tiny,
    yolo_boxes, yolo_nms,
    decode_predictions, cv_to_tf
)


class YoloDetector:
    def __init__(self,
                 name: str,
                 is_v4: bool,
                 is_tiny: bool,
                 num_classes: int = 80,
                 weights_folder: str = os.path.join("..", "data", "tf"),
                 anchors_file: str or None = None
                 ):
        self._name = name
        self._is_v4 = is_v4
        self._is_tiny = is_tiny
        if not self._is_v4 or (self._is_v4 and self._is_tiny):
            self._size_xy = (416, 416)
        else:
            self._size_xy = (608, 608)
        self._channels = 3
        self._anchors_file = anchors_file

        weights_file = 'darknet_weights'
        weights_file = weights_file + "_v3" if not self._is_v4 else weights_file + "_v4"
        weights_file = weights_file if not self._is_tiny else weights_file + "_tiny"
        weights_file = os.path.join(weights_folder, weights_file)
        self._model, _ = get_yolo(num_classes, self._size_xy, self._channels, self._anchors_file,
                                  self._is_v4, self._is_tiny)
        _ = self._model.load_weights(weights_file)

    def size_xy(self):
        return self._size_xy

    def predict(self, X):
        return self._model.predict(X)


def main():

    class VideoClip:
        def __init__(self,
                     name: str,
                     file_name: str,
                     class_names: list[str]
                     ):
            self._name = name
            self._file_name = os.path.join("..", file_name)
            self._class_names = class_names

        def start(self):
            return self._start

        def end(self):
            return self._end

        def file_name(self):
            return self._file_name

        def class_names(self):
            return self._class_names

    video_clips = [
        VideoClip("Biplanes", "13066173-uhd_3840_2160_50fps.mp4", "aeroplane"),
        VideoClip("RaceTrack", "3818936-hd_1920_1080_30fps.mp4", "car"),
        VideoClip("Elephants", "7499056-hd_1280_720_60fps.mp4",  "elephant"),
        VideoClip("Beach", "pexels-7147374.mp4", "person"),
    ]

    classes_filename = os.path.join("..", "data", 'classes.names.coco')
    class_map = {name: idx for idx, name in enumerate(open(classes_filename).read().splitlines())}
    class_names = [class_name.strip() for class_name in class_map]
    num_classes = len(class_names)

    yolo_detector = YoloDetector("Yolo V4", True, False, num_classes)

    win_name = "Yolo"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    draw_size = (960, 540)
    cv.resizeWindow(win_name, draw_size)

    for video_clip in video_clips:
        video_stream = VideoStream(video_clip.file_name())
        while (frame := video_stream.next_frame()) is not None:

            image_raw = frame[0]

            image_draw = cv.resize(image_raw, draw_size)
            image_data = cv_to_tf(image_raw, yolo_detector.size_xy())

            predictions = yolo_detector.predict(image_data)
            predictions = decode_predictions(predictions, image_draw.shape[:2], .50)
            for prediction in predictions:
                (x0y0, x1y1), cls, confidence = prediction
                if class_names[cls] in video_clip.class_names():
                    p1 = np.round(x0y0).astype(np.int32)
                    p2 = np.round(x1y1).astype(np.int32)
                    image_draw = cv.rectangle(image_draw, p1, p2, (255, 255, 0), 3)

            cv.imshow(win_name, image_draw)
            cv.waitKey(1)


if __name__ == '__main__':
    main()
