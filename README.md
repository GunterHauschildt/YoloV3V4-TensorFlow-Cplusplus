# YoloV3V4 Tensorflow & C++
YoloV3V4 in Tensorflow and C++. Before Ultralytics ruled the world!

https://github.com/user-attachments/assets/4c10c71b-b360-43fa-bffb-509fc9e9f09d

Before Ultralytics made Yolo easy, we had to either build Darknet's repo or home-brew our own Yolo code.

I used YoloV3 to deep-dive both Tensorflow and Yolo architectures.
Starting with https://github.com/zzh8829/yolov3-tf2, I
* modified to allow independent width, height
* modified to allow any number of anchors per the large, medium, small yolo branches. In particular,
this was to optimize for 1 anchor for single object at a single distance object detection - where all objects
would be the same size and need only one anchor.
* modified such that if the large, medium or small yolo branches had 0 anchors, that branch would be
removed from the model.

I later added YoloV4 to that YoloV3 code base. (Even if at time Yolo v5,6,7,8 had been already
introduced.)  Following the original paper I wrote the
* CSPDarknet53 layers
* PANetAndHead layers
* refactored the V3 code to share a common base.
* wrote a module to convert from the original darknet style weights to (my) V4 Tensorflow weights (for transfer
learning if this model was ever to be trained).

The V3 model has been trained on a custom data and is in use in a life-or-death appliciation. Life or death of
lifestock, but still life or death!



