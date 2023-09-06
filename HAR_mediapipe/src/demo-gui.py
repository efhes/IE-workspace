import time

import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights

import cv2
from PIL import Image

import json

import imutils
from imutils.video import FPS

from picamera2 import Picamera2

def class_id_to_label(i):
    return labels[i]

cv2.startWindowThread()

picam2 = Picamera2()
lsize = (640, 480)
video_config = picam2.create_video_configuration(
    #main={"size": (1280, 720), "format": "RGB888"},
    main={"size": lsize, "format": "RGB888"},
    lores={"size": lsize, "format": "YUV420"})

picam2.configure(video_config)
picam2.start()

with open('./imagenet-simple-labels/imagenet-simple-labels.json') as f:
    labels = json.load(f)

COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

torch.backends.quantized.engine = 'qnnpack'


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True)

# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

fps = FPS().start()

with torch.no_grad():
    while True:
        #read image 
        image = picam2.capture_array("main")

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)
        # do something with output ...

        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            current_fps = frame_count / (now-last_logged)
            print(f"{current_fps} fps")
            
            last_logged = now
            frame_count = 0
        
            top = list(enumerate(output[0].softmax(dim=0)))
            top.sort(key=lambda x: x[1], reverse=True)
            for idx, val in top[:10]:
                print(f"{val.item()*100:.2f}% {class_id_to_label(idx)}")
            
        # Recognition result
        predicted_id = top[0][0]
        predicted_conf = top[0][1]
        
        # draw the prediction on the frame
        #image = imutils.resize(image, width=400)
        startX=0
        startY=0
        endX=200
        endY=200
        image = np.ascontiguousarray(image, dtype=np.uint8)
        label = "{}: {:.2f}%".format(class_id_to_label(predicted_id), predicted_conf * 100) 
        text = "{} - fps: {:.2f}".format(label, current_fps)
        #cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        pt1 = (int(startX), int(y))
        cv2.putText(image, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
        # show the output frame
        cv2.imshow("Result", image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
