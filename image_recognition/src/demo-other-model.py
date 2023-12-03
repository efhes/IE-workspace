import time

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import torchvision.transforms as T
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights
import cv2
from torchvision import transforms

import json
import imutils
from imutils.video import FPS
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

def class_id_to_label(i):
    return labels[i]
    
def tensor_imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    #plt.imshow(inp)
    cv2.imshow(title, inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

from picamera2 import Picamera2

cv2.startWindowThread()

picam2 = Picamera2()
lsize = (640, 480)
video_config = picam2.create_video_configuration(
    #main={"size": (1280, 720), "format": "RGB888"},
    main={"size": lsize, "format": "RGB888"},
    lores={"size": lsize, "format": "YUV420"})

picam2.configure(video_config)
picam2.start()

model_path = '../models/custom_model_ft.pth'
#model_path = '../models/model_ft-2.pth'
#model_path = '../models/model_conv.pth'
with open('./imagenet-simple-labels/imagenet-simple-labels.json') as f:
    labels = json.load(f)

labels = ['bottle', 'mouse', 'pencilcase', 'raspberry']

num_classes = len(labels)
COLORS = np.random.uniform(0, 255, size=(len(labels), 3))
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.quantized.engine = 'qnnpack'

# MobileNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MobileNet
weights = MobileNet_V2_Weights.IMAGENET1K_V1

# EfficientNet
#preprocess = transforms.Compose([
#        transforms.ToPILImage(),
#        weights.transforms(),
#    ])

# EfficientNet
#weights = EfficientNet_B0_Weights.IMAGENET1K_V1

#net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
#net = models.efficientnet_b0(weights='DEFAULT')
net = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

print(f'\nModifying pre-trained model: last fc layer updated to {num_classes} classes\n')
for i in range(num_classes):
    print(f'\t[{i}][{labels[i]}]')
    
net.classifier[1] = nn.Linear(1280, num_classes) #FFM para MobileNet_v2 fine-tuneado con Colab
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

fps = FPS().start()

with torch.set_grad_enabled(False):
    while True:
        # read frame
        #ret, image = cap.read()
        #if not ret:
        #    raise RuntimeError("failed to read frame")
    
        # convert opencv output from BGR to RGB
        #image = image[:, :, [2, 1, 0]]

        #read image 
        image = picam2.capture_array("main")
        
        #read image 
        #image = cv2.imread('/home/pi/workspace/image_recognition/data/DATASET/val/mouse/mouse0000.jpg')
        
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
            for idx, val in top[:num_classes]:
                print(f"{val.item()*100:.2f}% {class_id_to_label(idx)}")
    
        # Recognition result
        predicted_id = top[0][0]
        predicted_conf = top[0][1]
        
        # Draw the prediction on the frame
        #image = imutils.resize(image, width=400)
        startX=0
        startY=0
        endX=200
        endY=200
        #image = np.ascontiguousarray(image, dtype=np.uint8)
        label = "{}: {:.2f}%".format(class_id_to_label(predicted_id), predicted_conf * 100) 
        text = "{} - fps: {:.2f}".format(label, current_fps)
        #cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[predicted_id], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        pt1 = (int(startX), int(y))
        cv2.putText(image, text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[predicted_id], 2)
        
        # show the output frame
        cv2.imshow("Result", image)
        #tensor_imshow(input_tensor)
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
