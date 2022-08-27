from PIL import Image
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  _verbose=False)
im_pil = Image.open('prova1.png')
results = model(im_pil, size=112)

for p in results.pred[0]:
    x1,y1,x2,y2,conf,pred = list(p.numpy())
    class_name = results.names[int(pred)]
    print(class_name)
