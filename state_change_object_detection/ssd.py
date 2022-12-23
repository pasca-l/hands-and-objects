import torchvision.models as model
import torch
import cv2
import numpy as np


ssd_model = model.detection.ssd300_vgg16(pretrained=True)
ssd_model.eval()

video_path = "../test_data/d.mp4"
video_save_path = "../test_data/result_ssd_d.mp4"

video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# v_size = (v_width, v_height)
v_size = (224,224)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_save_path, fourcc, fps, v_size)

for _ in range(int(frame_count)):
    ret, frame = video.read()

    # image = cv2.imread("../test_data/frame1.png")
    image = cv2.resize(frame, (224,224))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.unsqueeze(0).permute(0,3,1,2) # [b, c, h, w]

    pred = ssd_model(img)
    if len(pred) != 0:
        for i in range(len(pred[:5])):
            x1, y1, x2, y2 = pred[0]['boxes'].detach().numpy().astype('int32')[i]
            score = pred[0]['scores'].detach().numpy()[i]
            label = pred[0]['labels'].detach().numpy()[i]

            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0,0,255), thickness=3)

    writer.write(image.astype('uint8'))

writer.release()
video.release()
