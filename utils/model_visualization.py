import cv2
import torch
import numpy as np
import torchvision.models as models
from torchcam.methods import GradCAM


def main():
    """
    Visualizes middle layer weights using GradCAM.
    """

    image_path = "../test_data/test.jpg"

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    model = models.resnet101(weights='DEFAULT').eval()
    # rcnn = models.detection.fasterrcnn_resnet50_fpn()
    cam_extractor = GradCAM(model)

    img = torch.from_numpy(image.astype(np.float32)).clone()
    img = img.unsqueeze(0).permute(0,3,1,2)

    out = model(img)
    class_idx = 294
    cams = cam_extractor(class_idx, out)

    for i, cam in enumerate(cams):
        map = cam.squeeze(0).numpy()

        map_min = np.min(map)
        map_max = np.max(map)
        map = (map - map_min)/(map_max - map_min)

        map = cv2.resize(map, (224,224))
        map = (map*255).astype(np.uint8)

        heatmap = cv2.applyColorMap(map, cv2.COLORMAP_JET)
        output = heatmap * 0.5 + image * 0.5
        cv2.imwrite(f"../test_data/heatmap{i}.png", output)


if __name__ == '__main__':
    main()