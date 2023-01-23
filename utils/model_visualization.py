import cv2
import torch
import numpy as np
import torchvision.models as models
from torchcam.methods import GradCAM


def main():
    """
    Visualizes middle layer weights using GradCAM.
    """

    image_path = "../objectness_classification/test_img/3dead068-318a-49c5-8036-e176e50cbf50/pnr_503.jpg"

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    model = models.resnet101(weights='DEFAULT')
    # rcnn = models.detection.fasterrcnn_resnet50_fpn()
    param_path = "../objectness_classification/logs/unet.ckpt"
    param = torch.load(param_path)["state_dict"]
    new_param = {}

    for k in param.keys():
        if "model.unet.encoder" in k:
            new_param[k[19:]] = param[k]

    model.load_state_dict(new_param, strict=False)
    model.eval()
    cam_extractor = GradCAM(model)

    img = torch.from_numpy(image.astype(np.float32)).clone()
    img = img.unsqueeze(0).permute(0,3,1,2)

    out = model(img)
    class_idx = out.squeeze(0).argmax().item()
    print(out.squeeze(0).argmax().item())
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
        cv2.imwrite(f"./heatmap{i}.jpg", output)


if __name__ == '__main__':
    main()