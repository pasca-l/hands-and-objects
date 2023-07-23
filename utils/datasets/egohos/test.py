import glob
import os
import cv2

home = os.path.expanduser("~")
dataset_dir = os.path.join(home, "Documents/datasets")
# egohos_ca_dir = os.path.join(dataset_dir, "egohos_ca")
# files = glob.glob(f"{egohos_ca_dir}/*/image/*.jpg")

# for file in files:
#     if "71d7d352-00e1-455e-8779-90503c9404da" not in file:
#         continue
#     print(file)
#     image = cv2.imread(file)
#     label = cv2.imread(file.replace("image", "label").replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
#     _, _ = image.shape, label.shape


egohos_files = glob.glob(f"{dataset_dir}/egohos/test_outdomain/image/*.jpg")

for file in egohos_files:
    label_file = file.replace('image', 'label').replace('.jpg', '.png')
    filename = os.path.basename(file)
    label_filename = os.path.basename(label_file)

    egohos_ca_dir = os.path.join(dataset_dir, "egohos_ca/egohos/test_outdomain")
    os.symlink(file, os.path.join(egohos_ca_dir, "image", filename))
    os.symlink(label_file, os.path.join(egohos_ca_dir, "label", label_filename))
