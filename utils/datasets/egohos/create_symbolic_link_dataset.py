import os
import glob


def main():
    home = os.path.expanduser("~")
    dataset_dir = os.path.join(home, "Documents/datasets")
    egohos_dir = os.path.join(dataset_dir, "egohos")
    files = glob.glob(f"{egohos_dir}/*/image/ego4d_*.jpg")

    egohos_ca_dir = os.path.join(dataset_dir, "egohos_ca")

    with open("./video_scene.txt", mode="r") as f:
        scenarios = f.readlines()
    video_id_to_scenario = {s.split(" ")[0]: " ".join(s.split(" ")[2:]).strip() for s in scenarios}

    test_scenarios = [
        "Baker", "Carpenter", "Car - commuting, road trip", "Farmer", "Maker Lab (making items in different materials, wood plastic and also electronics), some overlap with construction etc. but benefit is all activities take place within a few rooms", "Hosting a party", "Gardener", "Car mechanic", "Riding motorcycle", "Fixing PC", "BasketBall", "Eating", "Household management - caring for kids", "Playing board games", "Bike"
    ]
    for file in files:
        video_id = os.path.splitext(os.path.basename(file))[0].split("_")[1]

        label_file = file.replace('image', 'label').replace('.jpg', '.png')
        filename = os.path.basename(file)
        label_filename = os.path.basename(label_file)

        if video_id_to_scenario[video_id] in test_scenarios:
            egohos_ca_test_img_dir = os.path.join(egohos_ca_dir, "test_indomain/image")
            os.symlink(file, os.path.join(egohos_ca_test_img_dir, filename))

            egohos_ca_test_label_dir = os.path.join(egohos_ca_dir, "test_indomain/label")
            os.symlink(label_file, os.path.join(egohos_ca_test_label_dir, label_filename))

        # keep train and val splits, for original test group, append to train
        else:
            new_file = file.replace('egohos', 'egohos_ca')
            new_label_file = label_file.replace('egohos', 'egohos_ca')

            if "test_indomain" in file:
                new_file = new_file.replace('test_indomain', 'train')
                new_label_file = new_label_file.replace('test_indomain', 'train')

            os.symlink(file, new_file)
            os.symlink(label_file, new_label_file)


if __name__ == '__main__':
    main()
