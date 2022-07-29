import json
from tqdm import tqdm


def main():
    clip_dir = "../../../data/ego4d/clips/"
    ann_dir = "../../../data/ego4d/annotations/"
    train_json_file = f"{ann_dir}fho_scod_train.json"

    frame_dict
    json_data = json.load(open(train_json_file, 'r'))
    for data in tqdm(json_data['clips'], desc='Loading json'):
        json_dict = {
            "video_uid": data['video_uid'],
            "clip_uid": data['clip_uid'],
            "pre_frame": data['pre_frame']['frame_num'],
            "pnr_frame": data['pnr_frame']['frame_num'],
            "post_frame": data['post_frame']['frame_num']
        }
        print(data)


if __name__ == '__main__':
    main()