import glob
import os
from json_handler import JsonHandler


def main():
    ann_dir = '~/data/ego4d/annotations/'
    task_name = 'fho_scod'

    json_handler = JsonHandler(task_name)
    json_partial_name = ann_dir + task_name
    json_dict = {
        "train": json_handler(f"{json_partial_name}_train.json"),
        "val": json_handler(f"{json_partial_name}_val.json"),
        # "infer": json_handler(f"{json_partial_name}_test_unannotated.json")
    }

    files_path = glob.glob("~/data/ego4d/clips/*")
    files = [os.path.splitext(os.path.basename(i))[0] for i in files_path]

    clips = set()
    for flatten_json in json_dict.values():
        for data in flatten_json:
            if data['clip_uid'] not in files:
                clips.add(data['clip_uid'])

    for i in clips:
        print(i, end=' ')


if __name__ == '__main__':
    main()