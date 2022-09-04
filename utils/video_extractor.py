import os
import numpy as np
import cv2
from tqdm import tqdm


class Extractor():
    def __init__(self, data_dir, flatten_json):
        self.flatten_json = flatten_json
        self.data_dir = data_dir
        self.clip_dir = f"{data_dir}clips/"

    def trim_around_action(self):
        """
        Trims video to 8s clips containing action, under
        DATA_DIR/action_clips/{clip_id}_{start_frame}_{end_frame}.mp4
        """

        action_data_dir = f"{self.data_dir}action_clips/"
        os.makedirs(action_data_dir, exist_ok=True)

        for info in tqdm(self.flatten_json, desc='Trimming clip near action'):
            start_frame = info["clip_start_frame"]
            end_frame = info["clip_end_frame"]
            video_path = f"{self.clip_dir}{info['clip_uid']}.mp4"
            video_save_path = f"{action_data_dir}{info['clip_uid']}" +\
                              f"_{start_frame}_{end_frame}.mp4"

            if os.path.exists(video_save_path):
                continue

            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_size = (v_width, v_height)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_save_path, fourcc, fps, v_size)

            for i in range(end_frame + 1):
                ret, frame = video.read()
                if ret == True and start_frame <= i:
                    writer.write(frame)

            writer.release()
            video.release()
    
    def extract_frame_as_image(self):
        """
        Saves all frames of 8s clips containing action as png image, under 
        DATA_DIR/action_clip_frames/{clip_uid}/*.png
        """

        frame_dir = f"{self.data_dir}action_clip_frames/"
        os.makedirs(frame_dir, exist_ok=True)

        frame_dict = {}
        for info in tqdm(self.flatten_json, desc='Finding frames to extract'):
            frame_dict.setdefault(info["clip_uid"], set())

            start_frame = info["clip_start_frame"]
            end_frame = info["clip_end_frame"]
            frame_dict[info["clip_uid"]] |=\
                {i for i in range(start_frame, end_frame + 1)}

        existing_frame_dirs = [d for d in os.listdir(frame_dir)
                               if os.path.isdir(f"{frame_dir}{d}")]
        for d in tqdm(existing_frame_dirs,
                      desc='Excluding existing frames to extract'):
            try:
                frame_dict[d] -=\
                    {int(f[:-4]) for f in os.listdir(
                        f"{frame_dir}{d}")}
            except KeyError:
                continue

        for clip_id, frame_nums in tqdm(frame_dict.items(),
                                        desc='Extracting frames'):
            if len(frame_nums) == 0:
                continue

            frame_save_dir = f"{frame_dir}{clip_id}/"
            os.makedirs(frame_save_dir, exist_ok=True)

            video_path = f"{self.clip_dir}{clip_id}.mp4"
            video = cv2.VideoCapture(video_path)

            counter = 1
            while True:
                ret, frame = video.read()
                if ret == False:
                    break
                if counter in frame_nums:
                    frame_save_path = f"{frame_save_dir}{counter}.png"
                    cv2.imwrite(frame_save_path, frame)
                counter += 1

            video.release()

    def extract_frame_as_array(self):
        """
        Saves all frames clips as compressed npz (file containing multiple 
        numpy arrays) binary file, under 
        DATA_DIR/clip_arrays/{clip_uid}_{array_per_file}.npz
        """

        array_dir = f"{self.data_dir}clip_arrays/"
        os.makedirs(array_dir, exist_ok=True)

        for info in tqdm(self.flatten_json,
                         desc='Loading frames, and saving as npz'):
            clip_dir = f"{array_dir}{info['clip_uid']}"
            os.makedirs(clip_dir, exist_ok=True)
            video_path = f"{self.data_dir}clips/{info['clip_uid']}.mp4"

            video = cv2.VideoCapture(video_path)
            array_per_file = -(-video.get(cv2.CAP_PROP_FRAME_COUNT) // 10)

            counter = 0
            array_list = []

            while True:
                ret, frame = video.read()
                if ret == False:
                    break

                if counter != 0 and counter % array_per_file == 0:
                    array_save_path =\
                        f"{clip_dir}/{int(counter // array_per_file)}"
                    if not os.path.exists(f"{array_save_path}.npz"):
                        np.savez_compressed(array_save_path, array_list)

                array_list.append(frame)
                counter += 1

            video.release()