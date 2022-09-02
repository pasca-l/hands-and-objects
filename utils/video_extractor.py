import os
import cv2
from tqdm import tqdm


class Extractor():
    def __init__(self, data_dir, flatten_json):
        self.flatten_json = flatten_json
        self.action_data_dir = f"{data_dir}actions/"
        os.makedirs(self.action_data_dir, exist_ok=True)
        self.action_frame_dir = f"{data_dir}action_frames/"
        os.makedirs(self.action_frame_dir, exist_ok=True)

    def trim_around_action(self):
        """
        Trims video to 8s clips containing action, under
        DATA_DIR/actions/*.mp4
        """
        for info in tqdm(self.flatten_json, desc='Trimming clip near action'):
            start_frame = info["clip_start_frame"]
            end_frame = info["clip_end_frame"]
            video_save_path = f"{self.action_data_dir}{info['clip_uid']}" +\
                              f"_{start_frame}_{end_frame}.mp4"

            if os.path.exists(video_save_path):
                continue

            video = cv2.VideoCapture(f"{self.data_dir}{info['clip_uid']}.mp4")
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
    
    def extract_action_clip_frame(self):
        """
        Saves all frames of 8s clips containing action, under 
        DATA_DIR/action_frames/{clip_uid}/*.png
        """
        frame_dict = {}
        for info in tqdm(self.flatten_json, desc='Finding frames to extract'):
            frame_dict.setdefault(info["clip_uid"], set())

            start_frame = info["clip_start_frame"]
            end_frame = info["clip_end_frame"]
            frame_dict[info["clip_uid"]] |=\
                {i for i in range(start_frame, end_frame + 1)}

        existing_frame_dirs = [d for d in os.listdir(self.action_frame_dir)
                               if os.path.isdir(f"{self.action_frame_dir}{d}")]
        for d in tqdm(existing_frame_dirs,
                      desc='Excluding existing frames to extract'):
            try:
                frame_dict[d] -=\
                    {int(f[:-4]) for f in os.listdir(
                        f"{self.action_frame_dir}{d}")}
            except KeyError:
                continue

        for clip_id, frame_nums in tqdm(frame_dict.items(),
                                        desc='Extracting frames'):
            if len(frame_nums) == 0:
                continue

            frame_save_dir = f"{self.action_frame_dir}{clip_id}/"
            os.makedirs(frame_save_dir, exist_ok=True)

            video = cv2.VideoCapture(f"{self.data_dir}{clip_id}.mp4")

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