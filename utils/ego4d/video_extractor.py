import os
import cv2
from tqdm import tqdm


class Extractor():
    def __init__(self, dataset_dir, task_name):
        self.task_name = task_name
        self.dataset_dir = dataset_dir
        self.clip_dir = os.path.join(dataset_dir, "ego4d/clips")

    def trim_around_action(self, flatten_json):
        """
        Trims video to 8s clips containing action, under
        "DATA_DIR/action_clips/{clip_id}_{start_frame}_{end_frame}.mp4".
        """

        action_data_dir = os.path.join(self.dataset_dir, "action_clips")
        os.makedirs(action_data_dir, exist_ok=True)

        for info in tqdm(flatten_json, desc='Trimming clip near action'):
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

            for counter in range(end_frame + 1):
                ret, frame = video.read()
                if ret == True and start_frame <= counter:
                    writer.write(frame)

            writer.release()
            video.release()

    def extract_frame_as_image(self, flatten_json, resize=True):
        """
        Saves necessary frames of clips containing action as png image, under 
        "DATA_DIR/frames/{clip_uid}/*.jpg".
        Also, saves first original frame as "sample.jpg".
        """

        frame_dir = os.path.join(self.dataset_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        frame_dict = {}
        for info in tqdm(flatten_json, desc='Finding frames to extract'):
            frame_dict.setdefault(info["clip_uid"], set())

            # different extraction for tasks
            if self.task_name == 'fho_hands':
                start_frame = info["clip_start_frame"]
                end_frame = info["clip_end_frame"]
                frame_dict[info["clip_uid"]] |=\
                    {i for i in range(start_frame, end_frame + 1)}

            elif self.task_name == 'fho_scod':
                frame_dict[info["clip_uid"]] |=\
                    {
                        info["pre_frame_num_clip"],
                        info["pnr_frame_num_clip"],
                        info["post_frame_num_clip"]
                }

            else:
                raise Exception

        existing_frame_dirs = [d for d in os.listdir(frame_dir)
                               if os.path.isdir(f"{frame_dir}{d}")]

        for d in tqdm(
            existing_frame_dirs,
            desc='Excluding existing frames to extract'
        ):
            try:
                frame_dict[d] -=\
                    {int(f[:-4]) for f in os.listdir(f"{frame_dir}{d}")
                     if int(f[:-4] != "sample")}
            except KeyError:
                continue

        for clip_id, frame_nums in tqdm(
            frame_dict.items(),
            desc='Extracting frames'
        ):
            if len(frame_nums) == 0:
                continue

            frame_save_dir = f"{frame_dir}{clip_id}/"
            os.makedirs(frame_save_dir, exist_ok=True)

            video_path = f"{self.clip_dir}{clip_id}.mp4"
            video = cv2.VideoCapture(video_path)

            for counter in range(1, int(video.get(cv2.CAP_PROP_FRAME_COUNT)) + 1):
                ret, frame = video.read()
                if ret == False:
                    break

                # saves original first frame as sample
                if counter == 1:
                    frame_save_path = f"{frame_save_dir}sample.jpg"
                    cv2.imwrite(frame_save_path, frame)

                if counter in frame_nums:
                    if resize:
                        frame = cv2.resize(frame, (224, 224))
                    frame_save_path = f"{frame_save_dir}{counter}.jpg"
                    cv2.imwrite(frame_save_path, frame)

            video.release()
