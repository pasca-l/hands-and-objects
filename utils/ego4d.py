import os
import cv2
import glob
from tqdm import tqdm
from dotenv import load_dotenv
import pexpect


class Ego4DDatasetUtility:
    def __init__(
        self,
        info,
        dataset_dir,
        mode="all", # ["center", "range", "all"]
    ):
        data_dir = os.path.join(dataset_dir, "ego4d/v2")
        self.video_dir = os.path.join(data_dir, "full_scale")
        self.frame_dir = os.path.join(data_dir, "frames")
        self.subclip_dir = os.path.join(data_dir, "subclips")
        self.mode = mode

        load_dotenv()
        self.rmt_host = os.environ["REMOTE_HOST"]
        self.rmt_path = "/home/ubuntu/data/ego4d/v2/frames"
        self.rmt_pass = os.environ["REMOTE_PASS"]

        if self.mode == "center":
            self.iterator = info.select(
                "video_uid", "center_frame",
            ).iter_rows()
        elif self.mode == "range":
            self.iterator = info.select(
                "video_uid", "parent_start_frame", "parent_end_frame",
            ).iter_rows()
        elif self.mode == "all":
            self.iterator = info.select(
                "video_uid", "parent_frame_num",
            ).unique(
                maintain_order=True,
            ).iter_rows()

    def find_missing_videos(self):
        video_paths = glob.glob(f"{self.video_dir}/*")
        videos = [os.path.splitext(os.path.basename(i))[0] for i in video_paths]

        missing = set()
        for video_id, *_ in self.iterator:
            if video_id not in videos:
                missing.add(video_id)

        return missing

    def find_missing_frames(self):
        frame_dict = {}
        for video_uid, *frames in tqdm(
            self.iterator,
            desc="Finding frames to extract",
        ):
            frame_dict.setdefault(video_uid, set())

            if self.mode == "center":
                [center] = frames
                frame_dict[video_uid] |= {center}
            elif self.mode == "range":
                start_frame, end_frame = frames
                frame_dict[video_uid] |= \
                    {i for i in range(start_frame, end_frame + 1)}
            elif self.mode == "all":
                [frame_num] = frames
                frame_dict[video_uid] |= \
                    {i for i in range(1, frame_num + 1)}

        existing_frame_dirs = [
            d for d in os.listdir(self.frame_dir)
            if os.path.isdir(os.path.join(self.frame_dir, d))
        ]
        for d in tqdm(
            existing_frame_dirs,
            desc="Excluding existing frames to extract",
        ):
            try:
                frame_dict[d] -= {
                    int(f[:-4]) for f in os.listdir(
                        os.path.join(self.frame_dir, d)
                    )
                    if int(f[:-4] != "sample")
                }
            except KeyError:
                continue

        return frame_dict

    def extract_frames(self, resize=True):
        """
        Saves necessary annotated frames of a video as jpg image, under 
        "DATA_DIR/frames/{video_uid}/*.jpg".
        Also, saves first original frame as "sample.jpg".
        """
        os.makedirs(self.frame_dir, exist_ok=True)

        frame_dict = self.find_missing_frames()

        with tqdm(frame_dict.items()) as pbar:
            for video_uid, frame_nums in pbar:
                pbar.set_description(f"Extracting frames from {video_uid}")

                if len(frame_nums) == 0:
                    continue

                save_dir = os.path.join(self.frame_dir, video_uid)
                os.makedirs(save_dir, exist_ok=True)

                video_path = os.path.join(self.video_dir, f"{video_uid}.mp4")
                if not os.path.exists(video_path):
                    raise Exception(f"Video does not exist at: {video_path}")
                video = cv2.VideoCapture(video_path)

                for i, f in enumerate(sorted(frame_nums)):
                    pbar.set_postfix({"frame": f})

                    video.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ret, frame = video.read()
                    if ret == False:
                        break

                    if i == 0:
                        cv2.imwrite(os.path.join(save_dir, "sample.jpg"), frame)

                    if resize:
                        frame = cv2.resize(frame, (224, 224))
                    cv2.imwrite(os.path.join(save_dir, f"{f}.jpg"), frame)

                video.release()

    def copy_from_remote(self):
        """
        Copies necessary frames from remote storage, under "DATA_DIR/frames/{video_uid}/*.jpg".
        """
        os.makedirs(self.frame_dir, exist_ok=True)

        frame_dict = self.find_missing_frames()

        with tqdm(frame_dict.items()) as pbar:
            for video_uid, frame_nums in pbar:
                pbar.set_description(f"Extracting frames from {video_uid}")

                if len(frame_nums) == 0:
                    continue

                save_dir = os.path.join(self.frame_dir, video_uid)
                os.makedirs(save_dir, exist_ok=True)

                for f in sorted(frame_nums):
                    pbar.set_postfix({"frame": f})

                    scp = pexpect.spawn(
                        f"scp \
                        {self.rmt_host}:{self.rmt_path}/{video_uid}/{f}.jpg \
                        {self.frame_dir}/{video_uid}/{f}.jpg"
                    )
                    scp.expect('.ssword:*')
                    scp.sendline(self.rmt_pass)
                    scp.interact()
