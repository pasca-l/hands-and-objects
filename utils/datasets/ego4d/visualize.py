import os
import sys
import git
import cv2
import lightning as L

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(f"{git_root}/keypoint_estimation/")

from datamodule import KeypointEstDataModule


class Visualizer:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        L.seed_everything(42, workers=True)

        datamodule = KeypointEstDataModule(
            dataset_dir=dataset_dir,
            dataset_mode="ego4d",
            batch_size=1,
            transform_mode="display",
            selection="segsec",
            sample_num=16,
            with_info=True,
            neg_ratio=None,
        )
        datamodule.setup(stage="test")

        self.dataset = datamodule.test_data
        self.df = self.dataset.ann_df.with_row_count()

    def create_video_with_pnr_annotation(
        self, idx, save_as, from_video=False, fps_out=10,
    ):
        _, _, _, [info] = self.dataset[idx]

        # prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_as, fourcc, fps_out, (224,224))

        start, end = info["segment_start_frame"], info["segment_end_frame"]
        for n in range(start, end + 1):

            # if using video as source
            if from_video:
                video_path = os.path.join(
                    self.dataset_dir,
                    f"ego4d/v2/full_scale/{info['video_uid']}.mp4",
                )
                video = cv2.VideoCapture(video_path)

                video.set(cv2.CAP_PROP_POS_FRAMES, n)
                ret, frame = video.read()
                if ret == False:
                    raise Exception(f"Cannot read frame {n} from: {video_path}")
                frame = cv2.resize(frame, (224,224))

            # if using cut out frames as source
            else:
                frame_path = os.path.join(
                    self.dataset_dir,
                    f"ego4d/v2/frames/{info['video_uid']}/{n}.jpg",
                )
                if not os.path.exists(frame_path):
                    raise Exception(f"No path at: {frame_path}")
                frame = cv2.imread(frame_path)

            # add red boundary for pnr frames
            if n in info["parent_pnr_frame"]:
                w, h, _ = frame.shape
                frame = cv2.rectangle(frame, (0,0), (w,h), (255,0,0), 5)

            writer.write(frame)

        if from_video:
            video.release()
        writer.release()


if __name__ == "__main__":
    visualizer = Visualizer(
        os.path.join(
            os.path.expanduser("~"), "Documents/datasets"
        )
    )

    visualizer.create_video_with_pnr_annotation(0, "./output.mp4")
