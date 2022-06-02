import os
import argparse
import cv2
import mediapipe.solutions as mpsoln


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=os.path.abspath,
                        default='./video_input')
    parser.add_argument('--video_save_dir', type=os.path.abspath,
                        default='./video_annotated')

    return parser.parse_args()


def main():
    args = option_parser()

    hands = mpsoln.hands.Hands()

    for video_name in os.listdir(args.video_dir):
        video_path = os.path.join(args.video_dir, video_name)
        print(video_path)
        break
        video_save_path = os.path.join(args.video_save_dir, f"{video_name}.")

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_size = (v_width, v_height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_save_path, fourcc, fps, v_size)

        while True:
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)

            print(result.multi_hand_landmarks)

            h, w, _ = frame.shape
            annotated_frame = frame.copy()

            if not result.multi_hand_landmarks:
                writer.write(annotated_frame)
                continue

            for hand_landmarks in result.multi_hand_landmarks:
                print(hand_landmarks)

                mpsoln.drawing_utils.draw_landmarks(
                    annotated_frame, hand_landmarks,
                    mpsoln.hands.HAND_CONNECTIONS,
                    mpsoln.drawing_styles.get_default_hand_landmarks_style(),
                    mpsoln.drawing_styles.get_default_hand_connections_style())

            writer.write(annotated_frame)

        writer.release()
        video.release()


if __name__ == '__main__':
    main()
