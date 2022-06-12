import os
import argparse
import cv2
import mediapipe as mp


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=os.path.abspath,
                        default='./video_input')
    parser.add_argument('--video_save_dir', type=os.path.abspath,
                        default='./video_annotated')

    return parser.parse_args()


def main():
    args = option_parser()
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.video_save_dir, exist_ok=True)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(min_detection_confidence=0.0, min_tracking_confidence=0.1)

    for video_name in os.listdir(args.video_dir):
        video_path = os.path.join(args.video_dir, video_name)
        video_save_path = os.path.join(args.video_save_dir, video_name)

        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_size = (v_width, v_height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_save_path, fourcc, fps, v_size)

        for i in range(int(frame_count - (frame_count % fps))):
            ret, frame = video.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if not result.multi_hand_landmarks:
                writer.write(frame)
                continue

            for hand_landmarks in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            writer.write(frame)

        writer.release()
        video.release()


if __name__ == '__main__':
    main()
