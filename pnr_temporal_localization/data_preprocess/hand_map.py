import numpy as np
import mediapipe as mp


class AddHandMapTransform():
    def __init__(self):
        self.image_mode = True
        self.hand_num = 2
        self.detection_conf = 0.7

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=self.image_mode,
            max_num_hands=self.hand_num,
            min_detection_confidence=self.detection_conf
        )

    def __call__(self, frames):
        frame_num, h, w, c = frames.shape
        output = np.empty((frame_num, h, w, c + 1))

        for i in range(frame_num):
            frame = frames[i]
            saliency_map = self.hand_saliency_map(frame)
            output[i,:,:,:] = np.concatenate([frame, saliency_map], -1)

        return output

    def _twoD_gaussian(self, x, y, coord, sigma, amp):
        function = amp * np.exp(-0.5 * (((x - coord[0]) / sigma) ** 2 +
                                ((y - coord[1]) / sigma) ** 2))
        return function

    def _min_max(self, x):
        min = x.min(keepdims=True)
        max = x.max(keepdims=True)
        result = (x - min) / (max - min)
        return result

    def hand_saliency_map(self, frame):
        h, w, _ = frame.shape
        coords = []
        map = np.zeros_like(frame[:,:,0], dtype='f8')
        x = np.arange(0, map.shape[1], 1)
        y = np.arange(0, map.shape[0], 1)
        X, Y = np.meshgrid(x, y)

        image = frame.astype('uint8')
        result = self.hands.process(image)

        if result.multi_hand_landmarks is None:
            return np.expand_dims(map, -1)

        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                coords.append((landmark.x * w, landmark.y * h))

        for coord in coords:
            map += self._twoD_gaussian(X, Y, coord, 20, 1)
        map = self._min_max(map) * 255

        return np.expand_dims(map, -1)