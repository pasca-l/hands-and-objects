import json
from tqdm import tqdm


class JsonHandler():
    def __init__(self, ann_task_name):
        self.ann_task_name = ann_task_name
        self.unopenables = [
            '73de9018-dc67-48ca-a0a1-5697f9f100cd',
            'e964bb42-f596-4dca-96de-0940b52f0c75'
        ]
    
    def __call__(self, json_file):
        """
        Unpacks annotation json file to list of dicts,
        according to the task name.
        """
        if self.ann_task_name == 'fho_hands':
            return self._fho_hands_unpack(json_file)

    def _fho_hands_unpack(self, json_file, all_data=False):
        """
        The target annotation file is "fho_hands_PHASE.json".
        If all_data == True, unpacks possible pre and post frames, with hand 
        coordinates. (Some annotations might be missing; returns None).
        """
        flatten_json_list = []

        json_data = json.load(open(json_file, 'r'))
        for data in tqdm(json_data['clips'], desc='Preparing data'):
            if data['clip_uid'] in self.unopenables:
                continue

            for frame_data in data['frames']:
                # pnr frame must be included in any of the batch.
                try:
                    frame_data['pnr_frame']
                except KeyError:
                    continue

                json_dict = {
                    "clip_id": data['clip_id'],
                    "clip_uid": data['clip_uid'],
                    "video_uid": data['video_uid'],
                    "video_start_sec": frame_data['action_start_sec'],
                    "video_end_sec": frame_data['action_end_sec'],
                    "video_start_frame": frame_data['action_start_frame'],
                    "video_end_frame": frame_data['action_end_frame'],
                    "clip_start_sec": frame_data['action_clip_start_sec'],
                    "clip_end_sec": frame_data['action_clip_end_sec'],
                    "clip_start_frame": frame_data['action_clip_start_frame'],
                    "clip_end_frame": frame_data['action_clip_end_frame'],
                    "clip_pnr_frame": frame_data['pnr_frame']['clip_frame']
                }

                if all_data == True:
                    frame_alias_dict = {
                        'pre_45': "pre45",
                        'pre_30': "pre30",
                        'pre_15': "pre15",
                        'pre_frame': "pre",
                        'post_frame': "post",
                        'pnr_frame': "pnr"
                    }
                    for frame_type, alias in frame_alias_dict.items():
                        try:
                            temp_data = frame_data[frame_type]
                        except KeyError:
                            temp_dict = {
                                f"video_{alias}_frame": None,
                                f"clip_{alias}_frame": None,
                                f"{alias}_hands": None,
                            }
                        else:
                            temp_dict = {
                                f"video_{alias}_frame": temp_data['frame'],
                                f"clip_{alias}_frame": temp_data['clip_frame'],
                                f"{alias}_hands": temp_data['boxes'],
                            }
                        json_dict |= temp_dict

                flatten_json_list.append(json_dict)

        return flatten_json_list
