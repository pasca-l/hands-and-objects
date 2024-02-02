# Utilily files

## Description and Usage
### [ego4d.py](https://github.com/pasca-l/hands-and-objects/blob/main/utils/ego4d.py)
The following can be done with this file:
- Find missing videos required for a given task.
- Find missing frames given an annotation.
- Extracts necessary frames from videos.
- Copy necessary frames from a remote host.

To use this file:
1. Prepare requirements:
    - `REMOTE_HOST`, `REMOTE_PASS` in an `.env` file.
2. Create an instance of `Ego4DDatasetUtility` class defined in the file.
3. Run an executable script with the defined instance and called methods.
