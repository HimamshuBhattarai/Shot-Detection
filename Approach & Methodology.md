## Approach & Methodology

To gain insight into the data, I spent some time checking it before I wrote any code. I imported the video file and checked the details: It is 25 frames per second, 1920x1080 and contains 8125 frames (about 5.4 minutes of video). I then showed them random frames on the screen to help them understand the camera angle, player positions and the layout of the court. I also tried YOLO filtering a single frame to see what it was able to find, and ran MediaPipe pose estimation on a cropped player before developing the pipeline to ensure the landmarks are receiving.

The final pipeline works like this:
1. For every 3rd frame, YOLOv8n detects and tracks all players using ByteTrack
2. Players are sorted by bounding box area — the 2 largest (closest to camera) are selected for analysis
3. MediaPipe Pose extracts joint landmarks from each player crop
4. A rule-based classifier uses wrist and shoulder positions to determine shot type
5. Results are logged with frame number, timestamp, player ID and shot type

---

## Challenges

**Ball Detection** was the first big challenge. I did 2 approaches: First using YOLOv8 with pretrained COCO weights, and second using HSV color segmentation to catch the lime green ball. The sports ball class of YOLO did not work on the small speedy padel ball and missed it in all 11 frames sampled. Many false positives in HSV because the colors of the players' clothes and court matched each other. Eventually, the prototype was abandoned in favor of ball detection

**MediaPipe Version Compatibility** An unanticipated obstacle was MediaPipe Version Compatibility. The deleted API `solutions` occurs in the newest version of mediapipe (0.10.35), leading to an instant AttributeError. This was fixed by downgrading to 0.10.9.

**Player ID Instability** was another issue. It took 114 player IDs for a video, even though there were just 4 players. This is because as a player exits the frame and re-enters, the tracker issues a new ID to the player, rather than reconnecting the "old" ID to the same player.

**Unknown Classifications (23%)** — MediaPipe is not confident enough in spots where it fails to accurately identify pose landmarks, specifically far-court players that have a small bounding box.MediaPipe is not confident enough in spots where it fails to accurately identify pose landmarks, specifically far-court players that have a small bounding box. The following days, these frames were logged as “unknown”.

---

## Future Improvements

- **Ball Detection** — Fine-tune YOLOv8 on a padel-specific labeled dataset to properly detect the small fast-moving ball
- **Player Re-identification** — Use appearance-based ReID models to maintain consistent player IDs across the full match
- **Smash Accuracy** — Current smash detection triggers on any raised wrist. Adding wrist velocity would reduce false positives
- **Learned Classifier** — Replace rule-based logic with an LSTM trained on pose sequences to capture the full motion of a shot rather than a single frame snapshot
- **Frame Skip** — Processing every 3rd frame may miss quick shots. A motion-detection based approach could selectively process frames where significant movement occurs