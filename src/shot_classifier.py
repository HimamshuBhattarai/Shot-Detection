import mediapipe as mp

class ShotClassifier:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # False = video mode, faster
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_landmarks(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        results = self.pose.process(crop)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def classify(self, landmarks):
        if landmarks is None:
            return "unknown"

        lm = landmarks
        mp_pose = self.mp_pose

        r_wrist   = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        r_elbow   = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        l_wrist   = lm[mp_pose.PoseLandmark.LEFT_WRIST]

        # body center x
        body_center_x = (r_shoulder.x + l_shoulder.x) / 2

        # smash — wrist high above shoulder
        if r_wrist.y < r_shoulder.y - 0.25:
            return "smash"

        # forehand — right wrist on right side of body
        if r_wrist.x > body_center_x:
            return "forehand"

        # backhand — right wrist crosses to left side
        if r_wrist.x < body_center_x:
            return "backhand"

        return "unknown"