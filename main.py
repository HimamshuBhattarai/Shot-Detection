import cv2
import pandas as pd
import matplotlib.pyplot as plt
from src.detector import PlayerDetector
from src.shot_classifier import ShotClassifier

detector = PlayerDetector()
classifier = ShotClassifier()

results = []
frame_no = 0

cap = cv2.VideoCapture("data/input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_no += 1
    
    if frame_no % 150 == 0:
        print(f"Processing frame {frame_no}...")

    if frame_no % 3 != 0:
        continue

    players = detector.detect(frame)

    # sort by bbox area, take top 2 closest players
    players = sorted(players, key=lambda p: (p['bbox'][2]-p['bbox'][0]) * (p['bbox'][3]-p['bbox'][1]), reverse=True)
    players = players[:2]
    
    vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for p in players:
        x1, y1, x2, y2 = p['bbox']

        landmarks = classifier.get_landmarks(frame, p['bbox'])
        shot = classifier.classify(landmarks)

        timestamp = round(frame_no / fps, 2)

        results.append({
            'frame': frame_no,
            'timestamp': timestamp,
            'player_id': p['id'],
            'shot_type': shot
        })

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(vis, f"P{p['id']} | {shot}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    out.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
out.release()

# save results
df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
df.to_json('results.json', orient='records', indent=2)

print(df)
print(f"\nTotal shots logged: {len(df)}")
if len(df) > 0:
    print(df['shot_type'].value_counts())
else:
    print("No shots detected in the video.")

cap.release()