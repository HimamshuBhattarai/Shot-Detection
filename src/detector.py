import torch
from ultralytics import YOLO


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlayerDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.25):
        self.model = YOLO(model_path)
        self.model.to(device=device)
        self.conf = conf

    def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,      # keeps player IDs consistent across frames
            conf=self.conf,
            classes=[0],       # class 0 = person only
            verbose=False
        )[0]

        players = []
        for box in results.boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            player_id = int(box.id)
            conf = float(box.conf)
            players.append({
                'id': player_id,
                'bbox': (x1, y1, x2, y2),
                'conf': conf
            })

        return players
    
    