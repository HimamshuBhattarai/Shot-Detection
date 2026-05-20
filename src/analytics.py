import pandas as pd
import json

df = pd.read_csv('results.csv')

# 1. overall shot counts
print(df['shot_type'].value_counts())

# 2. shot counts per player
print(df.groupby(['player_id', 'shot_type']).size())

# 3. shots per minute per player
shots_per_min = (
    df.groupby('player_id')
      .size() / (df['timestamp'].max() / 60)
)

print(shots_per_min)

# 4. save analytics
analytics = {
    "overall_shot_counts": df['shot_type'].value_counts().to_dict(),
    
    "shots_per_player": {
        f"player_{pid}_{shot}": count
        for (pid, shot), count in df.groupby(['player_id', 'shot_type']).size().items()
    },

    "shots_per_minute_per_player": shots_per_min.to_dict()
}

with open("analytics.json", "w") as f:
    json.dump(analytics, f, indent=4)