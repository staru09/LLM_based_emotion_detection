import os
import csv
import random
from pathlib import Path

random.seed(42)

DATASET_ROOT = 'LLM_based_emotion_detection/indian_emotion_dataset'
SPLITS = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
OUTPUT_DIR = 'LLM_based_emotion_detection/data_processing/'

all_data = []
for emotion in os.listdir(DATASET_ROOT):
    emotion_path = os.path.join(DATASET_ROOT, emotion)
    if not os.path.isdir(emotion_path):
        continue
    for wav_file in os.listdir(emotion_path):
        if wav_file.endswith('.wav'):
            full_path = os.path.join(emotion_path, wav_file)
            rel_path = os.path.relpath(full_path, '.')  
            all_data.append((rel_path, emotion.lower())) 

random.shuffle(all_data)

total = len(all_data)
train_end = int(total * SPLITS['train'])
val_end = train_end + int(total * SPLITS['validation'])

split_data = {
    "train": all_data[:train_end],
    "validation": all_data[train_end:val_end],
    "test": all_data[val_end:]
}

for split_name, data in split_data.items():
    filename = f"{'dev' if split_name == 'validation' else split_name}_pumave.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "Emotion"])
        writer.writerows(data)

print("CSV splits created:")
for k in split_data:
    print(f"{k}: {len(split_data[k])} samples")
