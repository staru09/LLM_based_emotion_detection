import os
import csv
import random

# Emotion code to full label
emotion_map = {
    'ANG': 'anger',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# Step 1: Define paths
BASE_DIR = '/kaggle/input/crema-d/AudioWAV'  # Change if needed
OUTPUT_DIR = '/kaggle/working/'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'cremad.csv')

# Step 2: Parse filenames and collect metadata
metadata = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith('.wav'):
            try:
                parts = file.split('_')
                emotion_code = parts[2]
                emotion_label = emotion_map.get(emotion_code)
                if emotion_label:
                    file_path = os.path.join(root, file)
                    metadata.append([file_path, emotion_label])
            except Exception as e:
                print(f"Failed to parse {file}: {e}")

# Step 3: Save metadata
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Path', 'Emotions'])
    writer.writerows(metadata)

print(f"Metadata CSV created: {OUTPUT_CSV} ({len(metadata)} samples)")

# Step 4: Split the dataset
random.seed(42)
SPLITS = {'train': 0.8, 'validation': 0.1, 'test': 0.1}

all_data = []
with open(OUTPUT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        path = row['Path']
        emotion = row['Emotions'].lower()
        all_data.append((path, emotion))

random.shuffle(all_data)

total = len(all_data)
train_end = int(total * SPLITS['train'])
val_end = train_end + int(total * SPLITS['validation'])

split_data = {
    "train": all_data[:train_end],
    "validation": all_data[train_end:val_end],
    "test": all_data[val_end:]
}

# Step 5: Save split CSVs
for split_name, data in split_data.items():
    output_path = os.path.join(OUTPUT_DIR, f"{'dev' if split_name == 'validation' else split_name}_cremad_data.csv")
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "Emotion"])
        writer.writerows(data)

# Step 6: Summary
print("\nCSV splits created:")
for split, samples in split_data.items():
    print(f"{split}: {len(samples)} samples")
