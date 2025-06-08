import os
import csv
import random

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

BASE_DIR = 'main_data_set' 
OUTPUT_DIR = '/working/'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'ravdess.csv')

metadata = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith('.wav'):
            try:
                parts = file.split('.')[0].split('-')
                emotion_id = parts[2]
                emotion_label = emotion_map[emotion_id]
                file_path = os.path.join(root, file)
                metadata.append([file_path, emotion_label])
            except Exception as e:
                print(f"Error processing {file}: {e}")

with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Path', 'Emotions'])
    writer.writerows(metadata)

print(f"Metadata saved to {OUTPUT_CSV} with {len(metadata)} entries.")

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

for split_name, data in split_data.items():
    output_path = os.path.join(OUTPUT_DIR, f"{'dev' if split_name == 'validation' else split_name}_ravdess_data.csv")
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "Emotion"])
        writer.writerows(data)

# Step 5: Print summary
print("\nCSV splits created:")
for k in split_data:
    print(f"{k}: {len(split_data[k])} samples")
