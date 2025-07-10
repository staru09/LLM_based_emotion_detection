import os
import csv
import random

# Emotion letter to English label
emotion_map = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'fear',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral'
}

# Base directory containing EMO-DB `.wav` files
BASE_DIR = 'LLM_based_emotion_detection/emodb'  
OUTPUT_DIR = 'LLM_based_emotion_detection/data_processing/'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'emodb.csv')

# Step 1: Parse files and collect metadata
metadata = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith('.wav') and len(file) >= 6:
            try:
                emotion_letter = file[5].upper()  
                emotion_label = emotion_map.get(emotion_letter)
                if emotion_label:  # skip unknown labels
                    file_path = os.path.join(root, file)
                    metadata.append([file_path, emotion_label])
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Step 2: Write full metadata to CSV
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Path', 'Emotions'])
    writer.writerows(metadata)

print(f"Metadata CSV created: {OUTPUT_CSV} ({len(metadata)} samples)")

# Step 3: Dataset splitting
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

# Step 4: Write splits to separate CSVs
for split_name, data in split_data.items():
    output_path = os.path.join(OUTPUT_DIR, f"{'dev' if split_name == 'validation' else split_name}_emodb_data.csv")
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "Emotion"])
        writer.writerows(data)

# Step 5: Summary
print("\nCSV splits created:")
for split, samples in split_data.items():
    print(f"{split}: {len(samples)} samples")
