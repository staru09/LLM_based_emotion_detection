import csv
import json
import os

def convert_emotion_csv_to_json(input_csv, output_json, system_prompt):
    output_data = []

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row['file_path']
            emotion = row['Emotion']

            entry = {
                "messages": [
                    {
                        "content": "<video>What is the emotion expressed in this clip?",
                        "role": "user"
                    },
                    {
                        "content": f"The emotion expressed in this audio is {emotion}.",
                        "role": "assistant"
                    }
                ],
                "videos": [file_path],
                "system": system_prompt
            }
            output_data.append(entry)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(output_data)} samples from {input_csv} to {output_json}")
    return output_data

if __name__ == "__main__":
    base_dir = "./working"
    splits = {
        "train": "train_data.csv",
        "dev": "dev_data.csv",
        "test": "test_data.csv"
    }

    system_prompt = "You are an expert assistant trained to detect human emotions from the video clips accurately and explain your predictions when asked."

    for split, csv_file in splits.items():
        input_csv = os.path.join(base_dir, csv_file)
        output_json = os.path.join(base_dir, csv_file.replace('.csv', '.json'))
        convert_emotion_csv_to_json(input_csv, output_json, system_prompt)
