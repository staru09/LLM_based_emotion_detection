import pandas as pd
import re

def clean_raw_output(text):
    cleaned = re.sub(r'[0-9.,%]|with a confidence score of|confident|confidence|score|prediction|expressing|this is|best guess', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned) 
    cleaned = cleaned.strip()
    cleaned = cleaned.replace('with a a', '')
    cleaned = cleaned.replace('with with', '')
    cleaned = cleaned.replace('nessnessness', 'ness')
    cleaned = cleaned.replace('FearFear', 'Fear')
    
    emotions = ['Anger', 'Happy', 'Sad', 'Sadness', 'Fear', 'Neutral', 'Disgust']
    for emotion in emotions:
        if emotion.lower() in cleaned.lower():
            return emotion      
    return cleaned

def count_emotions(df):
    emotion_counts = {}
    for output in df['cleaned_output']:
        if output in emotion_counts:
            emotion_counts[output] += 1
        else:
            emotion_counts[output] = 1    
    return emotion_counts

def main():
    df = pd.read_csv('emotion_detection_finetuned.csv')
    df['cleaned_output'] = df['raw_output'].apply(clean_raw_output)
    emotion_counts = count_emotions(df)
    print("\nEmotion Counts in raw_output:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")
    df.to_csv('emotion_detection_cleaned.csv', index=False)
if __name__ == "__main__":
    main()