import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in DataFrame:", df.columns.tolist())
    return df

def compute_metrics(df):
    y_true = df['category']
    y_pred = df['cleaned_output']
    
    print("\nOverall Accuracy:", accuracy_score(y_true, y_pred))
    
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    print("\nPredictions per category:")
    confusion = pd.crosstab(df['category'], df['cleaned_output'])
    print(confusion)
    
    print("\nAccuracy per emotion:")
    for emotion in df['category'].unique():
        mask = df['category'] == emotion
        acc = accuracy_score(df[mask]['category'], df[mask]['cleaned_output'])
        total = mask.sum()
        correct = (df[mask]['category'] == df[mask]['cleaned_output']).sum()
        print(f"{emotion}: {acc:.3f} ({correct}/{total} correct)")

if __name__ == "__main__":
    file_path = "emotion_detection_cleaned.csv"
    df = load_and_clean_data(file_path)
    compute_metrics(df)

"""
Qwen2.5vl (Full_dataset)
Accuracy: 0.4001482030381623
Classification Report:
               precision    recall  f1-score   support

       Anger       0.26      0.15      0.19       450
     Disgust       0.57      0.24      0.33       450
        Fear       0.83      0.01      0.02       450
       Happy       0.76      0.95      0.84       450
     Neutral       0.27      0.98      0.42       450
         Sad       0.74      0.08      0.14       449
    Surprise       0.00      0.00      0.00         0

    accuracy                           0.40      2699
   macro avg       0.49      0.34      0.28      2699
weighted avg       0.57      0.40      0.33      2699

Samvedna (qwen2.5vl)
Accuracy: 0.4049382716049383
Classification Report:
               precision    recall  f1-score   support

       Anger       0.27      0.15      0.19      1350
     Disgust       0.54      0.24      0.33      1350
        Fear       0.80      0.01      0.02      1350
       Happy       0.79      0.96      0.86      1350
     Neutral       0.27      0.99      0.43      1350
         Sad       0.74      0.09      0.16      1350

    accuracy                           0.40      8100
   macro avg       0.49      0.35      0.28      8100
weighted avg       0.57      0.40      0.33      8100


Samvedna (qwen2vl finetuned)
Overall Accuracy: 0.6160493827160494

Classification Report:
               precision    recall  f1-score   support

       Anger       0.39      0.73      0.51       135
     Disgust       1.00      0.01      0.01       135
        Fear       0.81      0.48      0.60       135
       Happy       0.92      0.97      0.95       135
     Neutral       0.70      0.82      0.76       135
         Sad       0.52      0.69      0.59       135

    accuracy                           0.62       810
   macro avg       0.72      0.62      0.57       810
weighted avg       0.72      0.62      0.57       810


Predictions per category:
cleaned_output  Anger  Disgust  Fear  Happy  Neutral  Sad
category
Anger              98        0    14      1       14    8
Disgust            86        1     0      4        5   39
Fear               23        0    65      1       16   30
Happy               3        0     0    131        0    1
Neutral            13        0     1      2      111    8
Sad                26        0     0      3       13   93

"""