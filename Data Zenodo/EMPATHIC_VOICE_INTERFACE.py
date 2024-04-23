import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Directory to store the images
image_dir = 'IMAGES_Empathic_Voice_Interface'
os.makedirs(image_dir, exist_ok=True)

model_path = 'emotion_model.h5'
if not os.path.exists(model_path):
    raise Exception(f"The model file {model_path} does not exist. Please check the path.")

model = load_model(model_path)

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def predict_emotion(audio_path):
    features = preprocess_audio(audio_path)
    prediction = model.predict(np.array([features]))
    emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'calm', 'fearful']  # Updated with new emotions
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    return predicted_emotion

def respond_to_emotion(emotion):
    responses = {
        'angry': 'I am sensing some frustration. How can I assist you better?',
        'happy': 'Glad to hear you’re in high spirits! How can I help you today?',
        'sad': 'It sounds like a tough time. I’m here to help you.',
        'neutral': 'How can I assist you today?',
        'calm': 'You seem quite calm. Is there anything else you need?',
        'fearful': 'It seems like something is troubling you. How can I help to ease your concern?'
    }
    return responses.get(emotion, 'I am here to help you.')

def plot_mfcc(audio_path, emotion, iteration):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC - {emotion.capitalize()}')
    plt.savefig(os.path.join(image_dir, f'evi_mfcc_{emotion}_{iteration}.png'))
    plt.close()

def plot_emotion_distribution(audio_path, emotion, iteration):
    features = preprocess_audio(audio_path)
    prediction = model.predict(np.array([features]))[0]

    # Define a full list of possible emotions
    full_emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'calm', 'fearful']

    # Dynamically adjust the number of labels based on the prediction length
    used_emotion_labels = full_emotion_labels[:len(prediction)]

    plt.figure()
    plt.bar(used_emotion_labels, prediction)
    plt.title(f'EVI Emotion Prediction Distribution - {emotion.capitalize()}')
    plt.ylabel('Probability')
    plt.savefig(os.path.join(image_dir, f'evi_emotion_prediction_distribution_{emotion}_{iteration}.png'))
    plt.close()



def main(base_path):
    actors_dir = os.path.join(base_path, 'Audio_Song_Actors_01-24_split')
    for actor in sorted(os.listdir(actors_dir)):
        actor_path = os.path.join(actors_dir, actor)
        if os.path.isdir(actor_path):
            for emotion in sorted(os.listdir(actor_path)):
                emotion_folder_path = os.path.join(actor_path, emotion)
                if os.path.isdir(emotion_folder_path):
                    iteration = 1
                    for audio_file in sorted(os.listdir(emotion_folder_path)):
                        audio_path = os.path.join(emotion_folder_path, audio_file)
                        if audio_path.endswith('.wav'):
                            emotion = predict_emotion(audio_path)
                            response = respond_to_emotion(emotion)
                            print(f'Response for {audio_file}: {response}')
                            plot_mfcc(audio_path, emotion, iteration)
                            plot_emotion_distribution(audio_path, emotion, iteration)
                            iteration += 1

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


if __name__ == '__main__':

    base_path = 'Data Zenodo'
    main(base_path)
