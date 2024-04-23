import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Folder path where you want to save the plots
output_folder = "Convolutional_Neural_Network_PyTorch_Images"

# Check if the folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, n_mels, max_length):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc = nn.Linear(128 * (n_mels // 8) * (max_length // 8), num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Dataset and DataLoader
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

# Function to process a single dataset
def process_dataset(dataset_path, emotion_dict, n_mels, n_fft, hop_length, max_files):
    # Load data
    spectrograms = []
    labels = []
    files_processed = 0

    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav') and files_processed < max_files:
                emotion = os.path.basename(subdir)  # Get the emotion from the folder name
                if emotion in emotion_dict:
                    filepath = os.path.join(subdir, file)
                    S_DB = extract_mel_spectrogram(filepath)
                    spectrograms.append(S_DB)
                    labels.append(emotion)
                    files_processed += 1
                else:
                    print(f"Folder {subdir} does not match expected emotions and will be skipped.")
            if files_processed >= max_files:
                break

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Pad and reshape spectrograms
    max_length = max(s.shape[1] for s in spectrograms)
    X = np.array([librosa.util.fix_length(s, size=max_length, axis=1) for s in spectrograms])
    X = X[..., np.newaxis]  # Add channel dimension for CNN input
    X = np.transpose(X, (0, 3, 1, 2))  # Rearrange dimensions to [batch, channel, height, width]

    return X, encoded_labels, max_length

# Feature Extraction Function
def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # Ensure the original sampling rate is used
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

# Training the model and tracking metrics
def train_model(num_epochs, model, train_loader, test_loader, criterion, optimizer, dataset_name):
    # Lists to track progress
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []
    dataset_name = dataset_path.split('/')[-1]  # This will take the last part of the path as the dataset name
    # Prepare filenames for saving training progress and confusion matrix
    training_progress_filename = f"cnn_training_progress_{dataset_name}.png"
    confusion_matrix_filename = f"confusion_matrix_{dataset_name}.png"

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (spectrograms, labels) in enumerate(train_loader):
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Validation loss
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                spectrograms, labels = data
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(test_loader)
        val_epoch_accuracy = 100 * correct / total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%')

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('CNN PyTorch - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('CNN PyTorch - Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, training_progress_filename))  # Save in the correct folder
    plt.show()
    plt.close()

    # Generate and save the confusion matrix
    all_preds = []
    all_true = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            spectrograms, labels = data
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_true.extend(labels.numpy())

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN PyTorch - Confusion Matrix for ' + dataset_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_folder, confusion_matrix_filename))
    plt.show()
    plt.close()


# Parameters
n_mels = 128
n_fft = 2048
hop_length = 512
n_classes = 8
max_files = 1000  # Update this if you want to process more or fewer files

# Emotion labels mapped directly from folder names
emotion_dict = {
    'angry': 'angry', 'calm': 'calm', 'disgust': 'disgust', 'fearful': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad', 'surprised': 'surprised'
}

# Datasets to process with updated paths
dataset_paths = [
    'Data Zenodo/Audio_Speech_Actors_01-24_split',
    'Data Zenodo/Audio_Song_Actors_01-24_split'
]


for dataset_path in dataset_paths:
    X, encoded_labels, max_length = process_dataset(dataset_path, emotion_dict, n_mels, n_fft, hop_length, max_files)

    # Split and create DataLoaders
    # Split the dataset into training and testing sets with a specific random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the model, loss, and optimizer for each dataset
    model = CNN(n_classes, n_mels, max_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model on this dataset
    print(f"Training on dataset: {dataset_path}")
    train_model(10, model, train_loader, test_loader, criterion, optimizer, f"cnn_training_{dataset_path}.png")

import random

# Global definition (if chosen)
label_encoder = LabelEncoder()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Assuming emotion_dict and other necessary imports and functions are defined as before

def process_dataset(dataset_path, emotion_dict, n_mels, n_fft, hop_length, max_files):
    spectrograms = []
    labels = []
    files_processed = 0

    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav') and files_processed < max_files:
                emotion = os.path.basename(subdir)
                if emotion in emotion_dict:
                    filepath = os.path.join(subdir, file)
                    S_DB = extract_mel_spectrogram(filepath)
                    spectrograms.append(S_DB)
                    labels.append(emotion)
                    files_processed += 1

    if not labels:
        raise ValueError("No labels processed. Ensure that your directories contain the correct files.")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    if not spectrograms:
        raise ValueError("No spectrograms processed. Check the dataset path and contents.")

    max_length = max(s.shape[1] for s in spectrograms)
    X = np.array([librosa.util.fix_length(s, size=max_length, axis=1) for s in spectrograms])
    X = X[..., np.newaxis]
    X = np.transpose(X, (0, 3, 1, 2))

    return X, encoded_labels, label_encoder

# Then use the returned label_encoder in your visualization function
X, encoded_labels, label_encoder = process_dataset(dataset_path, emotion_dict, n_mels, n_fft, hop_length, max_files)


def visualize_predictions(model, test_loader, label_encoder, num_images=5):
    model.eval()
    data_iter = iter(test_loader)
    plt.figure(figsize=(15, 5 * num_images))  # Adjust the size dynamically based on num_images

    actual_num_images = min(num_images, len(test_loader.dataset))
    if actual_num_images < num_images:
        print(f"Warning: Only {actual_num_images} images available for visualization.")

    try:
        for i in range(actual_num_images):
            spectrograms, labels = next(data_iter)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.numpy())
            true_labels = label_encoder.inverse_transform(labels.numpy())

            plt.subplot(actual_num_images, 1, i + 1)
            plt.imshow(spectrograms[0][0].cpu(), aspect='auto', origin='lower')
            plt.title(f'CNN PyTorch - Predicted: {predicted_labels[0]}, Actual: {true_labels[0]}')
            plt.colorbar(format='%+2.0f dB')

    except StopIteration:
        pass  # Handled by checking actual_num_images
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "CNN_PyTorch_visualize_predictions_spectrogram.png"))
    plt.show()



# After training and evaluation
visualize_predictions(model, test_loader, label_encoder)




def visualize_predictions(model, test_loader, label_encoder, num_images=5):
    model.eval()
    data_iter = iter(test_loader)
    plt.figure(figsize=(15, 5 * num_images))  # Adjust the size dynamically based on num_images

    actual_num_images = min(num_images, len(test_loader.dataset))
    if actual_num_images < num_images:
        print(f"Warning: Only {actual_num_images} images available for visualization.")

    try:
        for i in range(actual_num_images):
            spectrograms, labels = next(data_iter)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.numpy())
            true_labels = label_encoder.inverse_transform(labels.numpy())

            plt.subplot(actual_num_images, 1, i + 1)
            plt.imshow(spectrograms[0][0].cpu(), aspect='auto', origin='lower')
            plt.title(f'The David G. Cooper Graph. CNN PyTorch - Predicted: {predicted_labels[0]}, Actual: {true_labels[0]}')
            plt.colorbar(format='%+2.0f dB')

    except StopIteration:
        pass  # Handled by checking actual_num_images
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "CNN_PyTorch_visualize_predictions_spectrogram.png"))
    plt.show()

# After training and evaluation
visualize_predictions(model, test_loader, label_encoder)




import torch.nn.functional as F

# Define a simple regression model
class EmotionIntensityModel(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(EmotionIntensityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dictionary to store models for each emotion
emotion_models = {}
intensity_criterion = nn.MSELoss()  # Using Mean Squared Error Loss for regression

for emotion in emotion_dict.keys():
    # Initialize model for each emotion
    model_size = 128 * (n_mels // 8) * (max_length // 8)
    emotion_models[emotion] = EmotionIntensityModel(model_size)
    optimizer = optim.Adam(emotion_models[emotion].parameters(), lr=0.001)

    # Filter data for the current emotion
    emotion_indices = [i for i, label in enumerate(encoded_labels) if label == label_encoder.transform([emotion])[0]]
    emotion_X = X[emotion_indices]
    emotion_y = np.random.rand(len(emotion_indices))  # Placeholder for actual intensity values

    # Create datasets
    emotion_dataset = AudioDataset(emotion_X, emotion_y)
    emotion_loader = DataLoader(emotion_dataset, batch_size=16, shuffle=True)

    # Train the regression model for this emotion
    for epoch in range(5):  # Assuming a smaller number of epochs for simplicity
        for i, (inputs, targets) in enumerate(emotion_loader):
            outputs = emotion_models[emotion](inputs)
            loss = intensity_criterion(outputs.squeeze(), targets.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Training {emotion}: Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Example of how you might use the models to predict emotional intensity
# Assuming you have a sample input for prediction
# sample_input = some_preprocessed_spectrogram
# predicted_intensity = emotion_models['happy'](sample_input)
# print("Predicted Intensity for Happy:", predicted_intensity.item())


import torch.nn.functional as F

# Define a regression model for predicting emotional intensity
class IntensityModel(nn.Module):
    def __init__(self, num_features):
        super(IntensityModel, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # No activation, as we are predicting a scalar value
        return x

# Assuming you have intensity data available as part of your dataset
# Example data generation (replace with actual data loading)
intensity_labels = np.random.uniform(0, 1, len(labels))  # Random intensities for demonstration

# Update dataset class to include intensity
class AudioIntensityDataset(Dataset):
    def __init__(self, X, y, y_intensity):
        self.X = X
        self.y = y
        self.y_intensity = y_intensity

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_intensity[idx]

# Modify existing data preprocessing to include intensity data
def prepare_data(X, labels, intensity_labels):
    X_train, X_test, y_train, y_test, intensity_train, intensity_test = train_test_split(
        X, labels, intensity_labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, intensity_train, intensity_test

# Example: prepare your data
X_train, X_test, y_train, y_test, intensity_train, intensity_test = prepare_data(X, encoded_labels, intensity_labels)

# Create datasets for training and testing
train_dataset = AudioIntensityDataset(X_train, y_train, intensity_train)
test_dataset = AudioIntensityDataset(X_test, y_test, intensity_test)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize and train a model for each emotional category
emotion_intensity_models = {emotion: IntensityModel(max_length * n_mels // 64) for emotion in emotion_dict.keys()}
optimizers = {emotion: optim.Adam(emotion_intensity_models[emotion].parameters(), lr=0.001) for emotion in emotion_dict.keys()}

def train_intensity_models(n_epochs):
    for emotion, model in emotion_intensity_models.items():
        for epoch in range(n_epochs):
            for inputs, labels, intensities in train_loader:
                model.train()
                optimizer = optimizers[emotion]
                optimizer.zero_grad()

                # Train only on data of the current emotion
                indices = labels == label_encoder.transform([emotion])[0]
                if sum(indices) == 0:
                    continue

                predictions = model(inputs[indices].float())
                loss = F.mse_loss(predictions, intensities[indices].unsqueeze(1).float())

                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}, {emotion} Loss: {loss.item():.4f}')

# Train models
train_intensity_models(5)

# Example function to demonstrate a prediction
def predict_intensity(model, sample_input):
    model.eval()
    with torch.no_grad():
        prediction = model(sample_input)
    return prediction.item()

# Assuming you have a preprocessed sample input
# predicted_intensity = predict_intensity(emotion_intensity_models['happy'], some_preprocessed_input)
# print(f"Predicted intensity for happy emotion: {predicted_intensity}")
