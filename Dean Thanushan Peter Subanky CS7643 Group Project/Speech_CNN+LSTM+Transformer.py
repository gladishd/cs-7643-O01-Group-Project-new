import os
import librosa
import numpy as np
import torch
torch.cuda.device_count()
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import json

# Define the RAVDESS dataset class
class RAVDESSDataset(Dataset):
    def __init__(self, audio_dirs, labels):
        self.audio_dirs = audio_dirs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            batch_size = idx.stop - idx.start
            batch_log_mel_specs = []
            batch_emotions = []

            for i in range(idx.start, idx.stop):
                audio_path = os.path.join(self.audio_dirs[self.labels[i]['vocal_channel']],
                                          f"Actor_{self.labels[i]['actor']:02d}",
                                          self.labels[i]['path'])
                emotion = self.labels[i]['emotion']

                # Load audio file and preprocess
                y, sr = librosa.load(audio_path, sr=16000)
                y = librosa.effects.trim(y, top_db=20)[0]
                y = librosa.util.fix_length(y, size=sr * 7)

                # Extract log Mel-filterbank features
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=1024)
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                batch_log_mel_specs.append(log_mel_spec)
                batch_emotions.append(emotion)

            batch_log_mel_specs = np.stack(batch_log_mel_specs)
            batch_emotions = np.array(batch_emotions)

            return batch_log_mel_specs, batch_emotions
        else:
            audio_path = os.path.join(self.audio_dirs[self.labels[idx]['vocal_channel']],
                                      f"Actor_{self.labels[idx]['actor']:02d}",
                                      self.labels[idx]['path'])
            emotion = self.labels[idx]['emotion']

            # Load audio file and preprocess
            y, sr = librosa.load(audio_path, sr=16000)
            y = librosa.effects.trim(y, top_db=20)[0]
            y = librosa.util.fix_length(y, size=sr * 7)

            # Extract log Mel-filterbank features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=1024)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            return log_mel_spec, emotion

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_emotions, dropout_rates=[0.3, 0.3, 0.3], d_model=256, nhead=8, num_layers=8, dim_feedforward=512):
        super(EmotionRecognitionModel, self).__init__()

        # Conv + BN + ReLU + Dropout Blocks with residual connections
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(dropout_rates[0])

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(dropout_rates[1])

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(dropout_rates[2])

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(dropout_rates[2])

        # Average Pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Residual connections
        self.residual_conv1 = nn.Conv2d(1, 64, kernel_size=1, stride=1)
        self.residual_bn1 = nn.BatchNorm2d(64)
        self.residual_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.residual_conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.residual_bn2 = nn.BatchNorm2d(128)
        self.residual_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.residual_conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.residual_bn3 = nn.BatchNorm2d(256)
        self.residual_pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Calculate the output size after the convolutional and pooling layers
        conv_output_size = 256 * (128 // (2 ** 4)) * (219 // (2 ** 4))
        self.bilstm_input_size = conv_output_size

        # BiLSTM Layer with attention
        self.bilstm = nn.LSTM(input_size=self.bilstm_input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Stacked Transformer Layers with positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout_rates[0], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        # Fully Connected Layers with dropout
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout5 = nn.Dropout(dropout_rates[0])
        self.fc2 = nn.Linear(128, num_emotions)

    def forward(self, x):
        # Block 1
        residual = self.residual_conv1(x)
        residual = self.residual_bn1(residual)
        residual = self.residual_pool1(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.avg_pool(x)
        x = x + residual

        # Block 2
        residual = self.residual_conv2(x)
        residual = self.residual_bn2(residual)
        residual = self.residual_pool2(residual)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.avg_pool(x)
        x = x + residual

        # Block 3
        residual = self.residual_conv3(x)
        residual = self.residual_bn3(residual)
        residual = self.residual_pool3(residual)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.avg_pool(x)
        x = x + residual

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.avg_pool(x)

        # Reshape for BiLSTM - ensure this matches the expected input size
        x = x.view(x.size(0), -1)  # Flattening
        x = x.view(x.size(0), 1, -1)  # Reshaping for BiLSTM

        # BiLSTM with attention
        self.bilstm.flatten_parameters()
        x, _ = self.bilstm(x)
        x, _ = self.attention(x, x, x)

        # Stacked Transformer Layers with positional encoding
        x = x + self.positional_encoding[:, :x.size(1)]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average pooling across sequence length

        # Fully Connected Layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)

        return x

# Set the paths and parameters
audio_dirs = {'speech': './data/RAVDESS/speech',
              'song': './data/RAVDESS/song'}
num_emotions = 8  # Number of emotion categories in RAVDESS
batch_size = 128
num_epochs = 100
num_folds = 5

# Create the labels
labels = []
for vocal_channel in ['speech', 'song']:
    for actor in range(1, 25):
        actor_dir = os.path.join(audio_dirs[vocal_channel], f"Actor_{actor:02d}")
        for filename in os.listdir(actor_dir):
            if filename.endswith('.wav'):
                modality, vocal_channel_id, emotion, intensity, statement, repetition, actor_id = filename.split('-')
                if modality == '03' and emotion in ['01', '02', '03', '04', '05', '06', '07', '08']:
                    labels.append({'path': filename, 'vocal_channel': vocal_channel,
                                   'emotion': int(emotion) - 1, 'actor': actor})

# Create the dataset
dataset = RAVDESSDataset(audio_dirs, labels)

# Initialize lists to store accuracies for each fold
train_accuracies = []
test_accuracies = []

# Create the cross-fold splits
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize a list to store the model weights from each fold
model_weights = []

# Training and evaluation loop
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Create data loaders for the current fold
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx),
                             batch_size=batch_size,
                             num_workers=4, pin_memory=True)

    # Create the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionRecognitionModel(num_emotions).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.unsqueeze(1).to(device)  # Add channel dimension
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluation on train set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in train_loader:
                data = data.unsqueeze(1).to(device)  # Add channel dimension
                target = target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            train_accuracy = 100 * correct / total
            train_accuracies.append(train_accuracy)

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            true_labels = []
            predicted_labels = []
            for data, target in test_loader:
                data = data.unsqueeze(1).to(device)  # Add channel dimension
                target = target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                true_labels.extend(target.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

            test_accuracy = 100 * correct / total
            test_accuracies.append(test_accuracy)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # Print classification report and confusion matrix for the current fold
    print(f"Classification Report for Fold {fold + 1}:")
    print(classification_report(true_labels, predicted_labels))
    print(f"Confusion Matrix for Fold {fold + 1}:")
    print(confusion_matrix(true_labels, predicted_labels))

    # Save the model weights for the current fold
    model_weights.append(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())



# Plot the train and test accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Print the average accuracies across all folds
print(f"Average Train Accuracy: {np.mean(train_accuracies):.2f}%")
print(f"Average Test Accuracy: {np.mean(test_accuracies):.2f}%")

# Create idx_to_label and label_to_idx mappings
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
idx_to_label = {idx: emotion for idx, emotion in enumerate(emotions)}
label_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}


# Ensemble the model weights by averaging
ensembled_weights = {}
for key in model_weights[0].keys():
    ensembled_weights[key] = torch.stack([weights[key].float() for weights in model_weights]).mean(dim=0)

# Create a new model instance and load the ensembled weights
ensembled_model = EmotionRecognitionModel(num_emotions)
ensembled_model.load_state_dict(ensembled_weights)

# Save the ensembled model
ensembled_model_path = 'ensembled_emotion_recognition_model.pth'
torch.save(ensembled_model.state_dict(), ensembled_model_path)
print(f"Ensembled model saved at: {ensembled_model_path}")
# Save idx_to_label and label_to_idx as JSON files
with open('idx_to_label.json', 'w') as f:
    json.dump(idx_to_label, f)

with open('label_to_idx.json', 'w') as f:
    json.dump(label_to_idx, f)

print("idx_to_label and label_to_idx saved as JSON files.")