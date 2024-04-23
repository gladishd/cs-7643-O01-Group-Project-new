import os  # Add this import statement
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, LSTM
import matplotlib.pyplot as plt  # Import Matplotlib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model

# Directory where the plots will be saved
plot_directory = 'Images_Empathic_Voice_Interface_Model'
os.makedirs(plot_directory, exist_ok=True)  # Create the directory if it doesn't exist
emotions = ['happy', 'sad', 'angry', 'neutral']


# "Ensure" your preprocess_audio function is defined here as well
def preprocess_audio(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Transpose the result to align with the expected input format and take the mean
    mfccs_processed = np.mean(mfccs.T,axis=0)

    return mfccs_processed

from tensorflow.keras.layers import Input

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=5, activation='relu'),
        Conv1D(64, kernel_size=5, activation='relu'),
        Dropout(0.5),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])
    return model

def load_and_preprocess_data():
    dataset_path = 'Data Zenodo/Audio_Song_Actors_01-24'
    emotions = ['happy', 'sad', 'angry', 'neutral']
    features = []
    labels = []

    for emotion in emotions:
        # Construct the path to the emotion directory
        emotion_path = os.path.join(dataset_path, emotion)
        # Check if the emotion directory exists
        if not os.path.isdir(emotion_path):
            print(f"Directory does not exist: {emotion_path}")
            continue

        files = glob.glob(os.path.join(emotion_path, '*.wav'))
        if not files:
            print(f"No .wav files found in {emotion_path}")
            continue

        for file in files:
            mfcc = preprocess_audio(file)  # Ensure this function returns an expected array
            features.append(mfcc)
            labels.append(emotion)

    if not features or not labels:
        print("No features or labels have been loaded. Check the dataset path and structure.")
        return None, None, None, None

    features = np.array(features)
    labels = np.array(labels)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_preprocess_data()

if X_train is not None:
    # Proceed with model creation, compilation, and training
    # Ensure y_train is accessible for model creation
    model = create_model((X_train.shape[1], 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))
    model_path = 'emotion_model.keras'
    model.save(model_path)
else:
    print("Data loading failed. Training aborted.")

# Model training and history capturing...
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Save plots to the directory
def save_plot(figure, filename):
    figure.savefig(os.path.join(plot_directory, filename))


# Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Empathic Voice Interface Model (The Model of Peter). Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
save_plot(plt, 'empathic_voice_interface_loss_plot.png')

# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Empathic Voice Interface Model (Peter\'s Model, and so much more). Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
save_plot(plt, 'empathic_voice_interface_accuracy_plot.png')

# Confusion Matrix
# Note: You will need to obtain your model's predictions and the true labels to generate this plot.
predictions = model.predict(X_test_reshaped)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Empathic Voice Interface. Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
save_plot(plt, 'empathic_voice_interface_confusion_matrix.png')

# Model Architecture Plot
plot_model(model, to_file=os.path.join(plot_directory, 'empathic_voice_interface_model_structure.png'), show_shapes=True)

# Histogram of Label Distribution in the Training Data
plt.figure(figsize=(10, 5))
plt.hist(np.argmax(y_train, axis=1), bins=np.arange(len(emotions)+1)-0.5, rwidth=0.8)
plt.title('Empathic Voice Interface. Distribution of Emotions in Training Data')
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.xticks(ticks=np.arange(len(emotions)), labels=emotions)
save_plot(plt, 'empathic_voice_interface_label_distribution.png')

# Ensure to close plots to avoid memory issues
plt.close('all')





from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming model.predict() returns a numpy array of predictions
# For binary classification, use model.predict(). For multi-class, use model.predict_proba() if available
predictions = model.predict(X_test_reshaped)


# Assuming y_train and y_test have already been binarized or are binary labels
# your_model_predictions should contain probability estimates of the positive class

# ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test.ravel(), predictions.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve and AUC Score')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
save_plot(plt, 'ROC_Curve.png')

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test.ravel(), predictions.ravel())

plt.figure(figsize=(10, 5))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
save_plot(plt, 'Precision_Recall_Curve.png')

# Predictions need to be converted from probabilities to binary labels if using softmax
y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
y_true = np.argmax(y_test, axis=1)
from sklearn.metrics import f1_score

# Calculate F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1}")


# T-SNE for Visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=1, n_iter=300)
X_tsne = tsne.fit_transform(X_train)

# Assuming y_train is one-hot encoded
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Example of using a colormap correctly
colors = np.argmax(y_train, axis=1)  # This assumes y_train is one-hot encoded
cmap = plt.get_cmap('viridis', np.unique(colors).size)  # Get a colormap with enough colors
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap=cmap)  # Use the colormap
plt.colorbar(ticks=np.arange(np.unique(colors).size))  # Optional: Add a colorbar
plt.title('T-SNE Visualization')
plt.xlabel('TSNE Feature 1')
plt.ylabel('TSNE Feature 2')
save_plot(plt, 'T-SNE_Visualization.png')
plt.show()
