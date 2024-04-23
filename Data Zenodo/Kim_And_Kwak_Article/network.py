import os
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Enable eager execution at program startup
tf.compat.v1.enable_eager_execution()

# Load the models directly from TensorFlow Hub
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def prepare_waveform(file_path, target_sr=16000, duration=1.0):
    """Prepare and process the waveform."""
    audio, sr = librosa.load(file_path, sr=target_sr)
    if len(audio) > sr * duration:
        audio = audio[:int(sr * duration)]
    elif len(audio) < sr * duration:
        audio = np.pad(audio, (0, int(sr * duration) - len(audio)))

    # Normalize audio
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    return audio.reshape(-1,)  # Reshape to match the expected shape for VGGish

def get_features(audio):
    """Extract features using the loaded models."""
    # Process with VGGish
    vggish_embeddings = vggish_model(audio)

    # Process with YAMNet
    yamnet_results = yamnet_model(audio)
    scores, embeddings, spectrogram = yamnet_results

    return vggish_embeddings, embeddings, spectrogram

# Example usage
file_path = "../Data Zenodo/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
audio = prepare_waveform(file_path)
vggish_embeddings, yamnet_embeddings, yamnet_spectrogram = get_features(audio)

# Plot VGGish embeddings
plt.figure(figsize=(10, 4))
plt.imshow(np.expand_dims(vggish_embeddings[0], axis=0), aspect='auto', cmap='viridis')
plt.title('VGGish Features')
plt.colorbar()
plt.savefig("kim_and_kwak_article_vggish_embeddings.png")
plt.show()

# Plot YAMNet spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(np.transpose(yamnet_spectrogram), aspect='auto', cmap='viridis', origin='lower')
plt.title('YAMNet Spectrogram')
plt.colorbar()
plt.savefig("kim_and_kwak_article_yamnet_spectrogram.png")
plt.show()


import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import tensorflow_hub as hub

# Load pretrained models
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
yamnet_model = hub.KerasLayer(hub.load("https://tfhub.dev/google/yamnet/1"))

# Helper function to plot spectrogram
def plot_spectrogram(spec, title=None):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.savefig("kim_and_kwak_article_spectrogram.png")
    plt.show()

# Function to apply STFT and plot
def process_audio_files(file_path):
    y, sr = librosa.load(file_path, sr=None)
    f, t, Zxx = stft(y, fs=sr, nperseg=1024, noverlap=256)
    spec = np.abs(Zxx)
    plot_spectrogram(librosa.amplitude_to_db(spec), title='Spectrogram')

    # Prepare waveform for VGGish and YAMNet
    waveform = y / 32768.0  # Assuming the audio is 16-bit PCM
    waveform = waveform.astype(np.float32)  # Ensure waveform is float32

    # Ensure waveform is in the correct shape for VGGish and YAMNet
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]  # Add batch dimension

    # Processing with VGGish
    try:
        embeddings_vggish = vggish_model(waveform)
    except Exception as e:
        print(f"Error processing with VGGish: {e}")
        embeddings_vggish = None

    # Processing with YAMNet
    try:
        scores, embeddings_yamnet, spectrogram = yamnet_model(waveform)
    except Exception as e:
        print(f"Error processing with YAMNet: {e}")
        scores, embeddings_yamnet, spectrogram = None, None, None

    return embeddings_vggish, embeddings_yamnet, spectrogram

# Main execution loop
if __name__ == "__main__":
    parent_dir = "../Data Zenodo/Audio_Speech_Actors_01-24"
    actors = ["Actor_01", "Actor_02"]
    sample_files = 10
    all_files = []
    for actor in actors:
        actor_path = os.path.join(parent_dir, actor)
        all_files.extend([os.path.join(actor_path, f) for f in os.listdir(actor_path) if f.endswith('.wav')])

    selected_files = np.random.choice(all_files, sample_files, replace=False)

    for file in selected_files:
        embeddings_vggish, embeddings_yamnet, spectrogram = process_audio_files(file)
        if embeddings_vggish is not None and embeddings_yamnet is not None:
            # Further processing can be done here
            pass
import os
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Enable eager execution at program startup
tf.compat.v1.enable_eager_execution()

# Load the models directly from TensorFlow Hub
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def prepare_waveform(file_path, target_sr=16000, duration=1.0):
    """Prepare and process the waveform."""
    audio, sr = librosa.load(file_path, sr=target_sr)
    if len(audio) > sr * duration:
        audio = audio[:int(sr * duration)]
    elif len(audio) < sr * duration:
        audio = np.pad(audio, (0, int(sr * duration) - len(audio)))

    # Normalize audio
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    return audio.reshape(-1,)  # Reshape to match the expected shape for VGGish

def get_features(audio):
    """Extract features using the loaded models."""
    # Process with VGGish
    vggish_embeddings = vggish_model(audio)

    # Process with YAMNet
    yamnet_results = yamnet_model(audio)
    scores, embeddings, spectrogram = yamnet_results

    return vggish_embeddings, embeddings, spectrogram

def plot_vggish_embeddings(vggish_embeddings):
    """Plot VGGish embeddings."""
    plt.figure(figsize=(10, 4))
    plt.imshow(np.expand_dims(vggish_embeddings[0], axis=0), aspect='auto', cmap='viridis')
    plt.title('VGGish Features')
    plt.colorbar()
    plt.show()

def plot_yamnet_embeddings(yamnet_embeddings):
    """Plot YAMNet embeddings."""
    # Additional visualization for YAMNet embeddings can be added here
    pass

def plot_other_visualization(feature):
    """Plot additional visualization for any other feature."""
    # Additional visualization for other features can be added here
    pass

# Example usage
file_path = "../Data Zenodo/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
audio = prepare_waveform(file_path)
vggish_embeddings, yamnet_embeddings, yamnet_spectrogram = get_features(audio)

# Plot VGGish embeddings
plot_vggish_embeddings(vggish_embeddings)

# Plot YAMNet spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(np.transpose(yamnet_spectrogram), aspect='auto', cmap='viridis', origin='lower')
plt.title('YAMNet Spectrogram')
plt.colorbar()
plt.show()
import os
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Enable eager execution at program startup
tf.compat.v1.enable_eager_execution()

# Load the models directly from TensorFlow Hub
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def prepare_waveform(file_path, target_sr=16000, duration=1.0):
    """Prepare and process the waveform."""
    audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
    # Normalize audio
    audio = audio.astype(np.float32) / np.max(np.abs(audio))
    return audio

def get_features(audio):
    """Extract features using the loaded models."""
    # Process with VGGish
    vggish_embeddings = vggish_model(audio)

    # Process with YAMNet
    yamnet_results = yamnet_model(audio)
    scores, embeddings, spectrogram = yamnet_results

    return vggish_embeddings, embeddings, spectrogram

def plot_vggish_embeddings(vggish_embeddings):
    """Plot VGGish embeddings."""
    plt.figure(figsize=(10, 4))
    plt.imshow(np.expand_dims(vggish_embeddings[0], axis=0), aspect='auto', cmap='viridis')
    plt.title('VGGish Features')
    plt.colorbar()
    plt.show()

def plot_yamnet_spectrogram(spectrogram):
    """Plot YAMNet spectrogram."""
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(spectrogram), aspect='auto', cmap='viridis', origin='lower')
    plt.title('YAMNet Spectrogram')
    plt.colorbar()
    plt.show()

# Example usage
file_path = "../Data Zenodo/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
audio = prepare_waveform(file_path)
audio = np.expand_dims(audio, axis=0)  # Add batch dimension

vggish_embeddings, yamnet_embeddings, yamnet_spectrogram = get_features(audio)

# Plot VGGish embeddings
plot_vggish_embeddings(vggish_embeddings)

# Plot YAMNet spectrogram
plot_yamnet_spectrogram(yamnet_spectrogram)
