"""
Script per la preparazione dei dati: caricamento segnali e label
"""
import numpy as np
import os
from pathlib import Path
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split
import pickle
from utils.getIndexesFromNumber import getIndexesFromNumber

def load_signal(file_path):
    """
    Carica un segnale da file .npy
    
    Args:
        file_path: percorso del file .npy contenente il segnale
        
    Returns:
        numpy array con il segnale
    """
    return np.load(file_path)


def compute_mel_spectrogram(time_signal, sr=22050, n_mels=65, n_fft=2048, hop_length=512):
    """
    Calcola lo spettrogramma Mel da un segnale temporale
    
    Args:
        time_signal: segnale nel dominio del tempo
        sr: sample rate
        n_mels: numero di bande mel
        n_fft: dimensione FFT
        hop_length: hop length per STFT
        
    Returns:
        spettrogramma mel in dB
    """
    mel_spec = librosa.feature.melspectrogram(
        y=time_signal, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Converti in scala dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def compute_stft(time_signal, nperseg=256, noverlap=128):
    """
    Calcola lo spettrogramma STFT da un segnale temporale
    
    Args:
        time_signal: segnale nel dominio del tempo
        nperseg: lunghezza di ogni segmento
        noverlap: numero di punti di overlap
        
    Returns:
        spettrogramma STFT (magnitudine)
    """
    f, t, Zxx = signal.stft(time_signal, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)


def load_dataset_from_folders(dataset_path):

    """
    Carica dataset organizzato in cartelle (ogni cartella = una classe)
    
    Args:
        dataset_path: percorso della cartella principale del dataset
        transform_func: funzione opzionale per trasformare i segnali (es. compute_mel_spectrogram)
        max_samples_per_class: numero massimo di campioni per classe (None = tutti)
        
    Returns:
        X: array di dati (shape dipende dalla trasformazione)
        y: array di label (nomi delle classi)
    """
    if dataset_path is None:
        dataset_path = '../dataset/test_object_1_1mm_spectrograms/'

    labels_path = Path(dataset_path).parent / 'labels.txt'
    labels = np.loadtxt(labels_path, dtype=int)
    X , Y = [], []
    for i in range(1, 1601):
        indexes = getIndexesFromNumber(i)  
        file_path = dataset_path + indexes.split()[0] + '/' + indexes + '.npy'
        signal = load_signal(file_path)
        X.append(signal)
        Y.append(labels[i - 1])  # i-1 perché l'array parte da 0

    return np.array(X), np.array(Y)


def prepare_data_for_cnn(X, target_shape=None):
    """
    Prepara i dati per la CNN aggiungendo la dimensione del canale
    
    Args:
        X: array di dati (N, H, W) o (N, L)
        target_shape: shape target opzionale per reshape/padding
        
    Returns:
        X con dimensione canale aggiunta: (N, H, W, 1)
    """
    # Se è 1D, aggiungi dimensioni
    if len(X.shape) == 2:
        X = X[..., np.newaxis, np.newaxis]  # (N, L) -> (N, L, 1, 1)
    # Se è 2D (spettrogramma), aggiungi solo canale
    elif len(X.shape) == 3:
        X = X[..., np.newaxis]  # (N, H, W) -> (N, H, W, 1)
    
    return X


def normalize_data(X, method='standard'):
    """
    Normalizza i dati
    
    Args:
        X: array di dati
        method: 'standard' (mean=0, std=1), 'minmax' (0-1), 'per_sample' (per campione)
        
    Returns:
        X normalizzato, parametri di normalizzazione
    """
    if method == 'standard':
        mean = np.mean(X)
        std = np.std(X)
        X_norm = (X - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(X)
        max_val = np.max(X)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
        
    elif method == 'per_sample':
        # Normalizza ogni campione indipendentemente
        X_norm = np.zeros_like(X)
        for i in range(len(X)):
            mean = np.mean(X[i])
            std = np.std(X[i])
            X_norm[i] = (X[i] - mean) / (std + 1e-8)
        params = None
    
    else:
        raise ValueError(f"Metodo di normalizzazione '{method}' non riconosciuto")
    
    return X_norm, params


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Divide il dataset in train, validation e test set
    
    Args:
        X: dati
        y: label
        test_size: frazione per test set
        val_size: frazione per validation set (rispetto al train)
        random_state: seed per riproducibilità
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Prima divisione: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y.argmax(axis=1)
    )
    
    # Seconda divisione: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_train_val.argmax(axis=1)
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_labels(y):
    """
    Codifica le label in formato one-hot
    
    Args:
        y: array di label (1D)
        
    Returns:
        y_encoded: label in formato one-hot (2D)
    """
    from tensorflow.keras.utils import to_categorical
    
    
    unique_labels = np.unique(y)
    print(f"Label uniche: {unique_labels}")
    
    # Codifica in one-hot
    y_encoded = to_categorical(y, num_classes=len(unique_labels))
    
    return y_encoded


def save_preprocessed_data(filename, **data):
    """
    Salva i dati preprocessati in un file pickle
    
    Args:
        filename: nome del file di output
        **data: dizionario di dati da salvare
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dati salvati in '{filename}'")


def load_preprocessed_data(filename):
    """
    Carica i dati preprocessati da un file pickle
    
    Args:
        filename: nome del file da caricare
        
    Returns:
        dizionario con i dati
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Dati caricati da '{filename}'")
    return data


if __name__ == "__main__":
    print("=== Esempio di preparazione dati ===\n")
    
    
    dataset_path = "dataset/test_object_2_3mm_mel_spectrograms/"
    output_file = "preprocessed_data_spectrograms_2_3mm_mel.pkl"
    
    # 1. Carica i dati dalle cartelle
    print("Caricamento dati...")
    X, y = load_dataset_from_folders(
        dataset_path
    )
    
    print(f"\nDati caricati: {X.shape}")
    print(f"Label caricate: {y.shape}")
    
    # 2. Prepara per CNN 
    X = prepare_data_for_cnn(X)
    print(f"Shape dopo prepare_data_for_cnn: {X.shape}")
    
    # 3. Normalizza i dati
    X, norm_params = normalize_data(X, method='standard')
    print(f"Dati normalizzati (mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f})")
    
    y_encoded = encode_labels(y)
    print(f"Label codificate in one-hot: {y_encoded.shape}")
    # 5. Dividi in train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y_encoded, test_size=0.2, val_size=0.1
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 6. Salva i dati preprocessati
    save_preprocessed_data(
        output_file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        norm_params=norm_params
    )
    
    print("\n=== Preparazione completata! ===")
