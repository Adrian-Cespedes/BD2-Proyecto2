import os
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib

# def extract_features(file_path, n_mfcc=13, duration=30, max_pad_len=1300):
#     try:
#         # Cargar la duración especificada del archivo de audio
#         audio, sample_rate = librosa.load(file_path, sr=None, duration=duration)
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
#         pad_width = max_pad_len - mfccs.shape[1]
#         if pad_width > 0:
#             mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#         else:
#             mfccs = mfccs[:, :max_pad_len]
#         return mfccs.flatten()
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None 

def extract_features(file_path, n_mfcc=13, duration=None, max_pad_len=1300):
    try:
        # Cargar la duración completa del archivo de audio si duration es None
        audio, sample_rate = librosa.load(file_path, sr=None, duration=duration)
        
        # Normalización
        audio = librosa.util.normalize(audio)
        
        # Características adicionales
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        
        # Concatenar características
        features = np.concatenate((mfccs, chroma, mel_spec), axis=0)
        
        pad_width = max_pad_len - features.shape[1]
        if pad_width > 0:
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_pad_len]
        return features.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None 


# def process_songs(directory, n_mfcc=13, duration=30, max_pad_len=1300):
#     features = []
#     filenames = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".m4a"):
#             file_path = os.path.join(directory, filename)
#             feature_vector = extract_features(file_path, n_mfcc, duration, max_pad_len)
#             if feature_vector is not None:
#                 features.append(feature_vector)
#                 filenames.append(filename)

#     return np.array(features), filenames

def process_songs(directory, n_mfcc=13, segment_duration=3.0, max_pad_len=1300):
    features = []
    filenames = []

    for filename in os.listdir(directory):
        if filename.endswith(".m4a"):
            file_path = os.path.join(directory, filename)
            feature_vector = extract_segmented_features(file_path, segment_duration, n_mfcc, max_pad_len)
            if feature_vector is not None:
                features.append(feature_vector)
                filenames.append(filename)

    return np.array(features), filenames


def save_features(features, filenames, output_file):
    df = pd.DataFrame(features)
    df['filename'] = filenames
    df.to_csv(output_file, index=False)

def load_features(file_path):
    df = pd.read_csv(file_path)
    filenames = df['filename'].values
    features = df.drop(columns=['filename']).values
    return features, filenames

# def reduce_dimensionality(features, n_components=0.95):
#     pca = PCA(n_components=n_components)
#     reduced_features = pca.fit_transform(features)
#     # Guardar el modelo PCA
#     joblib.dump(pca, 'pca_model.pkl')
#     return reduced_features

def reduce_dimensionality(features, n_components=0.99):  # Ajustar n_components para mantener más varianza
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    joblib.dump(pca, 'pca_model.pkl')
    return reduced_features


# def reduce_new_input(file_path, pca):
#     # Extraer características del nuevo archivo de audio
#     feature_vector = extract_features(file_path)
#     if feature_vector is not None:
#         feature_vector = np.array([feature_vector])  # Convertir a 2D array
#         # Reducir dimensionalidad utilizando el modelo PCA cargado
#         reduced_feature_vector = pca.transform(feature_vector)
#         return reduced_feature_vector
#     else:
#         return None
    
def reduce_new_input(file_path, pca, segment_duration=3.0, n_mfcc=13):
    feature_vector = extract_segmented_features(file_path, segment_duration, n_mfcc)
    if feature_vector is not None:
        feature_vector = np.array([feature_vector])  # Convertir a 2D array
        reduced_feature_vector = pca.transform(feature_vector)
        return reduced_feature_vector
    else:
        return None


def reduce_new_segmented_input(feature_vector, pca):
    reduced_feature_vector = pca.transform(np.array([feature_vector]))
    return reduced_feature_vector
    
##########

def extract_segmented_features(file_path, segment_duration=3.0, n_mfcc=13, max_pad_len=1300):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        total_duration = librosa.get_duration(y=audio, sr=sample_rate)
        segment_features = []
        
        for start in range(0, int(total_duration), int(segment_duration)):
            end = start + segment_duration
            if end > total_duration:
                break
            segment = audio[int(start*sample_rate):int(end*sample_rate)]
            features = extract_features_from_segment(segment, sample_rate, n_mfcc, max_pad_len)
            if features is not None:
                segment_features.append(features)
        
        if segment_features:
            # Promediar las características segmentadas y aplanarlas
            segment_features = np.mean(segment_features, axis=0)
            if segment_features.shape[0] < max_pad_len:
                # Padding si la longitud es menor que max_pad_len
                pad_width = max_pad_len - segment_features.shape[0]
                segment_features = np.pad(segment_features, (0, pad_width), mode='constant')
            else:
                # Truncar si la longitud es mayor que max_pad_len
                segment_features = segment_features[:max_pad_len]
            return segment_features
        else:
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features_from_segment(segment, sample_rate, n_mfcc, max_pad_len):
    try:
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs.flatten()
    except Exception as e:
        print(f"Error processing segment: {e}")
        return None

