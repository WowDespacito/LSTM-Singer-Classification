import sys
import torch
from myModule import bidLSTM
import librosa
import numpy as np
from GenreFeatureData import GenreFeatureData

def extract_audio_features(file):
    "Extract audio features from an audio file for genre classification"
    y, sr = librosa.load(file)
    sample_length = 8*sr
    timeseries_length = 256
    num_series = int(y.shape[0]/sample_length)
    features = np.zeros((num_series, timeseries_length, 33), dtype=np.float64)
    for i in range(num_series):
        mfcc = librosa.feature.mfcc(y=y[sample_length*i:sample_length*(i+1)], sr=sr, hop_length=512, n_mfcc=13)
        spectral_center = librosa.feature.spectral_centroid(y=y[sample_length*i:sample_length*(i+1)], sr=sr, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=y[sample_length*i:sample_length*(i+1)], sr=sr, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(y=y[sample_length*i:sample_length*(i+1)], sr=sr, hop_length=512)

        # print(mfcc.shape)
        # print(mfcc.T[0:timeseries_length, :].shape)

        features[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
        features[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
        features[i, :, 14:26] = chroma.T[0:timeseries_length, :]
        features[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features

def get_genre(model, music_path):
    "Predict genre of music using a trained model"
    feature = torch.from_numpy(extract_audio_features(music_path)).type(torch.Tensor)
    # print(feature.shape)
    feature = feature.permute(1, 0, 2)
    # print(feature.shape)
    prediction, _ = model(feature, None)
    # print(prediction)
    predict_genre = GenreFeatureData().genre_list
    result = []
    prediction = torch.max(prediction, 1)
    probability = prediction[0].detach().numpy()
    genre = prediction[1].detach().numpy()
    for i in range(probability.shape[0]):
        result.append([i , predict_genre[genre[i]], 10**probability[i]])
    return result

if __name__ == '__main__':
    Path = sys.argv[1] if len(sys.argv) == 2 else "./audio/ZhouJieLun.mp3"
    state_dict = torch.load('./result/0427/bidlstm_parameter.pkl')
    MODEL = bidLSTM(
        input_dim=33, hidden_dim=256, batch_size=35, output_dim=5, num_layers=2, bidirectional=True
    )
    MODEL.load_state_dict(state_dict)
    result = get_genre(MODEL, Path)
    print(result)
