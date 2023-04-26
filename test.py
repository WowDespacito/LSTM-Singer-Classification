from lstm_genre_classifier_pytorch import LSTM
import torch
state_dict = torch.load('./result/0412_3/model_parameter.pkl')
MODEL = LSTM(
    input_dim=33, hidden_dim=128, batch_size=35, output_dim=5, num_layers=2
)
MODEL.load_state_dict(state_dict)
print(MODEL)