import torch
from torch import nn
from torch.functional import F


# class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=5, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, input, hidden=None):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        x, state = self.lstm(input, hidden)
        # print(x.shape)
        x = self.dropout1(x[-1])
        # print(x.shape)
        x = self.linear(x)  # equivalent to return_sequences=False from Keras
        # print(x.shape)
        # x = self.dropout2(x)
        # print(x.shape)

        output = F.log_softmax(x, dim=1)
        return output, hidden

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()


class bidLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=5, num_layers=2, bidirectional=True):
        super(bidLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=self.bidirectional)
        self.dropout1 = nn.Dropout(0.5)
        # setup output layer
        self.linear = nn.Linear(2 * self.hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(0.45)

    def forward(self, input, hidden=None):
        x, state = self.lstm(input, hidden)
        # print(x.shape)
        # x = self.dropout1(x)
        # print(x.shape)
        x = self.linear(x[-1])
        # print(x.shape)
        x = self.dropout2(x)
        output = F.log_softmax(x, dim=1)
        return output, hidden

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
    
    
    
    


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, encoder_outputs, decoder_hidden):
        energy = self.linear(decoder_hidden).unsqueeze(1)
        attention_scores = torch.bmm(encoder_outputs, energy.transpose(1, 2)).squeeze(2)
        attention_weights = self.softmax(attention_scores)
        context_vector = torch.bmm(encoder_outputs.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return context_vector, attention_weights

class Att_bidLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=5, num_layers=2, bidirectional=True):
        super(Att_bidLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=self.bidirectional)
        self.dropout1 = nn.Dropout(0.5)
        
        # setup attention layer
        self.attention = Attention(2 * self.hidden_dim)
        
        # setup output layer
        self.linear = nn.Linear(2 * self.hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(0.4)

    def forward(self, input, hidden=None):
        x, state = self.lstm(input, hidden)
        # print(x.shape)
        # x = self.dropout1(x)
        # print(x.shape)
        
        # apply attention
        context_vector, attention_weights = self.attention(x.transpose(0, 1), x[-1])
        # x = self.dropout1(context_vector)
        
        x = self.linear(context_vector)
        # print(x.shape)
        x = self.dropout2(x)
        output = F.log_softmax(x, dim=1)
        return output, hidden

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()
