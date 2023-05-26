import torch
from sklearn.metrics import f1_score
import math
import myModule
from GenreFeatureData import (
    GenreFeatureData,
)
import torch

def recall(model, X, Y, batch_size, num_batches):
    """
    计算模型在给定数据集上的召回率
    """
    num_correct = 0
    num_positives = 0

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_X = X[start_idx:end_idx].permute(1, 0, 2)
            batch_Y = Y[start_idx:end_idx]
            # batch_Y=torch.max(batch_Y, 1)[1]

            # print(batch_X.shape)
            # print(batch_Y)
            logits, _ = model(batch_X)
            logits=10**logits
            # _, predictions = torch.max(logits, dim=1)
            # print(logits)
            predictions = torch.sigmoid(logits)

            # print(predictions.size())
            # print(batch_Y.size())

            num_correct += torch.sum(predictions * batch_Y)
            num_positives += torch.sum(batch_Y)

    recall = num_correct / num_positives
    return recall


def f1(model, X, Y, batch_size, num_batches):
    """
    计算模型在给定数据集上的F1得分
    """
    logits, _ = model(X.permute(1, 0, 2))
    _, prediction = torch.max(logits, dim=1)
    _, stand = torch.max(Y, dim=1)
    # print(prediction)
    # print(stand)
    score = f1_score(stand.numpy(), prediction.numpy(), average='weighted')
    # num_correct = 0
    # num_predicted = 0
    # num_true = 0
    #
    # with torch.no_grad():
    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = (i + 1) * batch_size
    #
    #         batch_X = X[start_idx:end_idx].permute(1, 0, 2)
    #         batch_Y = Y[start_idx:end_idx]
    #
    #         logits, _ = model(batch_X)
    #         logits = 10**logits
    #         predictions = torch.sigmoid(logits)
    #         max_values, _ = torch.max(predictions, dim=1, keepdim=True)
    #
    #         # 使用广播操作将每一行最大值置1，其他值置0
    #         predictions[predictions != max_values] = 0
    #         predictions[predictions == max_values] = 1
    #         print(predictions)
    #         # num_correct += torch.sum(torch.eq(predictions, batch_Y))
    #         # num_predicted += torch.sum(predictions)
    #         # num_true += torch.sum(batch_Y)
    #
    # precision = num_correct / num_predicted
    # recall = num_correct / num_true
    # f1 = 2 * (precision * recall) / (precision + recall)

    return score



def evaluate(model, dev_X, dev_Y, batch_size):
    """
    计算模型在验证集上的召回率和F1得分
    """
    with torch.no_grad():
        num_dev_batches = math.ceil(dev_X.shape[0] / batch_size)
        val_recall = recall(model, dev_X, dev_Y, batch_size, num_dev_batches)
        val_f1 = f1(model, dev_X, dev_Y, batch_size, num_dev_batches)

    return val_recall, val_f1
    # return val_recall
    # return val_f1
if __name__ == "__main__":
    # train_on_gpu = torch.cuda.is_available()
    # device = torch.device('cuda' if train_on_gpu else 'cpu')
    batch_size = 100

    # model = myModule.bidLSTM(
    #     input_dim=33, hidden_dim=128, batch_size=batch_size, output_dim=5, num_layers=1, bidirectional=True
    # )
    # state_dict = torch.load('./result/0526_5/bidlstm_parameter.pkl')

    # model = myModule.LSTM(
    #     input_dim=33, hidden_dim=128, batch_size=batch_size, output_dim=5, num_layers=1
    # )
    # state_dict = torch.load('./result/lstm0525_3/lstm_parameter.pkl')

    model = myModule.Att_bidLSTM(
        input_dim=33, hidden_dim=96, batch_size=batch_size, output_dim=5, num_layers=1, bidirectional=True
    )
    state_dict = torch.load('./result/att0526_8/AttBilstm_parameter.pkl')

    model.load_state_dict(state_dict)

    genre_features = GenreFeatureData()
    genre_features.load_deserialize_data()
    dev_X = torch.from_numpy(genre_features.dev_X).type(torch.Tensor)
    dev_Y = torch.from_numpy(genre_features.dev_Y).type(torch.LongTensor)
    print(evaluate(model, dev_X, dev_Y, batch_size))