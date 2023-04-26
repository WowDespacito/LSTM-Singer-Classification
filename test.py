import matplotlib.pyplot as plt

Epoch_list = [1, 2, 3]
tra_loss_list = [1,1,1]
epoch_list = [1, 2, 3]
val_loss_list = [2,2,2]
val_accuracy_list=[3,3,3]
tra_accuracy_list=[4,4,4]


fig, ax = plt.subplots(1, 2)
ax1 = ax[0]
ax2 = ax[1]
ax1.plot(Epoch_list, tra_loss_list, label='train')
ax1.plot(epoch_list, val_loss_list, label='val')
ax1.set_xlabel("epochs")
ax1.set_ylabel("Loss")
ax1.set_title("bidLSTM: Loss")
# visualization accuracy
ax2.plot(epoch_list, val_accuracy_list, label='train')
ax2.plot(Epoch_list, tra_accuracy_list, label='val')
ax2.set_xlabel("epochs")
ax2.set_ylabel("Accuracy")
ax2.set_title("bidLSTM: Accuracy")
plt.legend()
plt.tight_layout()
plt.show()