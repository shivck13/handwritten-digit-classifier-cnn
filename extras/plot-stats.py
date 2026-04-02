import matplotlib.pyplot as plt
import csv
import os

file = os.path.join(os.path.dirname(__file__), '../stats/stats.csv')

data = list(csv.reader(open(file)))
header = data[0]
values = data[1:]

def plot_training_statistics():
    epochs = [int(row[0]) for row in values]
    time = [float(row[1]) for row in values]
    train_acc = [float(row[2]) for row in values]
    train_loss = [float(row[3]) for row in values]
    val_acc = [float(row[4]) for row in values]
    val_loss = [float(row[5]) for row in values]

    plt.figure(figsize=(12, 5))

    # plot accuracy, loss, validation accuracy, validation loss per epoch (y range 0-1, x range 1-10)
    # line styles must be different
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Training Accuracy', linestyle='-', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', linestyle='--', marker='o')
    plt.ylim(0.8, 1.0)
    plt.xlim(1, max(epochs))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss', linestyle='-', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', linestyle='--', marker='o')
    plt.ylim(0, 0.6)
    plt.xlim(1, max(epochs))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_statistics()
