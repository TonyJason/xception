import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
def main():
    df = pd.read_csv('./models/_emotion_training.log')

    # 绘制折线图
    plt.plot(df['epoch'], df['acc'], label='Training Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(df['epoch'], df['loss'], label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
