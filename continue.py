
"""""
Description: 训练人脸表情识别程序
"""
import warnings
import os
import keras
import tensorflow as tf
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #为使用CPU

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import big_XCEPTION
from models.cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.callbacks import TensorBoard #训练可视化
from draw import plot_confuse #h混淆矩阵可视化
from matplotlib import pyplot as plt #准确率可视化


from keras.models import load_model

model = load_model('./models/_miniF_XCEPTION.07-0.56.hdf5')


# 参数
batch_size = 32
num_epochs = 10
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7 #分类的类别个数
patience = 50
base_path = 'models/'

# 构建模型
# 编译模型，指定优化器，损失函数，度量
model.compile(optimizer='adam',  # 优化器采用adam
              loss='categorical_crossentropy',  # 多分类的对数损失函数
              metrics=['accuracy'])

#输出模型各层的参数状况
#Param，它表示每个层参数的个数
model.summary()

#keras.utils.plot_model(model, 'multi_input_and_output_big-model.png', show_shapes=True) #打制网络结构图
#plot_model(model, "big-model.png") #打印网洛结构图
#TensorBoard = tf.keras.callbacks.TensorBoard(log_dir="./log",histogram_freq=1,write_graph=True)# 绘制损失函数和准确率

# 定义回调函数 Callbacks 用于训练过程
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience / 4),
                              verbose=1)

# 模型位置及命名
trained_models_path = base_path + '_miniF_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

# 定义模型权重位置、命名等，该回调函数将在每个epoch后保存模型到filepath
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]  #回调函数EarlyStopping：当训练过程中验证指标不再变化时，停止训练
                                                                    #CSVLogger：将损失和指标数据流式传输到CSV文件
#TensorBoard = tf.keras.callbacks.TensorBoard(log_dir="./log", histogram_freq=1, write_graph=True)# 绘制损失函数和准确率

# 载入数据集
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape

# 划分训练、测试集
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

# 图片产生器，在批量中对数据进行增强，扩充数据集大小
data_generator = ImageDataGenerator(
    featurewise_center=False, #对输入的图片每个通道减去每个通道对应均值
    featurewise_std_normalization=False,
    rotation_range=20, #10改20  #旋转范围
    width_shift_range=0.1, #水平平移范围
    height_shift_range=0.1, #垂直平移范围, 浮点数、一维数组或整数（同width_shift_range）
    zoom_range=.1, #缩放范围，参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作
    horizontal_flip=True) #水平反转

# 模型拟合，即训练
# 利用数据增强进行训练

history = model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),
                    steps_per_epoch=len(xtrain) / batch_size,
                    epochs=num_epochs,
                    verbose=1, callbacks=callbacks, #TensorBoard换callbacks
                    validation_data=(xtest, ytest))

labels=["anger","disgust","fear","happy","sad","surprise","netural"]
plot_confuse(model,xtest,ytest,labels ,batch_size)