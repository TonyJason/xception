import cv2
import numpy as np
from keras.models import Model
import tensorflow as tf


categories = ["anger","disgust","fear","happy","sad","surprise","netural"]

def cv_imread(self, filePath):
    # 读取图片
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ## cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def prepare(path):
    img_size = 48
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


if __name__ == '__main__':

    model = tf.keras.models.load_model('./models/_tiny_XCEPTION.54-0.64.hdf5')

    prediction = model.predict([prepare('./picture/001.jpg')])#输入数据(data)，输出预测结果
    print(prediction)
    max_index = np.argmax(prediction) #输出数组最大值的位置
    print('预测结果：'+categories[max_index])

   # loss, acc = model.evaluate(test_images, test_labels, verbose=2)#输入数据(data)和真实标签(label),然后将预测结果与真实标签相比较,得到两者误差并输出.
   # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))