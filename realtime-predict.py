# CV：基于keras利用训练好的hdf5模型进行目标检测实现输出模型中的表情或性别的gradcam——Jason Niu

import sys

import cv2
import numpy as np
from keras.models import load_model

# getting the correct model given the input
# 1、首先指定想实现人脸灰凸特征图像(salient region detection)a检测的是emotion还是gender
# task = sys.argv[1]
# class_name = sys.argv[2]
task = 'emotion'
# task = 'gender'

# 2、if条件判断给定的是性别模型还是表情模型
if task == 'gender':
    model_filename = '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'
    class_to_arg = get_class_to_arg('imdb')
    #     predicted_class = class_to_arg[class_name]
    predicted_class = 0
    offsets = (0, 0)
elif task == 'emotion':
    model_filename = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'  # 默认开启
    #     model_filename = '../trained_models/fer2013_big_XCEPTION.54-0.66.hdf5'
    class_to_arg = get_class_to_arg('fer2013')
    #     predicted_class = class_to_arg[class_name]
    predicted_class = 1
    #     predicted_class = 'fear'
    offsets = (0, 0)

# 3、加载模型、梯度函数，指导模型、凸函数(灰凸化特征)
model = load_model(model_filename, compile=False)
gradient_function = compile_gradient_function(model, predicted_class,
                                              'conv2d_7')  # 调用compile_gradient_function编译梯度函数，返回名称为conv2d_7的卷积层
register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp', task)  # 调用modify_backprop函数，修改CNN更新为一个新的模型
saliency_function = compile_saliency_function(guided_model,
                                              'conv2d_7')  # 调用compile_saliency_function函数，激活层采用conv2d_7层；saliency是指灰色图像下凸出特征

# parameters for loading data and images 加载人脸检测识别默认库haarcascade_frontalface_default.xml
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
face_detection = load_detection_model(detection_model_path)
color = (0, 255, 0)  # 绿色

# getting input model shapes for inference获取输入模型形状进行推理(输入hadf5库内张量集合中的下标1~3)
target_size = model.input_shape[1:3]  # 输入hadf5库内张量集合中的下标1~3

# starting lists for calculating modes表情窗口列表初始化：通过计算模型，开始列表
emotion_window = []

# 4、打开本地摄像头，进行实时捕捉实现salient region detection灰凸特征图像(绘制面部方框)
# starting video streaming 第一步、先定义摄像头窗口名称，再打开摄像头，并开始实时读取画面
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
# 第二步、while循环间隔刷新图像实时捕获人脸，实现人脸变为凸优化特征图像
while True:
    bgr_image = video_capture.read()[1]  # 从摄像设备中实时读入图像数据，(第一个参数[0]表示读取是否成功，第二个参数[1]是读取的图像)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)  # 分别将读取的图像进行灰化、RGB化处理
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection,
                         gray_image)  # detect_faces函数：调用detectMultiScale函数进行识别人脸(检测出图片中所有的人脸)，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用

    for face_coordinates in faces:  # for循环对人脸表情进行实时将图像进行灰凸化特征

        x1, x2, y1, y2 = apply_offsets(face_coordinates,
                                       offsets)  # apply_offsets函数：大概是根据图像实时偏移( HoG检测窗口移动时的步长,原图外围添加像素)
        gray_face = gray_image[y1:y2, x1:x2]  # [坐标参数，尺寸参数]
        try:
            gray_face = cv2.resize(gray_face,
                                   (target_size))  # cv2.resize(image, image2,dsize) 图像缩放方法;即(输入原始图像，输出新图像，图像的大小)
        except:
            continue

        gray_face = preprocess_input(gray_face, True)  # preprocess_input函数先将gray_face转换为'float32'然后 /255.0
        gray_face = np.expand_dims(gray_face, 0)  # 在标签数据上增加一个维度，0是增加在第一个轴上
        gray_face = np.expand_dims(gray_face, -1)
        guided_gradCAM = calculate_guided_gradient_CAM(gray_face,
                                                       gradient_function,
                                                       saliency_function)  # calculate_guided_gradient_CAM函数？
        guided_gradCAM = cv2.resize(guided_gradCAM, (x2 - x1, y2 - y1))
        try:
            rgb_guided_gradCAM = np.repeat(guided_gradCAM[:, :, np.newaxis],
                                           3, axis=2)
            rgb_image[y1:y2, x1:x2, :] = rgb_guided_gradCAM
        except:
            continue
        draw_bounding_box((x1, y1, x2 - x1, y2 - y1), rgb_image, color)  # draw_bounding_box函数：在人脸区域画一个正方形出来

    # 输出图像先颜色空间转换，然后命名窗口、显示窗口、结束程序条件：cv2.waitKey函数用来检测特定键q是否被按下，则break直接跳出当前循环,也就是结束了
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # 颜色空间转换
    # 命名窗口、显示窗口、结束程序条件：cv2.waitKey函数用来检测特定键q是否被按下，则break退出程序
    try:
        cv2.imshow('window_frame', bgr_image)
    except:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):

