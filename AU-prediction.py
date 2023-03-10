from feat import Detector
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os
from feat.utils.io import read_feat

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

# Helper to point to the test data folder
#test_data_dir = get_test_data_path()

# Get the full path
single_face_img_path = os.path.join('/picture', "001.jpg")

# Plot it
imshow(single_face_img_path)

single_face_prediction = detector.detect_image(single_face_img_path)
single_face_prediction.to_csv("output.csv", index=False)

input_prediction = read_feat("output.csv")
figs = single_face_prediction.plot_detections(poses=True)