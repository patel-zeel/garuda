import numpy as np
from garuda.od import yolo_aa_to_geo, ConfusionMatrix


def test_yolo_aa_to_geo():
    yolo_label = np.array([[0, 0.5, 0.5, 0.1, 0.1], [0, 0.5, 0.6, 0.3, 0.1]]).reshape(-1, 5)
    img_center_lon = -122.4194
    img_center_lat = 37.7749
    zoom = 17
    img_width = 1120
    img_height = 1120
    geo = yolo_aa_to_geo(yolo_label, zoom, img_center_lat, img_center_lon, img_width, img_height)
    print(geo)
    
def test_confusion_matrix():
    predictions = [np.random.rand(7, 10).astype(np.float32), np.random.rand(11, 10).astype(np.float32)]
    predictions[0][:, 0] = 0
    predictions[1][:, 0] = 1
    true_labels = [np.random.rand(13, 9).astype(np.float32), np.random.rand(15, 9).astype(np.float32)]
    true_labels[0][:, 0] = 0
    true_labels[1][:, 0] = 1
    
    cm = ConfusionMatrix.from_obb_tensors(predictions, true_labels, classes=["FCBK", "Zigzag"], iou_threshold=0.5, conf_threshold=0.25)
    print(cm)