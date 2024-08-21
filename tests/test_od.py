import numpy as np
from garuda.od import yolo_aa_to_geo


def test_yolo_aa_to_geo():
    yolo_label = np.array([[0, 0.5, 0.5, 0.1, 0.1], [0, 0.5, 0.6, 0.3, 0.1]]).reshape(-1, 5)
    img_center_lon = -122.4194
    img_center_lat = 37.7749
    zoom = 17
    img_width = 1120
    img_height = 1120
    geo = yolo_aa_to_geo(yolo_label, zoom, img_center_lat, img_center_lon, img_width, img_height)
    print(geo)