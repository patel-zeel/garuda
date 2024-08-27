import pytest
import torch
import numpy as np
from ultralytics.utils import ops
from ultralytics.utils.metrics import probiou

from garuda.ops import webm_pixel_to_geo, geo_to_webm_pixel, obb_to_aa, label_studio_csv_to_obb, obb_iou, xyxyxyxy2xywhr, xywhr2xyxyxyxy

# test zoom levels
test_zoom_levels = [0, 8, 17, 20]
test_delta = [(0, 1), (1, 0)]

# helper functions
def project(lat, lon, zoom):
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = (128/np.pi)*(2**zoom)*(lon + np.pi)
    y = (128/np.pi)*(2**zoom)*(np.pi - np.log(np.tan(np.pi/4 + lat/2)))
    return x, y

def inverse_project(x, y, zoom):
    F  = 128 / np.pi * 2 ** zoom
    lon = (x / F) - np.pi
    lat = (2 * np.arctan(np.exp(np.pi - y/F)) - np.pi / 2)
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon

# Testing webm_pixel_to_geo

def test_webm_pixel_to_geo_random():
    zoom = 17
    x = np.random.randint(0, 2**zoom * 256)
    y = np.random.randint(0, 2**zoom * 256)
    zoom = 17
    lat, lon = webm_pixel_to_geo(x, y, zoom)
    lat_, lon_ = inverse_project(x, y, zoom)
    assert np.isclose(lat, lat_), f"lat={lat}, lat_={lat_}"
    assert np.isclose(lon, lon_), f"lon={lon}, lon_={lon_}"
    

@pytest.mark.filterwarnings("error") # if there is a warning, raise an error
def test_webm_pixel_to_geo_lower_limit():
    zoom = 17
    x = 0
    y = 0
    
    lat, lon = webm_pixel_to_geo(x, y, zoom)    
    
    lat_, lon_ = inverse_project(x, y, zoom)
    assert np.isclose(lat, lat_), f"lat={lat}, lat_={lat_}"
    assert np.isclose(lon, lon_), f"lon={lon}, lon_={lon_}"


@pytest.mark.filterwarnings("error") # if there is a warning, raise an error
def test_webm_pixel_to_geo_upper_limit():
    zoom = 17
    x = 2**zoom * 256
    y = 2**zoom * 256
    lat, lon = webm_pixel_to_geo(x, y, zoom)
    lat_, lon_ = inverse_project(x, y, zoom)
    assert np.isclose(lat, lat_), f"lat={lat}, lat_={lat_}"
    assert np.isclose(lon, lon_), f"lon={lon}, lon_={lon_}"

@pytest.mark.parametrize("delta_x, delta_y", test_delta)
@pytest.mark.parametrize("zoom", test_zoom_levels)
def test_webm_pixel_to_geo_beyond_lower_limit(delta_x, delta_y, zoom):
    x = 0 - delta_x
    y = 0 - delta_y
    
    with pytest.warns(UserWarning):
        lat, lon = webm_pixel_to_geo(x, y, zoom)

    lat_, lon_ = inverse_project(x, y, zoom)
    assert np.isclose(lat, lat_), f"lat={lat}, lat_={lat_}"
    assert np.isclose(lon, lon_), f"lon={lon}, lon_={lon_}"
    

@pytest.mark.parametrize("delta_x, delta_y", test_delta)
@pytest.mark.parametrize("zoom", test_zoom_levels)
def test_webm_pixel_to_geo_beyond_upper_limit(delta_x, delta_y, zoom):
    x = 2**zoom * 256 + delta_x
    y = 2**zoom * 256 + delta_y
    
    with pytest.warns(UserWarning):
        lat, lon = webm_pixel_to_geo(x, y, zoom)
    lat_, lon_ = inverse_project(x, y, zoom)
    assert np.isclose(lat, lat_), f"lat={lat}, lat_={lat_}"
    assert np.isclose(lon, lon_), f"lon={lon}, lon_={lon_}"
    
    
# Testing geo_to_webm_pixel

def test_geo_to_webm_pixel_random():
    zoom = 17
    lat = np.random.uniform(-85, 85)
    lon = np.random.uniform(-180, 180)
    x, y = geo_to_webm_pixel(lat, lon, zoom)
    x_, y_ = project(lat, lon, zoom)
    assert np.isclose(x, x_), f"x={x}, x_={x_}"
    assert np.isclose(y, y_), f"y={y}, y_={y_}"
    

@pytest.mark.filterwarnings("error") # if there is a warning, raise an error
def test_geo_to_webm_pixel_lower_limit():
    zoom = 17
    lat = -85.0
    lon = -180.0
    x, y = geo_to_webm_pixel(lat, lon, zoom)
    x_, y_ = project(lat, lon, zoom)
    assert np.isclose(x, x_), f"x={x}, x_={x_}"
    assert np.isclose(y, y_), f"y={y}, y_={y_}"
    

@pytest.mark.filterwarnings("error") # if there is a warning, raise an error
def test_geo_to_webm_pixel_upper_limit():
    zoom = 17
    lat = 85.0
    lon = 180.0
    x, y = geo_to_webm_pixel(lat, lon, zoom)
    x_, y_ = project(lat, lon, zoom)
    assert np.isclose(x, x_), f"x={x}, x_={x_}"
    assert np.isclose(y, y_), f"y={y}, y_={y_}"


@pytest.mark.parametrize("delta_lat, delta_lon", test_delta)
@pytest.mark.parametrize("zoom", test_zoom_levels)
def test_geo_to_webm_pixel_beyond_lower_limit(delta_lat, delta_lon, zoom):
    lat = -85.1 - delta_lat*0.01
    lon = -180 - delta_lon*0.01
    with pytest.warns(UserWarning):
        x, y = geo_to_webm_pixel(lat, lon, zoom)
    x_, y_ = project(lat, lon, zoom)
    assert np.isclose(x, x_), f"x={x}, x_={x_}"
    assert np.isclose(y, y_), f"y={y}, y_={y_}"


@pytest.mark.parametrize("delta_lat, delta_lon", test_delta)
@pytest.mark.parametrize("zoom", test_zoom_levels)
def test_geo_to_webm_pixel_beyond_upper_limit(delta_lat, delta_lon, zoom):
    lat = 85.1 + delta_lat*0.01
    lon = 180 + delta_lon*0.01
    with pytest.warns(UserWarning):
        x, y = geo_to_webm_pixel(lat, lon, zoom)
    x_, y_ = project(lat, lon, zoom)
    assert np.isclose(x, x_), f"x={x}, x_={x_}"
    assert np.isclose(y, y_), f"y={y}, y_={y_}"
    
    
def test_obb_to_aa():
    obb_label_path = "data/labels/obb/22.32,87.93.txt"
    aa_label_path = "data/labels/aa/22.32,87.93.txt"
    aa_label = np.loadtxt(aa_label_path, ndmin=2)
    
    # load with path
    obb_to_aa_label = obb_to_aa(obb_label_path)
    assert np.allclose(obb_to_aa_label, aa_label)
    
    # load with numpy array
    obb_label = np.loadtxt(obb_label_path, ndmin=2)
    obb_to_aa_label = obb_to_aa(obb_label)
    assert np.allclose(obb_to_aa_label, aa_label)
    
def test_label_studio_csv_to_obb():
    x1 = 20
    y1 = 10
    width = 30
    height = 40
    rotation = 45
    label = "Zigzag"
    label_map = {"Zigzag": 0}
    
    rotation_rad = np.radians(rotation)
    sin_rot = np.sin(rotation_rad)
    cos_rot = np.cos(rotation_rad)
    
    x_c = x1 + width / 2 * cos_rot - height / 2 * sin_rot
    y_c = y1 + width / 2 * sin_rot + height / 2 * cos_rot
    
    xywhr = np.array([x_c, y_c, width, height, rotation_rad])
    ultralytics_xyxyxyxy = ops.xywhr2xyxyxyxy(xywhr).ravel() / 100
    our_label = label_studio_csv_to_obb(x1, y1, width, height, rotation, label, label_map)
    assert np.allclose(our_label[1:], ultralytics_xyxyxyxy)

def test_xyxyxyxy2xywhr():
    xyxyxyxy = torch.rand(10, 8)
    ultralytics_xywhr = ops.xyxyxyxy2xywhr(xyxyxyxy)
    garuda_xywhr = xyxyxyxy2xywhr(xyxyxyxy.numpy())
    assert np.allclose(ultralytics_xywhr.numpy(), garuda_xywhr, atol=1e-3)
    
def test_xywhr2xyxyxyxy():
    xywhr = torch.rand(10, 5)
    ultralytics_xyxyxyxy = ops.xywhr2xyxyxyxy(xywhr).reshape(-1, 8)
    garuda_xyxyxyxy = xywhr2xyxyxyxy(xywhr.numpy())
    assert np.allclose(ultralytics_xyxyxyxy.numpy(), garuda_xyxyxyxy, atol=1e-3)
    
def test_obb_iou():
    n = 11
    m = 13
    obb1 = torch.rand(n, 8)
    obb2 = torch.rand(m, 8)
    xywhr1 = ops.xyxyxyxy2xywhr(obb1)
    xywhr2 = ops.xyxyxyxy2xywhr(obb2)

    ultralytics_iou_list = []
    for i in range(m):
        ultralytics_iou = probiou(xywhr1, xywhr2[i:i+1])
        ultralytics_iou_list.append(ultralytics_iou)
        
    ultralytics_iou = torch.cat(ultralytics_iou_list, dim=1)
    
    garuda_iou = obb_iou(obb1.numpy(), obb2.numpy())
    assert np.allclose(ultralytics_iou.numpy(), garuda_iou, atol=1e-3)