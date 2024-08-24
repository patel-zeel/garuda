import numpy as np
from numpy import ndarray

from garuda.ops import webm_pixel_to_geo, geo_to_webm_pixel, local_to_geo
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import Union

@jaxtyped(typechecker=beartype)
def yolo_aa_to_geo(yolo_label: Union[str, Float[ndarray, "n 4"], Float[ndarray, "n 5"]], zoom: int, img_center_lat: float, img_center_lon: float, img_width: int, img_height: int) -> Float[ndarray, "n 3"]:
    """
    Convert YOLO label to geographic coordinates.
    
    yolo_label: YOLO label (or str path) in the format [class, x_center, y_center, width, height] or [class, x_center, y_center, width, height, confidence].
        class range: [0, 1, 2, ...]
        x_center range: [0, 1]
        y_center range: [0, 1]
        width range: [0, 1]
        height range: [0, 1]
        confidence range: [0, 1]
    
        Example 1: [0, 0.5, 0.5, 0.1, 0.1]
        Example 2: [0, 0.5, 0.5, 0.1, 0.1, 0.9]

    zoom: Zoom level of the map.
        Range: [0, 20]
        Example: 17

    img_center_lon: Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
        
    img_center_lat: Latitude of the center of the image.
        Range: approx [-85, 85] (valid range for Web Mercator projection)
        Example: 37.7749
        
    img_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    img_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
    
    Returns
    -------
    geo_coords: Geographic coordinates in decimal degrees.
        Format: [class, latitude, longitude]
        Example: [0, 37.7749, -122.4194]
    """
    if isinstance(yolo_label, str):
        yolo_label = np.loadtxt(yolo_label, ndmin=2)
        return yolo_aa_to_geo(yolo_label, zoom, img_center_lat, img_center_lon, img_width, img_height)  # To trigger type/shape checking
    
    # Get bbox center in image coordinates
    x_c = yolo_label[:, 1]
    y_c = yolo_label[:, 2]
    
    # Get bbox center in Web Mercator projection
    bbox_geo = local_to_geo(x_c, y_c, zoom, img_center_lat, img_center_lon, img_width, img_height)
    
    # Append class ID to bbox_geo
    class_ids = yolo_label[:, 0:1]
    output = np.concatenate((class_ids, bbox_geo), axis=1)
    
    return output

@jaxtyped(typechecker=beartype)
def yolo_obb_to_geo(yolo_label: Union[str, Float[ndarray, "n 9"], Float[ndarray, "n 10"]], zoom: int, img_center_lat: float, img_center_lon: float, img_width: int, img_height: int) -> Float[ndarray, "n 3"]:
    """
    Convert YOLO label to geographic coordinates.
    
    yolo_label: YOLO label (or str path) in the format [class, x1, y1, x2, y2, x3, y3, x4, y4] or [class, x1, y1, x2, y2, x3, y3, x4, y4, confidence].
        class range: [0, 1, 2, ...]
        x1, x2, x3, x4 range: [0, 1]
        y1, y2, y3, y4 range: [0, 1]
        confidence range: [0, 1]
    
        Example 1: [0, 0.5, 0.5, 0.1, 0.1, 0.0]
        Example 2: [0, 0.5, 0.5, 0.1, 0.1, 0.0, 0.9]
        
    zoom: Zoom level of the map.
        Range: [0, 20]
        Example: 17

    img_center_lon: Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
        
    img_center_lat: Latitude of the center of the image.
        Range: approx [-85, 85] (valid range for Web Mercator projection)
        Example: 37.7749

    img_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    img_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
    
    Returns
    -------
    geo_coords: Geographic coordinates in decimal degrees.
        Format: [class, latitude, longitude]
        Example: [0, 37.7749, -122.4194]
    """
    
    if isinstance(yolo_label, str):
        yolo_label = np.loadtxt(yolo_label, ndmin=2)
        return yolo_obb_to_geo(yolo_label, zoom, img_center_lat, img_center_lon, img_width, img_height)  # To trigger type/shape checking
    
    # Get bbox center in image coordinates
    xyxyxyxy = yolo_label[:, 1:9]
    x_c = xyxyxyxy[:, ::2].mean(axis=1)
    y_c = xyxyxyxy[:, 1::2].mean(axis=1)
    
    # Get bbox center in Web Mercator projection
    bbox_geo = local_to_geo(x_c, y_c, zoom, img_center_lat, img_center_lon, img_width, img_height)

    # Append class ID to bbox_geo
    class_ids = yolo_label[:, 0:1]
    output = np.concatenate((class_ids, bbox_geo), axis=1)
    
    return output