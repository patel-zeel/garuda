import numpy as np
from numpy import ndarray
import pyproj
from typing import Tuple, Union
import warnings
from beartype import beartype
from jaxtyping import Float, jaxtyped

# @jaxtyped(typechecker=beartype)
def local_to_geo(x: float, y: float, zoom: int, img_center_lat: float, img_center_lon: float, img_width: int, img_height: int) -> Union[Float[ndarray, "2"], Float[ndarray, "n 2"]]:
    """
    Convert local coordinates to Web Mercator projection at a given zoom level for given image center and image dimensions.
    
    Parameters
    ----------
    x : X-coordinate in local coordinates. Generally mentioned in the YOLO label.
        Range: [0, 1]
        Example: 0.5
    
    y : Y-coordinate in local coordinates. Generally mentioned in the YOLO label.
        Range: [0, 1]
        Example: 0.5
        
    zoom : Zoom level.
        Range: [0, 20]
        Example: 17
        
    img_center_lat : Latitude of the center of the image.
        Range: approx [-85, 85] (valid range for Web Mercator projection)
        Example: 37.7749
        
    img_center_lon : Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
        
    img_width : Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    img_height : Height of the image in pixels.
        Range: [0, inf]
        Example: 480
        
    Returns
    -------
    (x_webm, y_webm) : X and Y coordinates in Web Mercator projection.
        Range: [0, 2^zoom * 256]
        Example: (1000, 1000)
    """
    
    # Get image center in Web Mercator projection
    image_center_webm_x, image_center_webm_y = geo_to_webm_pixel(img_center_lat, img_center_lon, zoom)
    
    # get delta_x_c: (0, 1) -> (-img_width/2, img_width/2)
    # get delta_y_c: (0, 1) -> (-img_height/2, img_height/2)
    delta_x = x * img_width - img_width/2    
    delta_y = y * img_height - img_height/2
    
    # Get bbox center in Web Mercator projection
    bbox_center_webm_x = image_center_webm_x + delta_x
    bbox_center_webm_y = image_center_webm_y + delta_y
    
    # Convert bbox center to geographic coordinates
    bbox_geo = webm_pixel_to_geo(bbox_center_webm_x, bbox_center_webm_y, zoom)
    bbox_geo = np.array(bbox_geo).T
    
    return bbox_geo


@jaxtyped(typechecker=beartype)
def geo_to_webm_pixel(lat:Union[float, Float[ndarray, "n"]], lon:Union[float, Float[ndarray, "n"]], zoom:int) -> Tuple[Union[float, Float[ndarray, "n"]], Union[float, Float[ndarray, "n"]]]:
    """
    Convert latitude and longitude to Web Mercator projection at a given zoom level.
    
    Parameters
    ----------
    lat : Latitude in decimal degrees.
        Range: approximately [-85, 85]
        Example: 37.7749
        Beyond the specified range, the projection becomes distorted.
        
    lon : Longitude in decimal degrees.
        Range: [-180, 180]
        Example: -122.4194
        
    zoom : Zoom level.
        Range: [0, 20]
        Example: 17
        
    Returns
    -------
    x : X-coordinate in Web Mercator projection.
        Range: [0, 2^zoom * 128]
        Example: 1000

    y : Y-coordinate in Web Mercator projection.
        Range: [0, 2^zoom * 128]
        Example: 1000
    """
    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Project latitude and longitude to Web Mercator
    x = lon_rad + np.pi
    y = np.pi - np.log(np.tan(np.pi/4 + lat_rad/2))
    
    if np.any(y < 0):
        warnings.warn(f"y-coordinate is negative. Latitude='{lat}' might be beyond the valid range of laitude for Web Mercator projection (approx [-85, 85]).")
    elif np.any(y > 2*np.pi):
        warnings.warn(f"y-coordinate is greater than 256*2^zoom. Latitude='{lat}' might be beyond the valid range of latitude for Web Mercator projection (approx [-85, 85]).")
        
    if np.any(x < 0):
        warnings.warn(f"x-coordinate is negative. Longitude='{lon}' might be beyond the valid range of longitude for Web Mercator projection ([-180, 180]).")
    elif np.any(x > 2*np.pi):
        warnings.warn(f"x-coordinate is greater than 256*2^zoom. Longitude='{lon}' might be beyond the valid range of longitude for Web Mercator projection ([-180, 180]).")
    
    # Scale Web Mercator to zoom level
    x = (128/np.pi)*(2**zoom) * x
    y = (128/np.pi)*(2**zoom) * y
    
    return x, y

def webm_pixel_to_geo(x:float, y:float, zoom:int) -> Tuple[Union[float, Float[ndarray, "n"]], Union[float, Float[ndarray, "n"]]]:
    """
    Convert Web Mercator projection to latitude and longitude at a given zoom level.
    
    Parameters
    ----------
    x : X-coordinate in Web Mercator projection.
        Range: [0, 2^zoom * 256]
        Example: 1000

    y : Y-coordinate in Web Mercator projection.
        Range: [0, 2^zoom * 256]
        Example: 1000
        
    zoom : Zoom level.
        Range: [0, 20]
        Example: 17
        
    Returns
    -------
    lat : Latitude in decimal degrees.
        Range: approximately [-85, 85]
        Example: 37.7749
        
    lon : Longitude in decimal degrees.
        Range: [-180, 180]
        Example: -122.4194
    """
    # Scale Web Mercator to radians
    x_rad = x / (128/np.pi) / (2**zoom)
    y_rad = y / (128/np.pi) / (2**zoom)
    
    if np.any(x_rad<0):
        warnings.warn(f"x-coordinate is negative. x='{x}' might be beyond the valid range of x-coordinate for Web Mercator projection ([0, 2^zoom * 256]).")
    elif np.any(x_rad>2*np.pi):
        warnings.warn(f"x-coordinate is greater than 2*pi. x='{x}' might be beyond the valid range of x-coordinate for Web Mercator projection ([0, 2^zoom * 256]).")
        
    if np.any(y_rad<0):
        warnings.warn(f"y-coordinate is negative. y='{y}' might be beyond the valid range of y-coordinate for Web Mercator projection ([0, 2^zoom * 256]).")
    elif np.any(y_rad>2*np.pi):
        warnings.warn(f"y-coordinate is greater than 2*pi. y='{y}' might be beyond the valid range of y-coordinate for Web Mercator projection ([0, 2^zoom * 256]).")
    
    # Inverse project Web Mercator to latitude and longitude
    lon_rad = x_rad - np.pi
    lat_rad = 2*np.arctan(np.exp(np.pi - y_rad)) - np.pi/2
    
    # Convert latitude and longitude to degrees
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)
    
    return lat, lon


def obb_to_aa(yolo_label: Union[str, Float[ndarray, "n 9"], Float[ndarray, "n 10"]]) -> Union[str, Float[ndarray, "n 5"], Float[ndarray, "n 6"]]:
    """
    Convert YOLO OBB labels with format [class_id, x1, y1, x2, y2, x3, y3, x4, y4] to YOLO axis-aligned format [class_id, x_c, y_c, width, height].
    
    Parameters
    ----------
    yolo_label: YOLO label (or path) in OBB format.
    
    Returns
    ----------
    yolo_label: YOLO label in axis-aligned format.
    """
    
    if isinstance(yolo_label, str):
        yolo_label = np.loadtxt(yolo_label, ndmin=2)
        return obb_to_aa(yolo_label)  # to trigger type/shape checking
    
    # Split the label into various components
    class_id = yolo_label[:, 0:1]
    confidence_scores = yolo_label[:, 9:] # will be (n, 0) array if confidence scores are not present
    xyxyxyxy = yolo_label[:, 1:9]
    
    # Get the x and y coordinates
    x = xyxyxyxy[:, ::2]
    y = xyxyxyxy[:, 1::2]
    
    # Find the bounds
    x_max = np.max(x, axis=1)
    x_min = np.min(x, axis=1)
    y_max = np.max(y, axis=1)
    y_min = np.min(y, axis=1)
    
    # Convert to axis-aligned format
    x_c = (x_max + x_min) / 2
    y_c = (y_max + y_min) / 2
    width = x_max - x_min
    height = y_max - y_min
    xywh = np.stack([x_c, y_c, width, height], axis=1)
    
    # Concatenate the class_id and confidence scores
    yolo_label = np.concatenate([class_id, xywh, confidence_scores], axis=1)
    
    return yolo_label