import numpy as np
import cv2
from numpy import ndarray
import pyproj
from beartype.typing import Tuple, Union
import warnings
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int

import planetary_computer as pc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from shapely.geometry import box
import xarray as xr
import rioxarray

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


def label_studio_csv_to_obb(x1, y1, width, height, rotation, label, label_map) -> Float[ndarray, "9"]:
    """
    Convert from Label studio CSV (x1, y1, w, h, r) to YOLO OBB (x1, y1, x2, y2, x3, y3, x4, y4) format
    
    Parameters
    ----------
    
    x1: x-cordinate of one corner of the box, range(0, 100)
    y1: y-cordinate of one corner of the box, range(0, 100)
    w: width of the box, range(0, 100)
    h: height of the box, range(0, 100)
    r: rotation angle of the box in degrees
    label: label of the box
    label_map: dictionary mapping label to class_id
    
    Returns
    ----------
    yolo_label: YOLO OBB label in the format [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    """
        
    rotation_rad = np.radians(rotation)
    
    cos_rot = np.cos(rotation_rad)
    sin_rot = np.sin(rotation_rad)
    
    x_1 = x1 + width * cos_rot - height * sin_rot
    y_1 = y1 + width * sin_rot + height * cos_rot
    x_2 = x1 + width * cos_rot
    y_2 = y1 + width * sin_rot
    x_3 = x1
    y_3 = y1
    x_4 = x1 - height * sin_rot
    y_4 = y1 + height * cos_rot
    xyxyxyxy = np.array([x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4])
    
    # Normalize to range(0, 1)
    xyxyxyxy = xyxyxyxy / 100
    
    label_id = np.array([label_map[label]])
    yolo_label = np.concatenate([label_id, xyxyxyxy])
    return yolo_label

@jaxtyped(typechecker=beartype)
def get_sentinel2_visual(lat_c: float, lon_c: float, img_height: int, img_width: int, time_of_interest: str, max_cloud_cover: float, max_items: int = 10) -> xr.DataArray:
    """
    Get Sentinel-2 image as a PNG file from Microsoft Planetary Computer API.
    
    Image is centered at the given latitude and longitude with the given width and height.
    
    TODO: Figure out how to allow more bands. Different resolution bands can not be merged directly.
    
    Parameters
    ----------
    lat_c: Latitude of the center of the image.
        Range: [-90, 90]
        Example: 37.7749
    
    lon_c: Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
    
    img_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
    
    img_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    time_of_interest: Time of interest in the format "start_date/end_date".
        Example: "2021-01-01/2021-01-31"
        
    max_cloud_cover: Maximum cloud cover percentage.
        Range: [0, 100]
        Example: 10
        
    max_items: Maximum number of items to return from the API. Least cloud cover item will be used to crop and return the image.
        Range: [1, inf]
        Example: 10
        
    Returns
    -------
    tif: GeoTIFF file of the Sentinel-2 image
        Metadata: 
            - CRS: EPSG:4326
            - Timestamp: Timestamp of the image
            - href: URL of the image
    """
    
    polygon = box(lon_c - 0.01, lat_c - 0.01, lon_c + 0.01, lat_c + 0.01)
    
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=polygon,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=max_items,
    )
    
    items = search.item_collection()
    sorted_items = sorted(items, key=lambda x: eo.ext(x).cloud_cover)
    
    def get_least_cloudy_raster(sorted_items):
        if len(sorted_items) == 0:
            raise ValueError("Search returned no valid items. Try with different parameters (e.g. expand `time_of_interest`, increase `max_cloud_cover`, increase `max_items`).")
        least_cloud_cover_item = sorted_items[0]
        href = least_cloud_cover_item.assets["visual"].href
        visual_raster = rioxarray.open_rasterio(pc.sign(href))
        
        inverse_transform = pyproj.Transformer.from_crs("EPSG:4326", visual_raster.rio.crs)
        x, y = inverse_transform.transform(lat_c, lon_c)

        x_idx = np.abs(visual_raster.x - x).argmin().item()
        y_idx = np.abs(visual_raster.y - y).argmin().item()
        
        x_exact = int(visual_raster.x[x_idx].item())
        y_exact = int(visual_raster.y[y_idx].item())

        cropped_raster = visual_raster.isel(x=slice(x_idx-img_width//2, x_idx+img_width//2), y=slice(y_idx-img_height//2, y_idx+img_height//2))
        
        try:
            assert cropped_raster.shape == (3, img_height, img_width)
        except AssertionError as e:
            sorted_items.pop(0)
            return get_least_cloudy_raster(sorted_items)
        
        # add timestamp
        cropped_raster.attrs["timestamp"] = least_cloud_cover_item.datetime
        # add center coordinates
        cropped_raster.attrs["x_c"] = x_exact
        cropped_raster.attrs["y_c"] = y_exact
        # add all hrefs
        for key, val in least_cloud_cover_item.assets.items():
            cropped_raster.attrs[key] = val.href
        
        return cropped_raster
    
    return get_least_cloudy_raster(sorted_items)

@jaxtyped(typechecker=beartype)
def xyxyxyxy2xywhr(xyxyxyxy: Float[ndarray, "n 8"]) -> Float[ndarray, "n 5"]:
    """
    Convert Oriented Bounding Boxes (OBB) from [x1, y1, x2, y2, x3, y3, x4, y4] format to [x_c, y_c, w, h, r] format. `r` will be returned in radians.
    Modified from `xyxyxyxy2xywhr` function in Ultralytics library.

    Args:
        xyxyxyxy: Oriented Bounding Boxes in [x1, y1, x2, y2, x3, y3, x4, y4] format.

    Returns:
        xywhr: Oriented Bounding Boxes in [x_c, y_c, w, h, r] format.
    """
    
    points = xyxyxyxy.reshape(len(xyxyxyxy), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return np.asarray(rboxes)

@jaxtyped(typechecker=beartype)
def xywhr2xyxyxyxy(xywhr: Float[ndarray, "n 5"]) -> Float[ndarray, "n 8"]:
    """
    Convert Oriented Bounding Boxes (OBB) from [x_c, y_c, w, h, r] format to [x1, y1, x2, y2, x3, y3, x4, y4] format. `r` should be in radians.

    Args:
        xywhr: Oriented Bounding Boxes in [x_c, y_c, w, h, r] format.

    Returns:
        xyxyxyxy: Oriented Bounding Boxes in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    """

    ctr = xywhr[:, :2]
    w, h, angle = (xywhr[:, i : i + 1] for i in range(2, 5))
    cos_value, sin_value = np.cos(angle), np.sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, -1)
    vec2 = np.concatenate(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.concatenate([pt1, pt2, pt3, pt4], axis=1)

@jaxtyped(typechecker=beartype)
def _get_covariance_matrix(boxes: Float[ndarray, "n 5"]):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes: A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    
    gbbs = np.concatenate((boxes[:, 2:4] ** 2 / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = cos ** 2
    sin2 = sin ** 2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
    
@jaxtyped(typechecker=beartype)
def obb_iou(true_obb: Float[ndarray, "n 8"], pred_obb: Float[ndarray, "m 8"], eps=1e-7) -> Float[ndarray, "n m"]:
    """
    Probalistic IOU for Oriented Bounding Boxes (OBB).
    Inspired from `probiou` function in Ultralytics library.
    
    Parameters
    ----------
    true_obb: True OBB in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    pred_obb: Predicted OBB in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    eps: Small value to avoid division by zero.
        
    Returns
    -------
    iou: Intersection over Union (IOU) matrix of the two OBBs in [0, 1] range.
    """
    
    xywhr_1 = xyxyxyxy2xywhr(true_obb)
    xywhr_2 = xyxyxyxy2xywhr(pred_obb)
    x1, y1 = xywhr_1[:, 0:1], xywhr_1[:, 1:2]
    x2, y2 = xywhr_2[:, 0:1], xywhr_2[:, 1:2]
    a1, b1, c1 = _get_covariance_matrix(xywhr_1)
    a2, b2, c2 = _get_covariance_matrix(xywhr_2)
    
    t1 = (
        ((a1 + a2.T) * (y1 - y2.T)**2 + (b1 + b2.T) * (x1 - x2.T)**2) / ((a1 + a2.T) * (b1 + b2.T) - (c1 + c2.T)**2 + eps)
    ) * 0.25
    t2 = (((c1 + c2.T) * (x2.T - x1) * (y1 - y2.T)) / ((a1 + a2.T) * (b1 + b2.T) - (c1 + c2.T)**2 + eps)) * 0.5
    t3 = np.log(
        ((a1 + a2.T) * (b1 + b2.T) - (c1 + c2.T)**2)
        / (4 * (np.clip(a1 * b1 - c1**2, 0, np.inf) * np.clip(a2.T * b2.T - c2.T**2, 0, np.inf))**0.5 + eps)
        + eps
    ) * 0.5
    
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = (1.0 - np.exp(-bd) + eps) ** 0.5
    iou = 1 - hd
    
    return iou