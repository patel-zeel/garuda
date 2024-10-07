import io
import os
from os.path import join, basename
import cv2
import pyproj
import geojson
import numpy as np
from numpy import ndarray
from glob import glob

import planetary_computer as pc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from shapely.geometry import box, Polygon
import xarray as xr
import rioxarray
from os.path import basename

import logging
from beartype import beartype
from beartype.typing import Union, Sequence, Optional, Tuple, Literal
from jaxtyping import Float, Int, jaxtyped
import warnings
from einops import rearrange
from geopy import distance
from copy import deepcopy
from geemap import geemap
from hashlib import sha256
import utm

from tqdm.auto import tqdm as _tqdm

############################################################
# Base classes
############################################################

class AutoTypeChecker:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr, value in cls.__dict__.items():
            if isinstance(value, staticmethod):
                original_func = value.__func__
                setattr(cls, attr, staticmethod(jaxtyped(typechecker=beartype)(original_func)))
            elif isinstance(value, classmethod):
                original_func = value.__func__
                setattr(cls, attr, classmethod(jaxtyped(typechecker=beartype)(original_func)))
            elif callable(value):
                setattr(cls, attr, jaxtyped(typechecker=beartype)(value))


############################################################
# Common utils
############################################################
log_format = 'GARUDA %(levelname)-9s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format) # 9s is the length of the longest log level name
logger = logging.getLogger()

def tqdm(*args, **kwargs):
    if os.environ.get('GARUDA_DISABLE_TQDM') == 'True':
        return _tqdm(*args, disable=True, **kwargs)
    return _tqdm(*args, **kwargs)

@jaxtyped(typechecker=beartype)
def get_latlon_from_gms_path(path: str) -> dict:
    """
    Get latitude and longitude from Google Maps Static Image Label path.
    
    Parameters
    ----------
    path: Path of the Google Maps Static Image Label.
        Example: "37.7749,-122.4194.txt"
        
    Returns
    -------
    dict: Dictionary containing latitude and longitude in string and float format.
        Example: {"str": ("37.7749", "-122.4194"), "float": (37.7749, -122.4194)}
    """
    
    # remove extension from path. extension can be anything like .txt, .png, .jpg, etc.
    
    path = os.path.splitext(path)[0]
    path = path.replace("%2C", ",")
    base_name = basename(path)
    base_name = base_name.replace(".txt", "")
    lat_str, lon_str = base_name.split(",")
    lat, lon = float(lat_str), float(lon_str)
    return {"str": (lat_str, lon_str), "float": (lat, lon)}

@jaxtyped(typechecker=beartype)
def get_sentinel2_visual(lat_c: float, lon_c: float, img_height: int, img_width: int, time_of_interest: str, max_cloud_cover: float, max_items: int = 10, nodata_window_size: int = 2) -> xr.DataArray:
    """
    Get Sentinel-2 image as a xarray file from Microsoft Planetary Computer API.
    
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
        
    nodata_window_size: Size of the window to check for nodata values.
        Range: [0, inf]
        Example: 2
        We will check if a (nodata_window_size x nodata_window_size) window has all zeros. If yes, we will discard that item and move to the next one.
        
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
            print(f"Shape of the raster is not as expected. Shape: {cropped_raster.shape}")
            sorted_items.pop(0)
            return get_least_cloudy_raster(sorted_items)
        
        try:
            np_img = cropped_raster.values
            np_img = rearrange(np_img, "c h w -> h w c")
            # pad image to make it even sized
            # padded_img = np.pad(np_img, ((0, 1), (0, 1), (0, 0)), mode='edge')
            mask = (np_img == 0).all(axis=(2))
            n = nodata_window_size
            result = (mask.reshape(np_img.shape[0]//n, n, np_img.shape[1]//n, n).all(axis=(1, 3)))
            assert not result.any()
        except AssertionError as e:
            print("Nodata values found in the image. Discarding this item and moving to the next one.")
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
def local_to_geo(epsg: int, x: float, y: float, zoom: int | None, img_center_x: float, img_center_y: float, image_width: int, image_height: int, resolution: int | None) -> Tuple[float, float]:
    """
    Convert local pixel coordinates to geographic coordinates.
        
    Parameters
    ----------
    epsg: EPSG code of the projection.
        Only '3857' (web mercator) and '32XXX' (utm) are supported.
    
    x: Normalized X-coordinate in local pixel coordinates.
        Range: [0, 1]
        Example: 0.5
        
    y: Y-coordinate in local pixel coordinates.
        Range: [0, 1]
        Example: 0.3
        
    zoom: Zoom level.
        Range: [0, 20]
        
    img_center_x: Latitude of the center of the image if epsg is '3857'. UTM x-coordinate of the center of the image if epsg is '32XXX'.
        Example: 37.7749
        
    img_center_y: Longitude of the center of the image if epsg is '3857'. UTM y-coordinate of the center of the image if epsg is '32XXX'.
        Example: -122.4194
        
    image_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    image_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
        
    Returns
    -------
    lat: Latitude in decimal degrees.
        Range: [-85, 85] for Web Mercator projection
        Example: 37.7749
        
    lon: Longitude in decimal degrees.
        Range: [-180, 180]
        Example: -122.4194
    """
    # Get image center in Web Mercator projection
    if epsg == 3857:
        image_center_webm_x, image_center_webm_y = geo_to_webm_pixel(img_center_x, img_center_y, zoom)
    elif str(epsg).startswith("32") and len(str(epsg)) == 5:
        pass # nothing to do
    else:
        raise ValueError("Only '3857' (web mercator) and '32XXX' (utm) are supported.")
    
    # Get local pixel coordinates
    x = x * image_width
    y = y * image_height
    
    # Get delta from image center
    delta_x = x - image_width/2
    delta_y = y - image_height/2
    
    # Get geographic coordinates
    if epsg == 3857:
        x = image_center_webm_x + delta_x
        y = image_center_webm_y + delta_y
        lat, lon = webm_pixel_to_geo(x, y, zoom)
    elif str(epsg).startswith("32") and len(str(epsg)) == 5:
        x = img_center_x + delta_x * resolution
        y = img_center_y - delta_y * resolution
        zone = int(str(epsg)[-2:])
        lat, lon = utm.to_latlon(x, y, zone, northern=True)
    else:
        raise ValueError("Only '3857' (web mercator) and '32XXX' (utm) are supported.")
    
    return lat, lon

@jaxtyped(typechecker=beartype)
def geo_to_utm(lat: float, lon: float, epsg: int):
    """
    Convert latitude and longitude to UTM coordinates.
    
    Parameters
    ----------
    lat: Latitude in decimal degrees.
        Range: [-90, 90]
        Example: 37.7749
        
    lon: Longitude in decimal degrees.
        Range: [-180, 180]
        Example: -122.4194
        
    epsg: EPSG code of the projection.
        Example: 32610
    """
    assert str(epsg).startswith("32") and len(str(epsg)) == 5, "Only UTM projections are supported. Example: 32610"
    geo_to_utm_transform = pyproj.Transformer.from_proj(4326, epsg)
    x, y = geo_to_utm_transform.transform(lat, lon)
    return x, y

@jaxtyped(typechecker=beartype)
def geo_to_local(epsg: int, lat: float, lon: float, zoom: int | None, img_center_x: float, img_center_y: float, image_width: int, image_height: int, resolution: int) -> Tuple[float, float]:
    """
    Convert geographic coordinates to local pixel coordinates.
        
    Parameters
    ----------
    epsg: EPSG code of the projection.
        Only '3857' (web mercator) and '32XXX' (utm) are supported.
    
    lat: Latitude in decimal degrees.
        Range: [-85, 85] for Web Mercator projection
        Example: 37.7749
        
    lon: Longitude in decimal degrees.
        Range: [-180, 180]
        Example: -122.4194
        
    zoom: Zoom level. Applicable only for EPSG '3857'. Provide None for UTM projections.
        Range: [0, 20]
        
    img_center_x: Latitude of the center of the image if epsg is '3857'. UTM x-coordinate of the center of the image if epsg is '32XXX'.
        
    img_center_y: Longitude of the center of the image if epsg is '3857'. UTM y-coordinate of the center of the image if epsg is '32XXX'.
        
    image_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    image_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
        
    resolution: Resolution of the image in meters/pixel. Applicable only for UTM projections. Provide None for Web Mercator projection.
        
    Returns
    -------
    x: Normalized X-coordinate in local pixel coordinates.
        Range: [0, 1]
        Example: 0.1
        
    y: Normalized Y-coordinate in local pixel coordinates.
        Range: [0, 1]
        Example: 0.1
    """
    # Get image center in Web Mercator projection
    if epsg == 3857:
        img_center_x, img_center_y = geo_to_webm_pixel(img_center_x, img_center_y, zoom)
    elif str(epsg).startswith("326") and len(str(epsg)) == 5:
        pass # nothing to do
    else:
        raise ValueError("Only '3857' (web mercator) and '32XXX' (utm) are supported.")
    
    # Convert a point to Web Mercator projection
    if epsg == 3857:
        x, y = geo_to_webm_pixel(lat, lon, zoom)
    elif str(epsg).startswith("326") and len(str(epsg)) == 5:
        x, y = geo_to_utm(lat, lon, epsg)
    else:
        raise ValueError("Only '3857' (web mercator) and '32XXX' (utm) are supported.")
    
    # Get delta from image center
    if epsg == 3857:
        delta_x = x - img_center_x
        delta_y = y - img_center_y
    elif str(epsg).startswith("326") and len(str(epsg)) == 5:
        delta_x = (x - img_center_x) / resolution
        delta_y = (y - img_center_y) / resolution
    else:
        raise ValueError("Only '3857' (web mercator) and '32XXX' (utm) are supported.")
    
    # Get local pixel coordinates
    x = image_width/2 + delta_x
    
    if epsg == 3857:
        y = image_height/2 + delta_y
    elif str(epsg).startswith("326") and len(str(epsg)) == 5:
        y = image_height/2 - delta_y
    
    # Normalize
    x = x / image_width
    y = y / image_height
    
    for name, coord in zip(["x", "y"], [x, y]):
        if coord > 1 and coord <= 1.05:
            warnings.warn(f"{name}-coordinate is slightly greater than the image width. {name}='{coord}'. Perhaps you can work with it.")
        elif coord < 0 and coord >= -0.05:
            warnings.warn(f"{name}-coordinate is slightly less than 0. {name}='{coord}'. Perhaps you can work with it.")
        elif coord > 1.05:
            warnings.warn(f"{name}-coordinate is >5% greater than the image width. {name}='{coord}'. Label is not within the image.")
        elif coord < -0.05:
            warnings.warn(f"{name}-coordinate is <-5% less than 0. {name}='{coord}'. Label is not within the image.")
        else:
            pass
    
    return x, y

@jaxtyped(typechecker=beartype)
def xyxyxyxy2xywhr(xyxyxyxy: Float[ndarray, "4 2"]) -> Float[ndarray, "5"]:
    """
    Convert Oriented Bounding Boxes (OBB) from [x1, y1, x2, y2, x3, y3, x4, y4] format to [x_c, y_c, w, h, r] format. `r` will be returned in radians.
    Modified from `xyxyxyxy2xywhr` function in Ultralytics library.

    Args:
        xyxyxyxy: Oriented Bounding Boxes in [x1, y1, x2, y2, x3, y3, x4, y4] format.

    Returns:
        xywhr: Oriented Bounding Boxes in [x_c, y_c, w, h, r] format.
    """
    
    (cx, cy), (w, h), angle = cv2.minAreaRect(xyxyxyxy)
    rbox = [cx, cy, w, h, np.radians(angle)]
    return np.asarray(rbox)

@jaxtyped(typechecker=beartype)
def xyxyxyxy2xywhr_batch(xyxyxyxy: Float[ndarray, "n 4 2"]) -> Float[ndarray, "n 5"]:
    """
    Convert Oriented Bounding Boxes (OBB) from [x1, y1, x2, y2, x3, y3, x4, y4] format to [x_c, y_c, w, h, r] format. `r` will be returned in radians.
    Modified from `xyxyxyxy2xywhr` function in Ultralytics library.

    Args:
        xyxyxyxy: Oriented Bounding Boxes in [x1, y1, x2, y2, x3, y3, x4, y4] format.

    Returns:
        xywhr: Oriented Bounding Boxes in [x_c, y_c, w, h, r] format.
    """
    if xyxyxyxy.shape[0] == 0:
        return np.zeros((0, 5))
    
    rboxes = []
    for pts in xyxyxyxy:
        rboxes.append(xyxyxyxy2xywhr(pts))
    return np.asarray(rboxes)


@jaxtyped(typechecker=beartype)
def xyxytoxywh(xyxy: Float[ndarray, "4"]) -> Float[ndarray, "4"]:
    """
    Convert bounding boxes from [x1, y1, x2, y2] format to [x_center, y_center, width, height] format.

    Args:
        xyxy: Bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        xywh: Bounding boxes in [x_center, y_center, width, height] format.
    """
    x1, y1, x2, y2 = xyxy.tolist()
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return np.array([x_center, y_center, width, height])


@jaxtyped(typechecker=beartype)
def xywh2xyxy(xywh: Float[ndarray, "4"]) -> Float[ndarray, "4"]:
    """
    Convert bounding boxes from [x_center, y_center, width, height] format to [x1, y1, x2, y2] format.

    Args:
        xywh: Bounding boxes in [x_center, y_center, width, height] format.

    Returns:
        xyxy: Bounding boxes in [x1, y1, x2, y2] format.
    """
    x_center, y_center, width, height = xywh.tolist()
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return np.array([x1, y1, x2, y2])

@jaxtyped(typechecker=beartype)
def xywhr2xyxyxyxy(xywhr: Float[ndarray, "5"]) -> Float[ndarray, "4 2"]:
    """
    Convert Oriented Bounding Boxes (OBB) from [x_c, y_c, w, h, r] format to [x1, y1, x2, y2, x3, y3, x4, y4] format. `r` should be in radians.

    Args:
        xywhr: Oriented Bounding Boxes in [x_c, y_c, w, h, r] format.

    Returns:
        xyxyxyxy: Oriented Bounding Boxes in [x1, y1, x2, y2, x3, y3, x4, y4] format.
    """

    ctr = xywhr[:2]
    w, h, angle = xywhr[2:]
    cos_value, sin_value = np.cos(angle), np.sin(angle)
    vec1 = np.array([w / 2 * cos_value, w / 2 * sin_value])
    vec2 = np.array([-h / 2 * sin_value, h / 2 * cos_value])
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    pts = np.concatenate([pt1, pt2, pt3, pt4])
    return pts.reshape(4, 2)

@jaxtyped(typechecker=beartype)
def geo_to_webm_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
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

@jaxtyped(typechecker=beartype)
def webm_pixel_to_geo(x:float, y:float, zoom:int) -> Tuple[float, float]:
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
def obb_iou(true_obb: Float[ndarray, "n 4 2"], pred_obb: Float[ndarray, "m 4 2"], eps=1e-7) -> Float[ndarray, "n m"]:
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
    
    xywhr_1 = xyxyxyxy2xywhr_batch(true_obb)
    xywhr_2 = xyxyxyxy2xywhr_batch(pred_obb)
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

@jaxtyped(typechecker=beartype)
def obb_iou_shapely(obb1: Float[ndarray, "4 2"], obb2: Float[ndarray, "4 2"]) -> float:
    """
    Compute Intersection over Union (IoU) of two Oriented Bounding Boxes (OBB) using Shapely library.
    
    Args:
        obb1: Oriented Bounding Box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
        obb2: Oriented Bounding Box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
        
    Returns:
        iou: Intersection over Union (IoU) of the two OBBs in [0, 1] range.
    """    
    p1 = Polygon(obb1)
    p2 = Polygon(obb2)
    iou = p1.intersection(p2).area / p1.union(p2).area
    return iou

@jaxtyped(typechecker=beartype)
def obb_iou_shapely_batch(obb1: Float[ndarray, "n 4 2"], obb2: Float[ndarray, "m 4 2"]) -> Float[ndarray, "n m"]:
    """
    Compute Intersection over Union (IoU) of two Oriented Bounding Boxes (OBB) using Shapely library.
    
    Args:
        obb1: Oriented Bounding Box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
        obb2: Oriented Bounding Box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
        
    Returns:
        iou: Intersection over Union (IoU) of the two OBBs in [0, 1] range.
    """
    iou = np.zeros((obb1.shape[0], obb2.shape[0]))
    for i, obb1_ in enumerate(obb1):
        for j, obb2_ in enumerate(obb2):
            iou[i, j] = obb_iou_shapely(obb1_, obb2_)
    return iou



@jaxtyped(typechecker=beartype)
def obb_smaller_box_ioa(obb1: Float[ndarray, "4 2"], obb2: Float[ndarray, "4 2"]) -> float:
    """
    Compute intersection over area of smaller box with the larger box. Here, intersection is intersection between the two boxes and area is area of the smaller box.
    IoA = Intersection / Area of smaller box
    
    Args:
        obb1: Oriented Bounding Box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
        obb2: Oriented Bounding Box in [x1, y1, x2, y2, x3, y3, x4, y4] format.
        
    Returns:
        ioa: Intersection over area of smaller box with the larger box.
    """
    p1 = Polygon(obb1)
    p2 = Polygon(obb2)
    intersection = p1.intersection(p2).area
    area1 = p1.area
    area2 = p2.area
    ioa = intersection / min(area1, area2)
    return ioa


ultralytics_obb_to_aabb_docstring = """
Convert YOLO OBB labels to YOLO axis-aligned format.

Parameters
----------
label: YOLO label (or path) in OBB format.
    OBB formats: 
    - [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    - [class_id, x1, y1, x2, y2, x3, y3, x4, y4, confidence_score]

Returns
----------
label: YOLO label in axis-aligned format: [class_id, x_c, y_c, width, height]
"""

@jaxtyped(typechecker=beartype)
def ultralytics_obb_to_aabb(label: Float[ndarray, "9"] | Float[ndarray, "10"]) -> Float[ndarray, "5"] | Float[ndarray, "6"]:
    f"{ultralytics_obb_to_aabb_docstring}"
    
    # Split the label into various components
    class_id = label[0:1]
    confidence_score = label[9:] # will be shape (0,) array if confidence scores are not present
    xyxyxyxy = label[1:9]
    
    # Get the x and y coordinates
    x = xyxyxyxy[::2]
    y = xyxyxyxy[1::2]
    
    # Convert to axis-aligned format
    x_c = (x.max() + x.min()) / 2
    y_c = (x.max() + y.min()) / 2
    width = x.max() - x.min()
    height = y.max() - y.min()
    xywh = np.array([x_c, y_c, width, height])
    
    # Concatenate the class_id and confidence scores
    label = np.concatenate([class_id, xywh, confidence_score])
    
    return label

@jaxtyped(typechecker=beartype)
def ultralytics_obb_to_aabb_batch(label: Float[ndarray, "n 9"] | Float[ndarray, "n 10"]) -> Float[ndarray, "n 5"] | Float[ndarray, "n 6"]:
    f"{ultralytics_obb_to_aabb_docstring}"
    
    if label.shape[1] == 9:
        aabb_labels = np.zeros((label.shape[0], 5)) * np.nan
    elif label.shape[1] == 10:
        aabb_labels = np.zeros((label.shape[0], 6)) * np.nan
    else:
        raise ValueError("Invalid shape of the label. Expected shape: (n, 9) or (n, 10). This error should never occur if bear-typing is working correctly.")
    
    for idx, obb in enumerate(label):
        aabb_labels[idx, :] = ultralytics_obb_to_aabb(obb)
    return aabb_labels

@jaxtyped(typechecker=beartype)
def ultralytics_obb_to_aabb_batch_save(load_dir: str, save_dir: str) -> str:
    """
    Convert YOLO OBB labels to YOLO axis-aligned format and save them.
    """
    
    save_dir_empty = len(glob(join(save_dir, "*"))) == 0
    if not save_dir_empty:
        raise ValueError(f"Directory '{save_dir}' is not empty. Please provide an empty directory.")
    
    label_paths = glob(join(load_dir, "*.txt"))
    empty_label_paths = []
    for label_path in label_paths:
        obb_label = np.loadtxt(label_path, ndmin=2)
        if obb_label.size == 0:
            empty_label_paths.append(label_path)
            continue
        
        aabb_label = ultralytics_obb_to_aabb_batch(obb_label).tolist()
        
        with open(join(save_dir, basename(label_path)), "w") as f:
            txt_labels = []
            for label in aabb_label:
                label[0] = int(label[0])
                txt_labels.append(" ".join(map(str, label)))
            txt_label = "\n".join(txt_labels)
            f.write(txt_label)
            
    for empty_label_path in empty_label_paths:
        with open(join(save_dir, basename(empty_label_path)), "w") as f:
            f.write("")
            
    return save_dir