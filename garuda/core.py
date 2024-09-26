import cv2
import pyproj
import geojson
import numpy as np
from numpy import ndarray

import planetary_computer as pc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from shapely.geometry import box, Polygon
import xarray as xr
import rioxarray
from os.path import basename


from beartype import beartype
from beartype.typing import Union, Sequence, Optional, Tuple, Literal
from jaxtyping import Float, Int, jaxtyped
import warnings
from einops import rearrange
from geopy import distance
from copy import deepcopy
from geemap import geemap
from hashlib import sha256

############################################################
# Common utils
############################################################

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
    
    assert path.endswith(".txt"), "Path should be a .txt file."
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
def local_to_geo(x: float, y: float, zoom: int, img_center_lat: float, img_center_lon: float, image_width: int, image_height: int) -> Tuple[float, float]:
    """
    Convert local pixel coordinates to geographic coordinates.
        
    Parameters
    ----------
    x: Normalized X-coordinate in local pixel coordinates.
        Range: [0, 1]
        Example: 0.5
        
    y: Y-coordinate in local pixel coordinates.
        Range: [0, 1]
        Example: 0.3
        
    zoom: Zoom level.
        Range: [0, 20]
        
    img_center_lat: Latitude of the center of the image.
        Range: [-90, 90]
        Example: 37.7749
        
    img_center_lon: Longitude of the center of the image.
        Range: [-180, 180]
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
    image_center_webm_x, image_center_webm_y = geo_to_webm_pixel(img_center_lat, img_center_lon, zoom)
    
    delta_x = x*image_width - image_width/2  # (4,)
    delta_y = y*image_height - image_height/2  # (4,)
    
    # Get bbox center in Web Mercator projection
    x = image_center_webm_x + delta_x  # (4,) = () + (4,)
    y = image_center_webm_y + delta_y  # (4,) = () + (4,)
    
    # Convert bbox center to geographic coordinates
    lat, lon = webm_pixel_to_geo(x, y, zoom)
    
    return lat, lon

@jaxtyped(typechecker=beartype)
def geo_to_local(lat: float, lon: float, zoom: int, img_center_lat: float, img_center_lon: float, image_width: int, image_height: int) -> Tuple[float, float]:
    """
    Convert geographic coordinates to local pixel coordinates.
        
    Parameters
    ----------
    lat: Latitude in decimal degrees.
        Range: [-85, 85] for Web Mercator projection
        Example: 37.7749
        
    lon: Longitude in decimal degrees.
        Range: [-180, 180]
        Example: -122.4194
        
    zoom: Zoom level.
        Range: [0, 20]
        
    img_center_lat: Latitude of the center of the image.
        Range: [-90, 90]
        Example: 37.7749
        
    img_center_lon: Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
        
    image_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    image_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
        
    Returns
    -------
    x: X-coordinate in local pixel coordinates.
        Range: [0, img_width]
        Example: 100
        
    y: Y-coordinate in local pixel coordinates.
        Range: [0, img_height]
        Example: 100
    """
    # Get image center in Web Mercator projection
    image_center_webm_x, image_center_webm_y = geo_to_webm_pixel(img_center_lat, img_center_lon, zoom)
    
    # Convert a point to Web Mercator projection
    x, y = geo_to_webm_pixel(lat, lon, zoom)
    
    # Get delta from image center
    delta_x = x - image_center_webm_x
    delta_y = y - image_center_webm_y
    
    # Get local pixel coordinates
    x = image_width/2 + delta_x
    y = image_height/2 + delta_y
    
    # Normalize
    x = x / image_width
    y = y / image_height
    
    return x, y

@jaxtyped(typechecker=beartype)
def xyxyxyxy2xywhr(xyxyxyxy: Int[ndarray, "8"]) -> Float[ndarray, "5"]:
    """
    Convert Oriented Bounding Boxes (OBB) from [x1, y1, x2, y2, x3, y3, x4, y4] format to [x_c, y_c, w, h, r] format. `r` will be returned in radians.
    Modified from `xyxyxyxy2xywhr` function in Ultralytics library.

    Args:
        xyxyxyxy: Oriented Bounding Boxes in [x1, y1, x2, y2, x3, y3, x4, y4] format.

    Returns:
        xywhr: Oriented Bounding Boxes in [x_c, y_c, w, h, r] format.
    """
    
    points = rearrange(xyxyxyxy, "8", "4 2")
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    rbox = [cx, cy, w, h, angle / 180 * np.pi]
    return np.asarray(rbox)


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