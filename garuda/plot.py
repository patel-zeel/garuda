import numpy as np
from numpy import ndarray
from matplotlib.axes import Axes
from jaxtyping import Float, jaxtyped
from beartype import beartype

from garuda.ops import geo_to_webm_pixel, webm_pixel_to_geo

@jaxtyped(typechecker=beartype)
def plot_webm_pixel_to_geo(img: ndarray, img_center_lat: float, img_center_lon: float, zoom: int, ax:Axes) -> Axes:
    """
    Plot a satellite image with Web Mercator projection on Geo-coordinates.
    
    Parameters
    ----------
    img : Image data in the format of a numpy array.
        Example: np.random.rand(512, 512, 3)
        
    img_center_lat : Latitude of the center of the image.
        Range: approx [-85, 85] (valid range for Web Mercator projection)
        Example: 37.7749
        
    img_center_lon : Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
        
    zoom : Zoom level of the map.
        Range: [0, 20]
        Example: 17
        
    ax : Matplotlib axis to plot the image.
        
    Returns
    -------
    ax : Matplotlib axis with the plotted image.
    """
    
    img_center_webm_x, img_center_webm_y = geo_to_webm_pixel(img_center_lat, img_center_lon, zoom)
    
    x_range = np.arange(int(img_center_webm_x) - img.shape[0]//2, int(img_center_webm_x) + img.shape[0]//2)
    y_range = np.arange(int(img_center_webm_y) - img.shape[1]//2, int(img_center_webm_y) + img.shape[1]//2)

    X, Y = np.meshgrid(x_range, y_range)

    Lat, Lon = webm_pixel_to_geo(X, Y, zoom)

    ax.pcolormesh(Lon, Lat, img)
    ax.set_aspect('equal')
    return ax