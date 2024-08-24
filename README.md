<!-- align h1 to center -->
<h1 align="center">
    Garuda
</h1>

<p align="center">
  <img src="logo/garuda_profile_full1.png" width="100%">
</p>
<p align="center">
  A research-oriented computer vision library for satellite imagery.
</p>

[![Coverage Status](https://coveralls.io/repos/github/patel-zeel/garuda/badge.svg?branch=main)](https://coveralls.io/github/patel-zeel/garuda?branch=main)

## Installation

Stable version:
```bash
pip install garuda
```

Latest version:
```bash
pip install git+https://github.com/patel-zeel/garuda
```

## Terminology

| Term                             | Description                                                                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local co-ordinates               | (x, y) where x is the column number and y is the row number. Origin is at the top-left corner.                                                            |
| Web Mercator (webm) co-ordinates | (x, y) pixel co-ordinates as described on [Google Maps Developer Documentation](https://developers.google.com/maps/documentation/javascript/coordinates). |
| Geo co-ordinates                 | (latitude, longitude) as genereally used in GPS systems.                                                                                                  |

## Usage

See the [examples](examples) directory for more details.

## Functionality

### Operations

Convert Ultralytics format of YOLO oriented bounding box to YOLO axis aligned bounding box.

```python
from garuda.ops import obb_to_aa
aa_label = obb_to_aa(obb_label)
```

Convert local image pixel coordinates to geo coordinates (latitude, longitude).

```python
from garuda.ops import local_to_geo
geo_coords = local_to_geo(img_x, img_y, zoom, img_center_lat, img_center_lon, img_width, img_height)
```

Convert geo coordinates (latitude, longitude) to global image pixel coordinates in Web Mercator projection at a given zoom level.

```python
from garuda.ops import geo_to_webm_pixel
webm_x, webm_y = geo_to_webm_pixel(lat, lon, zoom)
```

Convert global image pixel coordinates in Web Mercator projection to geo coordinates (latitude, longitude) at a given zoom level.

```python
from garuda.ops import webm_pixel_to_geo
lat, lon = webm_pixel_to_geo(x, y, zoom)
```

### Object Detection in Satellite Imagery

Convert center of a YOLO axis-aligned or oriented bounding box to geo coordinates (latitude, longitude).

```python
from garuda.od import yolo_aa_to_geo # for axis aligned bounding box
from garuda.od import yolo_obb_to_geo # for oriented bounding box
geo_coords = yolo_aa_to_geo(yolo_aa_label, zoom, img_center_lat, img_center_lon, img_width, img_height)
# OR
geo_coords = yolo_obb_to_geo(yolo_obb_label, zoom, img_center_lat, img_center_lon, img_width, img_height)
```

### Visualization

Plot a satellite image with correct geo-coordinates on the x-axis and y-axis.

```python
from garuda.plot import plot_webm_pixel_to_geo
import matplotlib.pyplot as plt
from PIL import Image

img = plt.imread('path/to/image')
# OR
# img = Image.open('path/to/image')

fig, ax = plt.subplots()
ax = plot_webm_pixel_to_geo(img, img_center_lat, img_center_lon, zoom, ax)
plt.show()
```