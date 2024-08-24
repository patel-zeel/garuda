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

| Term                                   | Description                                                                                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local image co-ordinates               | (x, y) where x is the column number and y is the row number of an image. Origin is at the top-left corner.                                                       |
| Web Mercator (webm) pixel co-ordinates | (x, y) pixel co-ordinates as described on [Google Maps Developer Documentation](https://developers.google.com/maps/documentation/javascript/coordinates).        |
| Geo co-ordinates                       | (latitude, longitude) as genereally used in GPS systems.                                                                                                         |
| Oriented Bounding Box (OBB)            | A rectangular bounding box that is not aligned with the x and y axes. In Ultralytics YOLO format, it is defined as `[label_id, x1, y1, x2, y2, x3, y3, x4, y4]`. |
| Axis-aligned (AA) bounding box         | A rectangular bounding box that is aligned with the x and y axes. In Ultralytics YOLO format, it is defined as `[label_id, x_center, y_center, width, height]`.  |

## Usage

See the [examples](examples) directory for more details.

## Functionality

### Operations

Convert Ultralytics format of YOLO oriented bounding box to YOLO axis aligned bounding box.

```python
import numpy as np
from garuda.ops import obb_to_aa
obb_label_path = "data/labels/obb/22.32,87.93.txt"
aa_label = obb_to_aa(obb_label_path)
# OR
obb_label = np.loadtxt(obb_label_path, ndmin=2)
aa_label = obb_to_aa(obb_label)
```

Convert local image pixel coordinates to geo coordinates (latitude, longitude).

```python
from garuda.ops import local_to_geo
img_x, img_y = 100, 100
zoom = 17
img_center_lat, img_center_lon = 22.32, 87.93
img_width, img_height = 1120, 1120
geo_coords = local_to_geo(img_x, img_y, zoom, img_center_lat, img_center_lon, img_width, img_height)
```

Convert geo coordinates (latitude, longitude) to global image pixel coordinates in Web Mercator projection at a given zoom level.

```python
from garuda.ops import geo_to_webm_pixel
lat, lon = 22.32, 87.93
zoom = 17
webm_x, webm_y = geo_to_webm_pixel(lat, lon, zoom)
```

Convert global image pixel coordinates in Web Mercator projection to geo coordinates (latitude, longitude) at a given zoom level.

```python
from garuda.ops import webm_pixel_to_geo
x, y = 100, 100
lat, lon = webm_pixel_to_geo(x, y, zoom)
```

### Object Detection in Satellite Imagery

Convert center of a YOLO axis-aligned or oriented bounding box to geo coordinates (latitude, longitude).

```python
from garuda.od import yolo_aa_to_geo # for axis aligned bounding box
from garuda.od import yolo_obb_to_geo # for oriented bounding box
yolo_aa_label = "data/labels/aa/22.32,87.93.txt"
yolo_obb_label = "data/labels/obb/22.32,87.93.txt"
zoom = 17
img_center_lat, img_center_lon = 22.32, 87.93
img_width, img_height = 1120, 1120

# For axis aligned bounding box
geo_coords = yolo_aa_to_geo(yolo_aa_label, zoom, img_center_lat, img_center_lon, img_width, img_height)

# For oriented bounding box
geo_coords = yolo_obb_to_geo(yolo_obb_label, zoom, img_center_lat, img_center_lon, img_width, img_height)
```

Convert label studio "CSV" bounding boxes to YOLO format.
```python
import pandas as pd
from garuda.od import add_obb_to_label_studio_df
df = pd.read_csv("data/raw/22.32,87.93.csv")
label_map = {"FCBK": 0, "Zigzag": 1, "ZIGZAG": 1}
df = add_obb_to_label_studio_df(df, label_map) # obb added to the dataframe in "obb" column
print(df['obb'].iloc[0].shape)
# (9, 9) # (n, d): n=9 brick kiln bounding boxes with d=9 values each in [label_id, x1, y1, x2, y2, x3, y3, x4, y4] format
```

### Bulk Operations
Please be careful while using the bulk operation functions as they may lead to catastrophic data loss by overwriting some files. To ensure minimum damage, currently we follow these rules:
* No new directory will be created by the functions. All directories must exist before calling the function.
* Every new file written must not exist before calling the function.

#### Label Studio CSV to YOLO Labels
Write YOLO labels in a directory by image names (detected from label-studio CSV file).

```python
import os
import pandas as pd
import tempfile
from garuda.bulk_ops import write_obb_labels_from_label_studio_csv
df = pd.read_csv("data/raw/22.32,87.93.csv")
with tempfile.TemporaryDirectory() as save_dir:
  label_map = {"FCBK": 0, "Zigzag": 1, "ZIGZAG": 1}
  write_obb_labels_from_label_studio_csv(save_dir, df, label_map)
print(f"Labels written to {save_dir}")
```

#### OBB Labels to AA Labels
Read all OBB labels from a source directory and write them in AA format to a destination directory.

```python
from garuda.bulk_ops import write_aa_labels_from_obb
import tempfile
load_dir = "data/labels/obb"
with tempfile.TemporaryDirectory() as save_dir:
  write_aa_labels_from_obb(load_dir, save_dir)
print(f"Labels written to {save_dir}")
```

### Visualization

Plot a satellite image with correct geo-coordinates on the x-axis and y-axis.

```python
from garuda.plot import plot_webm_pixel_to_geo
import matplotlib.pyplot as plt

img = plt.imread('data/images/22.32,87.93.png')

fig, ax = plt.subplots()
ax = plot_webm_pixel_to_geo(img, img_center_lat, img_center_lon, zoom, ax)
```