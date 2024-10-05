import numpy as np
from numpy import ndarray

from beartype import beartype
from beartype.typing import Sequence, Tuple, Literal
from jaxtyping import Float, Int, jaxtyped
import warnings
from einops import rearrange
from geopy import distance
from copy import deepcopy
from leafmap import leafmap
from hashlib import sha256

from garuda.core import xywh2xyxy, local_to_geo, geo_to_local


# This class enforces type checking using beartype and jaxtyping on all the methods.
class AutoTypeChecker:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr, value in cls.__dict__.items():
            if callable(value):
                setattr(cls, attr, jaxtyped(typechecker=beartype)(value))


class BB(AutoTypeChecker):
    def __init__(
        self,
        geo_box: Float[ndarray, "4 2"] | Float[ndarray, "2 2"],
        class_name: str,
        confidence: float | None,
    ):
        """
        Base class for bounding boxes.

        Parameters
        ----------
        geo_box: ndarray
            Bounding box in geo coordinates.
            For OBB: [[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]]
            For AABB: [[lon1, lat1], [lon2, lat2]]

        class_name: str
            Class name of the bounding box.

        confidence: float
            Confidence of the bounding box. This will be ignored if the child class represents a label.
        """

        if type(self) is BB:
            raise TypeError(
                f"Instantiation of '{self.__class__.__name__}' is not allowed. Use one of the children classes instead."
            )
        self.geo_box = geo_box
        self.class_name = class_name
        self.confidence = confidence
        self.properties = {
            "geo_box": self.geo_box.tolist(),
            "class_name": self.class_name,
            "confidence": self.confidence,
        }

        self.add_additional_properties()

    def add_additional_properties(self):
        self.properties["max_lon"] = self.geo_box[:, 0].max().item()
        self.properties["min_lon"] = self.geo_box[:, 0].min().item()
        self.properties["max_lat"] = self.geo_box[:, 1].max().item()
        self.properties["min_lat"] = self.geo_box[:, 1].min().item()
        self.properties["center_lat"] = (
            self.properties["max_lat"] + self.properties["min_lat"]
        ) / 2
        self.properties["center_lon"] = (
            self.properties["max_lon"] + self.properties["min_lon"]
        ) / 2
        self.properties["width_of_box"] = distance.geodesic(
            (self.properties["max_lat"], self.properties["min_lon"]),
            (self.properties["max_lat"], self.properties["max_lon"]),
        ).meters
        self.properties["height_of_box"] = distance.geodesic(
            (self.properties["min_lat"], self.properties["max_lon"]),
            (self.properties["max_lat"], self.properties["max_lon"]),
        ).meters

        if self.geo_box.shape == (4, 2):  # OBB
            side1_length = distance.geodesic(
                (self.geo_box[0][1], self.geo_box[0][0]),
                (self.geo_box[1][1], self.geo_box[1][0]),
            ).meters
            side2_length = distance.geodesic(
                (self.geo_box[1][1], self.geo_box[1][0]),
                (self.geo_box[2][1], self.geo_box[2][0]),
            ).meters
            self.properties["length_of_object"] = max(side1_length, side2_length)
            self.properties["width_of_object"] = min(side1_length, side2_length)
        elif self.geo_box.shape == (2, 2):  # AABB
            self.properties["length_of_object"] = max(
                self.properties["width_of_box"], self.properties["height_of_box"]
            )
            self.properties["width_of_object"] = min(
                self.properties["width_of_box"], self.properties["height_of_box"]
            )
        else:
            raise ValueError(
                f"Invalid {self.geo_box=}. Box should be either OBB with (4, 2) shape or AABB with (2, 2) shape."
            )

        self.properties["area"] = (
            self.properties["length_of_object"] * self.properties["width_of_object"]
        )

        # unique ID
        self.properties["id"] = sha256(f"{self.properties}".encode()).hexdigest()

    @staticmethod
    def _from_ultralytics(
        label: (
            Float[ndarray, "10"]
            | Float[ndarray, "9"]
            | Float[ndarray, "6"]
            | Float[ndarray, "5"]
        ),
        classes: Sequence,
        image_width: int,
        image_height: int,
    ) -> Tuple[Float[ndarray, "8"] | Int[ndarray, "4"], str, float | None]:
        """
        Common operations to convert Ultralytics format to OBB or AABB.

        Parameters
        ----------
        bb_type: Literal["obb", "aabb"]
            Type of bounding box. "obb" for oriented bounding box and "aabb" for axis-aligned bounding box.

        label: ndarray
            Label in Ultralytics format.
            For OBB: [class_id, x1, y1, x2, y2, x3, y3, x4] or [class_id, x1, y1, x2, y2, x3, y3, x4, confidence]
            For AABB: [class_id, x, y, w, h] or [class_id, x, y, w, h, confidence]

        classes: sequence
            Sequence of class names. We will use this to get the class name from the class_id.

        image_width, image_height: int
            Original image size.

        Returns
        -------
        box: ndarray
            OBB: [x1, y1, x2, y2, x3, y3, x4, y4]
            AABB: [x1, y1, x2, y2]

        class_name: str
            Class name

        confidence: float
            Confidence of the bounding box. 1.0 if confidence is not provided.
        """
        class_name = classes[int(label[0])]
        if len(label) in (5, 6):  # AABB
            box = label[1:5]
            box = xywh2xyxy(box[None, ...]).ravel()
            confidence = label[5] if len(label) == 6 else 1.0
        elif len(label) in (9, 10):  # OBB
            box = label[1:9]
            confidence = label[9] if len(label) == 10 else 1.0

        x = box[::2]
        y = box[1::2]

        # scale to original image size
        # x = x * image_width
        # y = y * image_height

        box = np.stack((x, y)).T.ravel()#.round().astype(int)
        return box, class_name, confidence

    @classmethod
    def from_ultralytics(
        cls,
        epsg: int,
        label: (
            Float[ndarray, "10"]
            | Float[ndarray, "9"]
            | Float[ndarray, "6"]
            | Float[ndarray, "5"]
        ),
        classes: Sequence,
        zoom: int | None,
        image_center_x: float,
        image_center_y: float,
        image_width: int,
        image_height: int,
        resolution: int | None,
    ) -> "BBLabel":
        """
        Load label from Ultralytics format.

        Parameters
        ----------
        label: ndarray
            Label in Ultralytics format.
            For OBB: [class_id, x1, y1, x2, y2, x3, y3, x4] or [class_id, x1, y1, x2, y2, x3, y3, x4, confidence]
            For AABB: [class_id, x, y, w, h] or [class_id, x, y, w, h, confidence]

        classes: sequence
            Sequence of class names. We will use this to get the class name from the class_id.

        image_width, image_height: int
            Original image size.

        zoom: int
            Zoom level of the image.

        image_center_x, image_center_y: float
            Center of the image in projection coordinates defined by epsg.
        """
        box, class_name, confidence = cls._from_ultralytics(
            label, classes, image_width, image_height
        )
        geo_box = cls._local_box_to_geo(epsg, box, zoom, image_center_x, image_center_y, image_width, image_height, resolution)

        instance = cls(geo_box, class_name, confidence)
        instance.properties["epsg"] = epsg
        instance.properties["image_width"] = image_width
        instance.properties["image_height"] = image_height
        instance.properties["image_center_x"] = image_center_x
        instance.properties["image_center_y"] = image_center_y
        instance.properties["zoom"] = zoom
        instance.properties["resolution"] = resolution
        return instance

    @staticmethod
    def _local_box_to_geo(
        epsg: int,
        box: Float[ndarray, "8"] | Float[ndarray, "4"],
        zoom: int | None,
        image_center_x: float,
        image_center_y: float,
        image_width: int,
        image_height: int,
        resolution: int | None,
    ) -> Float[ndarray, "4 2"] | Float[ndarray, "2 2"]:
        x = box[::2].tolist()
        y = box[1::2].tolist()
        if len(box) == 8:  # OBB
            lat1, lon1 = local_to_geo(
                epsg,
                x[0],
                y[0],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            lat2, lon2 = local_to_geo(
                epsg,
                x[1],
                y[1],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,

            )
            lat3, lon3 = local_to_geo(
                epsg,
                x[2],
                y[2],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            lat4, lon4 = local_to_geo(
                epsg,
                x[3],
                y[3],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            return np.array([[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]])
        elif len(box) == 4:  # AABB
            lat1, lon1 = local_to_geo(
                epsg,
                x[0],
                y[0],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
            )
            lat2, lon2 = local_to_geo(
                epsg,
                x[1],
                y[1],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
            )
            return np.array([[lon1, lat1], [lon2, lat2]])
        else:
            raise ValueError(
                f"Invalid {box=}. Box should be either OBB with 8 elements or AABB with 4 elements."
            )

    @staticmethod
    def _geo_box_to_local(
        epsg: int,
        geo_box: Float[ndarray, "4 2"] | Float[ndarray, "2 2"],
        zoom: int,
        image_center_x: float,
        image_center_y: float,
        image_width: int,
        image_height: int,
        resolution: int | None,
    ) -> Float[ndarray, "8"] | Float[ndarray, "4"]:
        lons = geo_box[:, 0]
        lats = geo_box[:, 1]
        if geo_box.shape == (4, 2):
            x1, y1 = geo_to_local(
                epsg,
                lats[0],
                lons[0],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            x2, y2 = geo_to_local(
                epsg,
                lats[1],
                lons[1],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            x3, y3 = geo_to_local(
                epsg,
                lats[2],
                lons[2],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            x4, y4 = geo_to_local(
                epsg,
                lats[3],
                lons[3],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            return np.array([x1, y1, x2, y2, x3, y3, x4, y4])
        elif geo_box.shape == (2, 2):
            x1, y1 = geo_to_local(
                epsg,
                lons[0],
                lats[0],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            x2, y2 = geo_to_local(
                epsg,
                lons[1],
                lats[1],
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )
            return np.array([x1, y1, x2, y2])

    def to_geojson(self, source, task_name):
        if source is None:
            if "source" in self.properties:
                source = self.properties["source"]
            else:
                source = "Drawn|Azure Maps Satellite"
        if task_name is None:
            if "task_name" in self.properties:
                task_name = self.properties["task_name"]
            else:
                task_name = ""
        
        # add source and task_name to properties
        self.properties["source"] = source
        self.properties["task_name"] = task_name

        if self.geo_box.shape == (4, 2):  # OBB
            box = self.geo_box
        elif self.geo_box.shape == (2, 2):  # AABB
            # convert AABB to OBB format
            lon1, lat1 = self.geo_box[0]
            lon2, lat2 = self.geo_box[1]
            box = np.array([[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2]])
        else:
            raise ValueError(
                f"Invalid {self.geo_box=}. Box should be either OBB with (4, 2) shape or AABB with (2, 2) shape."
            )

        points = rearrange(box, "n d -> 1 n d").tolist()
        # append first point to close the polygon
        points[0].append(points[0][0])

        feature = {
            "type": "Feature",
            "properties": self.properties,
            "geometry": {
                "type": "Polygon",
                "coordinates": points,
            },
        }
        return feature

    def to_ultralytics_obb(
        self,
        epsg: int,
        classes: Sequence,
        zoom: int | None,
        image_center_x: float | None,
        image_center_y: float | None,
        image_width: int | None,
        image_height: int | None,
        resolution: int | None,
    ) -> Float[ndarray, "9"] | Float[ndarray, "10"]:
        """
        To Ultralytics oriented bounding box (OBB) format.

        Parameters
        ----------
        epsg: EPSG code of the projection of the image.
            Only '3857' (web mercator) and '32XXX' (utm) are supported.
        
        classes: Sequence of class names. We will use this to get the class_id from the class name.

        zoom: Zoom level of the image. Applicable only for epsg=3857. Provide None for UTM projection.

        image_center_x: Latitude if projection is "webm" else UTM x coordinate.
            
        image_center_y: Longitude if projection is "webm" else UTM y coordinate.

        image_width, image_height: Width and height of the image.
        
        resolution: Resolution of the image. Applicable only for UTM projection. Provide None for web mercator projection.
            
        Returns
        -------
        label: ndarray
            Label/Detection in Ultralytics format.
            Label: [class_id, x1, y1, x2, y2, x3, y3, x4]
            Detection: [class_id, x1, y1, x2, y2, x3, y3, x4, confidence]
        """
        assert isinstance(
            self, (OBBLabel, OBBDetection)
        ), f"this method should not be called on {self.__class__.__name__}. It is only for 'OBBLabel' and 'OBBDetection' instances."
        
        if str(epsg).startswith("32") and len(str(epsg)) == 5:
            zoom = 100 # dummy value

        def auto_set(key):
            nonlocal zoom, image_center_x, image_center_y, image_width, image_height
            if eval(key) is None:
                if key in self.properties:
                    return self.properties[key]
                else:
                    raise ValueError(
                        f"'{key}' is neither provided nor found in the properties."
                    )
            else:
                return eval(key)

        zoom = auto_set("zoom")
        image_width = auto_set("image_width")
        image_height = auto_set("image_height")
        image_center_x = auto_set("image_center_x")
        image_center_y = auto_set("image_center_y")

        class_id = classes.index(self.class_name)

        box = self._geo_box_to_local(
                epsg,
                self.geo_box,
                zoom,
                image_center_x,
                image_center_y,
                image_width,
                image_height,
                resolution,
            )

        label = np.zeros(10) * np.nan
        label[0] = class_id
        label[1:9] = box
        if isinstance(self, OBBLabel):
            # OBBLabel doesn't have confidence attribute
            return label[:9]
        elif isinstance(self, OBBDetection):
            label[9] = self.confidence
            return label
        else:
            raise ValueError(
                f"Invalid instance type: {self.__class__.__name__}. This error should not have occurred because it should have been caught by the assert statement at the beginning of the method."
            )

    def to_ultralytics_aabb(self, classes: Sequence) -> Float[ndarray, "6"]:
        """
        To Ultralytics axis-aligned bounding box format.
        """
        raise NotImplementedError

    def visualize(self, zoom):
        geojson_box = self.to_geojson(None, None)
        Map = leafmap.Map(
            center=[self.properties["center_lat"], self.properties["center_lon"]],
            zoom=zoom,
        )
        Map.add_basemap("SATELLITE")
        Map.add_geojson(geojson_box, layer_name=self.class_name)
        return Map


class BBLabel(BB):
    def __init__(
        self,
        geo_box: Float[ndarray, "4 2"] | Float[ndarray, "2 2"],
        class_name: str,
        confidence: float | None = None,
    ):
        super().__init__(geo_box, class_name, confidence=confidence)
        del self.confidence  # Remove confidence from the label

    # We didn't implement because a path can have multiple labels/predictions.
    # @classmethod
    # def from_ultralytics_gms_path(cls, path: str, classes: Sequence, image_width: int, image_height: int, zoom: int):
    #     assert path.endswith(".txt"), "Only .txt files are supported."
    #     path = path.replace("%2C", ",")
    #     base_name = basename(path)
    #     base_name = base_name.replace(".txt", "")
    #     lat_str, lon_str = base_name.split(",")
    #     lat, lon = float(lat_str), float(lon_str)
    #     label =
    #     return cls.from_ultralytics_gms()

    @staticmethod
    def _from_geojson(feature: dict):
        class_name = feature["properties"]["class_name"]
        points = feature["geometry"]["coordinates"]
        points = deepcopy(points)  # avoid modifying the original points
        points = points[0]  # ignore additional axis
        # delete extra point which was added to close the polygon
        points.pop()

        lon1, lat1 = points[0]
        lon2, lat2 = points[1]
        lon3, lat3 = points[2]
        lon4, lat4 = points[3]

        geo_box = np.array([[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]])

        return geo_box, class_name

    def __repr__(self):
        return f"""BBLabel(box={self.geo_box}, 
class_name={self.class_name}, 
length_of_object={self.properties['length_of_object']:.2f} m,
width_of_object={self.properties['width_of_object']:.2f} m,
"""


class BBDetection(BB):
    pass


class AABBLabel(BBLabel):
    @classmethod
    def from_geojson(cls, label: dict) -> "AABBLabel":
        geo_box, class_name = cls._from_geojson(label)
        min_lon = geo_box[:, 0].min()
        max_lon = geo_box[:, 0].max()
        min_lat = geo_box[:, 1].min()
        max_lat = geo_box[:, 1].max()
        geo_box = np.array([[min_lon, min_lat], [max_lon, max_lat]])
        instance = cls(geo_box, class_name)
        instance.properties.update(label["properties"])
        return instance

class AABBDetection(BBDetection):
    pass


class OBBDetection(BBDetection):
    # TODO: nms supress method
    @classmethod
    def from_ultralytics_gms(
        cls,
        label: Float[ndarray, "9"] | Float[ndarray, "10"],
        classes: Sequence,
        image_width: int,
        image_height: int,
        zoom: int,
        image_center_lat: float,
        image_center_lon: float,
    ) -> "OBBDetection":
        box, class_name, confidence = cls._from_ultralytics(
            label, classes, image_width, image_height
        )
        if confidence is None:
            warnings.warn("Confidence is not provided. Setting it to 1.0.")
            confidence = 1.0

        geo_box = cls._local_box_to_geo(
            box, zoom, image_center_lat, image_center_lon, image_width, image_height
        )

        return cls(geo_box, class_name, confidence, image_width, image_height)


class OBBLabel(BBLabel):
    @classmethod
    def from_geojson(cls, label: dict) -> "OBBLabel":
        geo_box, class_name = cls._from_geojson(label)
        instance = cls(geo_box, class_name)
        instance.properties.update(label["properties"])
        return instance

    @classmethod
    def from_label_studio_csv(
        cls, label: dict, zoom: int, image_center_lat: float, image_center_lon: float
    ) -> "OBBLabel":
        """
        Label Studio CSV label keys: {"x", "y", "width", "height", "rotation", "rectanglelabels", "original_width", "original_height"}

        x, y: top left corner of the bounding box. Scale: [0, 100]
        width, height: width and height of the bounding box. Scale: [0, 100]
        rotation: rotation of the bounding box in degrees. Scale: [0, 90]
        rectanglelabels: class label
        original_width, original_height: original image size
        """
        x = label["x"]
        y = label["y"]
        width = label["width"]
        height = label["height"]
        rotation = label["rotation"]
        original_height = label["original_height"]
        original_width = label["original_width"]

        rotation_rad = np.radians(rotation)
        cos_rot = np.cos(rotation_rad)
        sin_rot = np.sin(rotation_rad)

        x_1 = x + width * cos_rot - height * sin_rot
        y_1 = y + width * sin_rot + height * cos_rot
        x_2 = x + width * cos_rot
        y_2 = y + width * sin_rot
        x_3 = x
        y_3 = y
        x_4 = x - height * sin_rot
        y_4 = y + height * cos_rot

        # scale to [0, 1]
        (x_1, x_2, x_3, x_4) = map(
            lambda x: x / 100, (x_1, x_2, x_3, x_4)
        )
        (y_1, y_2, y_3, y_4) = map(
            lambda y: y / 100, (y_1, y_2, y_3, y_4)
        )
        box = np.array([x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4], dtype=np.float32)

        geo_box = cls._local_box_to_geo(
            box,
            zoom,
            image_center_lat,
            image_center_lon,
            original_width,
            original_height,
        )
        class_name = label["rectanglelabels"][0]
        return cls(geo_box, class_name)


class AABB:
    pass
