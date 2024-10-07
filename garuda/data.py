import os
from os.path import exists, basename, splitext, join
import shutil
from glob import glob
import utm
import networkx
import pyproj
import pandas as pd
import xarray as xr
import rioxarray as rxr
from beartype import beartype
from beartype.typing import Sequence, Tuple
from jaxtyping import jaxtyped
from shapely.geometry import box
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from garuda.base import tqdm, logger, AutoTypeChecker
from garuda.box import BB
from garuda.config import get_n_cpus
from garuda.utils import deduplicate

class GeoDataset(AutoTypeChecker, Dataset):
    def __init__(self, image_to_labels: dict):
        self.image_to_labels = image_to_labels
    
    @staticmethod
    def read_image(image_path: str) -> xr.DataArray:
        img = rxr.open_rasterio(image_path)
        GeoDataset._check_valid_image(img)
        return img
    
    @staticmethod
    def _check_valid_image(image: xr.DataArray):
        if not hasattr(image, "rio"):
            raise ValueError("Image must be a geo-referenced xarray.DataArray loaded with `rioxarray`")
        if len(image.shape) != 3:
            raise ValueError(f"Image shape must have only 3 axis (bands, y, x). Found {image.shape}")
        if image.shape[0] != 3:
            raise ValueError(f"Image must have exactly 3 bands (RGB). Found {image.shape[0]} bands.")
        
    @staticmethod
    def create_labels(image_path: str, image_attrs: dict, labels: Sequence[BB]):
        label_counter = pd.Series(index=range(len(labels)), data=0)
        image_counter = pd.Series(index=[image_path], data=0)
        label_dict = {}
        epsg = image_attrs["epsg"]
        geo_to_utm = pyproj.Transformer.from_crs(4326, epsg).transform
        for f_i, label in enumerate(labels):
            x_min, y_min = geo_to_utm(label.properties['min_lat'], label.properties['min_lon'])
            x_max, y_max = geo_to_utm(label.properties['max_lat'], label.properties['max_lon'])
            x_min_inside = image_attrs["x_min"] <= x_min <= image_attrs["x_max"]
            x_max_inside = image_attrs["x_min"] <= x_max <= image_attrs["x_max"]
            y_min_inside = image_attrs["y_min"] <= y_min <= image_attrs["y_max"]
            y_max_inside = image_attrs["y_min"] <= y_max <= image_attrs["y_max"]
            
            if x_min_inside and x_max_inside and y_min_inside and y_max_inside:
                # image_center_x = (image_attrs["x_min"] + image_attrs["x_max"] + 1) / 2
                # image_center_y = (image_attrs["y_min"] + image_attrs["y_max"] + 1) / 2
                # ultralytics_label = label.to_ultralytics_obb(epsg, classes, None, image_center_x, image_center_y, image_attrs['width'], image_attrs['height'], resolution).tolist()
                # ultralytics_label[0] = int(ultralytics_label[0])
                if image_path in label_dict:
                    # label_dict[image_path].append(" ".join(map(str, ultralytics_label)))
                    label_dict[image_path].append(label)
                else:
                    # label_dict[image_path] = [" ".join(map(str, ultralytics_label))]
                    label_dict[image_path] = [label]
                
                label_counter[f_i] += 1
                image_counter[image_path] += 1
        return label_dict, label_counter, image_counter
    
    @classmethod
    def from_images_and_labels(cls, image_paths: Sequence[str], labels: Sequence[BB], classes: Sequence[str], resolution: int, deduplicate_labels: bool = True, deduplication_ioa_threshold: float = 0.2, deduplication_iou_threshold: float = 0.2) -> "GeoDataset":
        """
        Create a GeoDataset from images and labels.
            
        Args:
            image_paths: List of image paths
            labels: List of bounding boxes
            classes: List of classes
            resolution: Image resolution in meters
            deduplicate_labels: Whether to deduplicate labels
            deduplication_ioa_threshold: IOA threshold for deduplication
            deduplication_iou_threshold: IOU threshold for deduplication
        """
        
        # cls._check_valid_images(image_paths)
        
        epsg = GeoDataset.read_image(image_paths[0]).rio.crs.to_epsg()
        if str(epsg).startswith("326"):
            logger.info("Images are in UTM projection.")
            projection = "utm"

        # deduplicate labels
        if deduplicate_labels:
            logger.info("Running deduplication on labels...")
            labels, _, _ = deduplicate(labels, iou_threshold=deduplication_iou_threshold, ioa_threshold=deduplication_ioa_threshold)
        else:
            logger.info("Skipping deduplication because `deduplicate` is set to False.")
            
        # get the range of each image file
        logger.info("Getting the range of each image file")
        def _get_ranges(path):
            ds = rxr.open_rasterio(path)
            x_min = ds.x.min().item()
            x_max = ds.x.max().item()
            y_min = ds.y.min().item()
            y_max = ds.y.max().item()
            epsg = ds.rio.crs.to_epsg()
            height = ds.rio.height
            width = ds.rio.width
            return (path, {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "epsg": epsg, "height": height, "width": width})

        label_dict = dict(Parallel(get_n_cpus())(delayed(_get_ranges)(path) for path in image_paths))
        
        # Assign labels to images
        results = Parallel(get_n_cpus())(delayed(GeoDataset.create_labels)(image_path, image_attrs, labels, classes, resolution) for image_path, image_attrs in tqdm(label_dict.items()))
        
        label_dicts = [result[0] for result in results]
        label_counters = [result[1] for result in results]
        image_counters = [result[2] for result in results]

        # add label key
        for key in label_dict.keys():
            label_dict[key].update({"label": []})
        for d in label_dicts:
            for key, value in d.items():
                label_dict[key]["label"].extend(value)

        label_counter = pd.concat(label_counters, axis=1).sum(axis=1)
        image_counter = pd.concat(image_counters, axis=1).sum(axis=1)
        
        # stats
        logger.info(f"Number of images: {len(image_paths)}")
        images_with_no_labels = len([k for k, v in label_dict.items() if len(v['label']) == 0])
        logger.info(f"Number of images with no labels: {images_with_no_labels}")
        logger.info(f"Number of labels: {len(labels)}")
        logger.info(f"Number of classes: {len(classes)}")
        logger.info(f"Classes: {classes}")

        series = label_counter.value_counts().sort_index()
        series.name = "index: number of times same label repeats in multiple images, values: number of such labels"
        logger.info("\n"+str(series))

        series = pd.Series(image_counter).value_counts().sort_index()
        series.name = "index: number of labels in a single image, values: number of such images"
        logger.info("\n"+str(series))

        return cls(label_dict)
    
    def to_ultralytics_obb(self, classes: Sequence[str], resolution: int, save_dir: str, write_empty_labels: bool, overwrite: bool = False):
        """
        Save the dataset to Ultralytics format
        
        Args:
            classes: List of classes
            resolution: Image resolution in meters (only applicable for UTM projection)
            save_dir: Directory to save the dataset
            write_empty_labels: Whether to write empty labels and images
            overwrite: Overwrite the directory
        """
        
        if os.path.exists(save_dir):
            if not overwrite:
                raise FileExistsError(f"Directory already exists: {save_dir}. Set overwrite=True to overwrite the directory.")
            shutil.rmtree(save_dir)
        
        os.makedirs(save_dir, exist_ok=False)
        
        # Create hard-links for the images
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=False)
        logger.info(f"Creating hard-links for images to {os.path.join(save_dir, 'images')}")
        for image_path in tqdm(self.image_to_labels.keys()):
            base_name = basename(image_path)
            os.link(image_path, os.path.join(save_dir, "images", base_name))
        
        # Create labels
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=False)
        logger.info(f"Creating labels in Ultralytics format to {os.path.join(save_dir, 'labels')}")
        for image_path, image_attrs in tqdm(self.image_to_labels.items()):
            image_center_x = (image_attrs["x_min"] + image_attrs["x_max"] + 1) / 2
            image_center_y = (image_attrs["y_min"] + image_attrs["y_max"] + 1) / 2
            base_name = splitext(basename(image_path))[0]
            path = join(save_dir, "labels", base_name+".txt")
            epsg = image_attrs["epsg"]
            
            with open(path, "w") as f:
                label_str_list = []
                for label in self.image_to_labels[image_path]["label"]:
                    ultralytics_obb = label.to_ultralytics_obb(epsg, classes, None, image_center_x, image_center_y, image_attrs['width'], image_attrs['height'], resolution).tolist()
                    ultralytics_obb[0] = int(ultralytics_obb[0])
                    label_str_list.append(" ".join(map(str, ultralytics_obb)))
                f.write("\n".join(label_str_list))
                
        # Create data.yml
        with open(join(save_dir, "data.yml"), "w") as f:
            f.write(f"train: {os.path.join(save_dir, 'images')}\n")
            f.write(f"val: {os.path.join(save_dir, 'images')}\n")
            f.write(f"predict: {os.path.join(save_dir, 'images')}\n")
            f.write(f"nc: {len(classes)}\n")
            f.write("names: " + " ".join(classes))