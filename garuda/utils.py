from os.path import exists
import numpy as np
from numpy import ndarray
from jaxtyping import jaxtyped, Float, Int
from beartype import beartype
from beartype.typing import Sequence, Tuple, Literal
import geojson
from garuda.core import obb_iou_shapely, obb_smaller_box_ioa
from garuda.box import BB, OBBLabel
from tqdm.auto import tqdm
import psutil
from joblib import Parallel, delayed
import pandas as pd
import geopandas as gpd

@jaxtyped(typechecker=beartype)
def obb_labels_from_geojson(collection = str | dict):
    """
    Extract OBB labels from GeoJSON collection
    
    Parameters
    ----------
    collection : GeoJSON collection or path to GeoJSON file
    
    Returns
    -------
    List of OBB labels
    """
    if isinstance(collection, str):
        with open(collection) as f:
            collection = geojson.load(f)
    return [OBBLabel.from_geojson(feature) for feature in collection['features']]

@jaxtyped(typechecker=beartype)
def obb_labels_to_geojson(labels: Sequence[BB], save_path: str, overwrite: bool, source: str = None, task_name: str = None):
    """
    Save OBB labels to GeoJSON file
    
    Parameters
    ----------
    labels : List of OBB labels
    
    save_path : Path to save GeoJSON file
    
    overwrite : Overwrite existing file
    """
    if not save_path.endswith(".geojson"):
        raise ValueError(f"Expected `save_path` to end with `.geojson`, got: {save_path}")
    if not overwrite and exists(save_path):
        raise FileExistsError(f"File already exists: {save_path}. Set overwrite=True to overwrite the file.")
    with open(save_path, "w") as f:
        geojson.dump(geojson.FeatureCollection([label.to_geojson(source=source, task_name=task_name) for label in labels]), f)

@jaxtyped(typechecker=beartype)
def points_within_polygons(geo_points: Float[ndarray, "n 2"], geo_polygons: gpd.GeoDataFrame) -> pd.DataFrame:
    """ Points within Polygons

    Args:
        geo_points: Array of geo-points (lat, lon)
        geo_polygons : GeoDataFrame of polygons (size: m)

    Returns:
        DataFrame of points within polygons (n, m)
    """
    assert geo_polygons.crs == "EPSG:4326", "Only EPSG:4326 is supported"
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(geo_points[:, 1], geo_points[:, 0]), crs="EPSG:4326")
    pairwise_check = points.geometry.apply(lambda x: geo_polygons.contains(x))
    pairwise_check.index.name = "points"
    return pairwise_check

@jaxtyped(typechecker=beartype)
def points_within_boxes(geo_points: Float[ndarray, "n 2"], geo_boxes: Sequence[BB]) -> dict:
    """
    Find Geo-Points within Geo-Boxes
    
    Parameters
    ----------
    geo_points : Array of geo-points (lat, lon)
        
    geo_boxes : List of geo-boxes
    
    Returns
    -------
    dict of various statistics
    """
    features = [box.to_geojson(None, None) for box in geo_boxes]
    polygons = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    pairwise_check = points_within_polygons(geo_points, polygons)

    points_within_index = pairwise_check.any(axis=1).values
    polygons_contains = pairwise_check.any(axis=0).values
    
    points_within = geo_points[points_within_index]
    geo_boxes_contains = [box for i, box in enumerate(geo_boxes) if polygons_contains[i]]
    return {"points_within": points_within, "geo_boxes_contains": geo_boxes_contains, "pairwise_check": pairwise_check}
    

@jaxtyped(typechecker=beartype)
def deduplicate(labels: Sequence[BB], ioa_threshold: float, iou_threshold: float, verbose: bool = True) -> Tuple[Sequence[BB], dict, dict]:
    """
    Deduplicate Bounding boxes based on IOA and IOU thresholds.
    
    Parameters
    ----------
    labels : List of bounding boxes
        
    ioa_threshold : IOA threshold to remove small bounding boxes contained in larger bounding boxes
        
    iou_threshold : IOU threshold to remove overlapping bounding boxes. The box with the smaller area is removed
        
    verbose : Print verbose output, by default True

    Returns
    -------
    - List of deduplicated bounding boxes
    - pair-wise IOA dictionary of returned bounding boxes (index is from the original list)
    - pair-wise IOU dictionary of returned bounding boxes (index is from the original list)
    """
    
    def verbose_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
            
    verbose_print("Number of bounding boxes:", len(labels))
    
    max_lon = np.array([bb.properties['max_lon'] for bb in tqdm(labels)])
    min_lon = np.array([bb.properties['min_lon'] for bb in tqdm(labels)])
    max_lat = np.array([bb.properties['max_lat'] for bb in tqdm(labels)])
    min_lat = np.array([bb.properties['min_lat'] for bb in tqdm(labels)])
    
    check = max_lat.reshape(-1, 1) < min_lat.reshape(1, -1)
    check = np.logical_or(min_lat.reshape(-1, 1) > max_lat.reshape(1, -1), check)
    check = np.logical_or(max_lon.reshape(-1, 1) < min_lon.reshape(1, -1), check)
    check = np.logical_or(min_lon.reshape(-1, 1) > max_lon.reshape(1, -1), check)
    
    ij_queries = np.argwhere(~check)
    ij_queries = set([(i, j) for i, j in ij_queries if i < j]) # don't double count
    verbose_print("Number of possible intersection pairs:", len(ij_queries))

    def compute_ioa_iou(i, j):
        ioa = obb_smaller_box_ioa(labels[i].geo_box, labels[j].geo_box)
        iou = obb_iou_shapely(labels[i].geo_box, labels[j].geo_box)
        return ioa, iou
    
    verbose_print(f"Computing IOA and IOU ...")
    ioa_dict = {}
    iou_dict = {}
    for i, j in tqdm(ij_queries):
        ioa, iou = compute_ioa_iou(i, j)
        ioa_dict.update({(i, j): ioa})
        iou_dict.update({(i, j): iou})

    verbose_print("#"*50)
    verbose_print("Removing overlapping bounding boxes based on IOA")
    verbose_print("#"*50)
    
    removed_indices = set()
    for master_i in range(len(labels)):
        verbose_print("Master iteration:", master_i)
        n_current_removed = len(removed_indices)
        verbose_print("Initial number of bounding boxes:", len(labels) - len(removed_indices))
        for i, j in ij_queries:
            if i in removed_indices or j in removed_indices:
                continue
            if ioa_dict[(i, j)] > ioa_threshold:
                if labels[i].properties['area'] < labels[j].properties['area']:
                    removed_indices.add(i)
                else:
                    removed_indices.add(j)
        
        n_after_removed = len(removed_indices)
        verbose_print("Number of bounding boxes after removing overlapping boxes:", len(labels) - len(removed_indices))
        if n_after_removed == n_current_removed:
            verbose_print("No more bounding boxes to remove")
            break
        
    verbose_print("#"*50)
    verbose_print("Removing overlapping bounding boxes based on IOU")
    verbose_print("#"*50)
        
    for master_i in range(len(labels)):
        verbose_print("Master iteration:", master_i)
        n_current_removed = len(removed_indices)
        verbose_print("Initial number of bounding boxes:", len(labels) - len(removed_indices))
        for i, j in ij_queries:
            if i in removed_indices or j in removed_indices:
                continue
            if iou_dict[(i, j)] > iou_threshold:
                if labels[i].properties['area'] < labels[j].properties['area']:
                    removed_indices.add(i)
                else:
                    removed_indices.add(j)
        
        n_after_removed = len(removed_indices)
        verbose_print("Number of bounding boxes after removing overlapping bounding boxes:", len(labels) - len(removed_indices))
        if n_after_removed == n_current_removed:
            verbose_print("No more bounding boxes to remove")
            break

    remaining_ioa_dict = {}
    remaining_iou_dict = {}
    for i, j in ij_queries:
        if i in removed_indices or j in removed_indices:
            continue
        remaining_ioa_dict.update({(i, j): ioa_dict[(i, j)]})
        remaining_iou_dict.update({(i, j): iou_dict[(i, j)]})
    
    verbose_print("Max IOA:", max(remaining_ioa_dict.values()))
    verbose_print("Max IOU:", max(remaining_iou_dict.values()))
    
    verbose_print("Final number of duplicate bounding boxes to remove:", len(removed_indices))
    final_labels = [labels[i] for i in range(len(labels)) if i not in removed_indices]
    return final_labels, remaining_ioa_dict, remaining_iou_dict