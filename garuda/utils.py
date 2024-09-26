import numpy as np
from jaxtyping import jaxtyped
from beartype import beartype
from beartype.typing import Sequence, Tuple

from garuda.core import obb_iou_shapely, obb_smaller_box_ioa
from garuda.box import BB

@jaxtyped(typechecker=beartype)
def possible_intersection(obb1: BB, obb2: BB) -> bool:
    max_lon1 = obb1.properties['max_lon']
    min_lon1 = obb1.properties['min_lon']
    max_lat1 = obb1.properties['max_lat']
    min_lat1 = obb1.properties['min_lat']
    
    max_lon2 = obb2.properties['max_lon']
    min_lon2 = obb2.properties['min_lon']
    max_lat2 = obb2.properties['max_lat']
    min_lat2 = obb2.properties['min_lat']
    
    if (max_lat1 < min_lat2) or (min_lat1 > max_lat2):
        return False
    if (max_lon1 < min_lon2) or (min_lon1 > max_lon2):
        return False
    return True

@jaxtyped(typechecker=beartype)
def deduplicate(labels: Sequence[BB], ioa_threshold: float, iou_threshold: float, verbose=True) -> Tuple[Sequence[BB], float, float]:
    def verbose_print(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    
    # compute pairwise IOA and IoU
    ioa_array = np.zeros((len(labels), len(labels)))
    iou_array = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        ioa_array[i, i] = 1
        for j in range(i+1, len(labels)):
            if possible_intersection(labels[i], labels[j]):
                ioa_array[i, j] = obb_smaller_box_ioa(labels[i].geo_box, labels[j].geo_box)
                ioa_array[j, i] = ioa_array[i, j]
                iou_array[i, j] = obb_iou_shapely(labels[i].geo_box, labels[j].geo_box)
                iou_array[j, i] = iou_array[i, j]
            else:
                ioa_array[i, j] = 0
                ioa_array[j, i] = 0
                iou_array[i, j] = 0
                iou_array[j, i] = 0
                
    removed_indices = []

    verbose_print("#"*50)
    verbose_print("Removing overlapping OBBs based on IOA")
    verbose_print("#"*50)

    for master_i in range(len(labels)):
        verbose_print("Master iteration:", master_i)
        initial_len = len(removed_indices)
        verbose_print("Initial number of OBBs:", len(labels) - len(removed_indices))
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if i in removed_indices or j in removed_indices:
                    continue
                if ioa_array[i, j] > ioa_threshold:
                    if labels[i].properties['area'] < labels[j].properties['area']:
                        removed_indices.append(i)
                    else:
                        removed_indices.append(j)
                    break
        
        after_len = len(removed_indices)
        verbose_print("Number of OBBs after removing overlapping OBBs:", len(labels) - len(removed_indices))
        if initial_len == after_len:
            verbose_print("No more OBBs to remove")
            break
        
    verbose_print("#"*50)
    verbose_print("Removing overlapping OBBs based on IOU")
    verbose_print("#"*50)
        
    for master_i in range(len(labels)):
        verbose_print("Master iteration:", master_i)
        initial_len = len(removed_indices)
        verbose_print("Initial number of OBBs:", len(labels) - len(removed_indices))
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if i in removed_indices or j in removed_indices:
                    continue
                if iou_array[i, j] > iou_threshold:
                    if labels[i].properties['area'] < labels[j].properties['area']:
                        removed_indices.append(i)
                    else:
                        removed_indices.append(j)
                    break
        
        after_len = len(removed_indices)
        verbose_print("Number of OBBs after removing overlapping OBBs:", len(labels) - len(removed_indices))
        if initial_len == after_len:
            verbose_print("No more OBBs to remove")
            break

    max_iou = 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if i in removed_indices or j in removed_indices:
                continue
            if iou_array[i, j] > max_iou:
                max_iou = iou_array[i, j]
    verbose_print("Max IOU:", max_iou)

    max_ioa = 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if i in removed_indices or j in removed_indices:
                continue
            if ioa_array[i, j] > max_ioa:
                max_ioa = ioa_array[i, j]
    verbose_print("Max IOA:", max_ioa)
    
    final_labels = [labels[i] for i in range(len(labels)) if i not in removed_indices]
    return final_labels, max_ioa, max_iou