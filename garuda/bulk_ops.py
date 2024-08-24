import os
from glob import glob
import numpy as np
import pandas as pd
from garuda.od import add_obb_to_label_studio_df
from beartype import beartype
from jaxtyping import jaxtyped

from garuda.ops import obb_to_aa

@jaxtyped(typechecker=beartype)
def write_obb_labels_from_label_studio_csv(save_dir: str, df:pd.DataFrame, label_map: dict):
    """
    Write YOLO labels from Label Studio CSV file to a directory.
    
    Parameters
    ----------
    save_dir: str
        Directory to save YOLO labels. MUST BE EMPTY.
        
    df: pd.DataFrame
        Label Studio DataFrame.
        This should be extracted from the Label Studio "CSV" option.
        
    label_map: dict
        Dictionary mapping class names to class IDs.
        Example: {"car": 0, "truck": 1, "bus": 2}
        
    Returns
    -------
    None
    """
    assert os.path.exists(save_dir), f"Directory {save_dir} does not exist. Please create the directory."
    
    items = glob(f"{save_dir}/*")
    assert len(items) == 0, f"Directory {save_dir} is not empty. Please provide an empty directory to avoid overwriting existing files."
    
    df = add_obb_to_label_studio_df(df, label_map)
    def get_image_basename(image_path):
        base_name = os.path.basename(image_path)
        # remove extensions
        base_name = base_name.replace(".jpg", "").replace(".png", "")
        # fix special characters
        base_name = base_name.replace("%2C", ",")
        return base_name
    df['image_basename'] = df['image'].apply(get_image_basename)
    
    for base_name, obb in zip(df['image_basename'], df['obb']):
        np.savetxt(f"{save_dir}/{base_name}.txt", obb, fmt="%d %f %f %f %f %f %f %f %f")
        
def write_aa_labels_from_obb(load_dir: str, save_dir: str, extension: str = "txt"):
    """
    Write AA labels from OBB labels.
    
    Parameters
    ----------
    load_dir: str
        Directory containing OBB labels. Default extension is ".txt", but can be changed using the "extension" parameter.
        
    save_dir: str
        Directory to save AA labels. MUST BE EMPTY.
        
    extension: str
        Extension of the OBB label files. Default is "txt".
        
    Returns
    -------
    None
    """
    
    items = glob(f"{save_dir}/*")
    assert len(items) == 0, f"Directory {save_dir} is not empty. Please provide an empty directory to avoid overwriting existing files."
    
    obb_labels = glob(f"{load_dir}/*.{extension}")
    for label in obb_labels:
        # Load OBB label
        obb = np.loadtxt(label, ndmin=2)
        assert obb.shape[1] in [9, 10], f"Invalid OBB shape: {obb.shape}. Expected (N, 9) or (N, 10)."
        
        # convert OBB to AA
        aa = obb_to_aa(obb)
        
        # Save AA label
        base_name = os.path.basename(label)
        save_path = f"{save_dir}/{base_name}"
        if aa.shape[1] == 5:
            fmt = "%d %f %f %f %f"
        elif aa.shape[1] == 6:
            fmt = "%d %f %f %f %f %f"
        else:
            raise ValueError(f"Invalid AA shape: {aa.shape}. Expected (N, 5) or (N, 6). Please raise an issue.")
        np.savetxt(save_path, aa, fmt=fmt)