import numpy as np
import pandas as pd
from numpy import ndarray

from dataclasses import dataclass

# from supervision.metrics.detection import ConfusionMatrix as SVConfusionMatrix, MeanAveragePrecision as SVMeanAveragePrecision

from garuda.utils import webm_pixel_to_geo, geo_to_webm_pixel, local_to_geo, label_studio_csv_to_obb, obb_iou
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from beartype.typing import Union, List
import warnings

@jaxtyped(typechecker=beartype)
def yolo_aa_to_geo(yolo_label: Union[str, Float[ndarray, "n 4"], Float[ndarray, "n 5"]], zoom: int, img_center_lat: float, img_center_lon: float, img_width: int, img_height: int) -> Float[ndarray, "n 3"]:
    """
    Convert YOLO label to geographic coordinates.
    
    yolo_label: YOLO label (or str path) in the format [class, x_center, y_center, width, height] or [class, x_center, y_center, width, height, confidence].
        class range: [0, 1, 2, ...]
        x_center range: [0, 1]
        y_center range: [0, 1]
        width range: [0, 1]
        height range: [0, 1]
        confidence range: [0, 1]
    
        Example 1: [0, 0.5, 0.5, 0.1, 0.1]
        Example 2: [0, 0.5, 0.5, 0.1, 0.1, 0.9]

    zoom: Zoom level of the map.
        Range: [0, 20]
        Example: 17

    img_center_lon: Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
        
    img_center_lat: Latitude of the center of the image.
        Range: approx [-85, 85] (valid range for Web Mercator projection)
        Example: 37.7749
        
    img_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    img_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
    
    Returns
    -------
    geo_coords: Geographic coordinates in decimal degrees.
        Format: [class, latitude, longitude]
        Example: [0, 37.7749, -122.4194]
    """
    if isinstance(yolo_label, str):
        yolo_label = np.loadtxt(yolo_label, ndmin=2)
        return yolo_aa_to_geo(yolo_label, zoom, img_center_lat, img_center_lon, img_width, img_height)  # To trigger type/shape checking
    
    # Get bbox center in image coordinates
    x_c = yolo_label[:, 1]
    y_c = yolo_label[:, 2]
    
    # Get bbox center in Web Mercator projection
    bbox_geo = local_to_geo(x_c, y_c, zoom, img_center_lat, img_center_lon, img_width, img_height)
    
    # Append class ID to bbox_geo
    class_ids = yolo_label[:, 0:1]
    output = np.concatenate((class_ids, bbox_geo), axis=1)
    
    return output

@jaxtyped(typechecker=beartype)
def yolo_obb_to_geo(yolo_label: Union[str, Float[ndarray, "n 9"], Float[ndarray, "n 10"]], zoom: int, img_center_lat: float, img_center_lon: float, img_width: int, img_height: int) -> Float[ndarray, "n 3"]:
    """
    Convert YOLO label to geographic coordinates.
    
    yolo_label: YOLO label (or str path) in the format [class, x1, y1, x2, y2, x3, y3, x4, y4] or [class, x1, y1, x2, y2, x3, y3, x4, y4, confidence].
        class range: [0, 1, 2, ...]
        x1, x2, x3, x4 range: [0, 1]
        y1, y2, y3, y4 range: [0, 1]
        confidence range: [0, 1]
    
        Example 1: [0, 0.5, 0.5, 0.1, 0.1, 0.0]
        Example 2: [0, 0.5, 0.5, 0.1, 0.1, 0.0, 0.9]
        
    zoom: Zoom level of the map.
        Range: [0, 20]
        Example: 17

    img_center_lon: Longitude of the center of the image.
        Range: [-180, 180]
        Example: -122.4194
        
    img_center_lat: Latitude of the center of the image.
        Range: approx [-85, 85] (valid range for Web Mercator projection)
        Example: 37.7749

    img_width: Width of the image in pixels.
        Range: [0, inf]
        Example: 640
        
    img_height: Height of the image in pixels.
        Range: [0, inf]
        Example: 480
    
    Returns
    -------
    geo_coords: Geographic coordinates in decimal degrees.
        Format: [class, latitude, longitude]
        Example: [0, 37.7749, -122.4194]
    """
    
    if isinstance(yolo_label, str):
        yolo_label = np.loadtxt(yolo_label, ndmin=2)
        return yolo_obb_to_geo(yolo_label, zoom, img_center_lat, img_center_lon, img_width, img_height)  # To trigger type/shape checking
    
    # Get bbox center in image coordinates
    xyxyxyxy = yolo_label[:, 1:9]
    x_c = xyxyxyxy[:, ::2].mean(axis=1)
    y_c = xyxyxyxy[:, 1::2].mean(axis=1)
    
    # Get bbox center in Web Mercator projection
    bbox_geo = local_to_geo(x_c, y_c, zoom, img_center_lat, img_center_lon, img_width, img_height)

    # Append class ID to bbox_geo
    class_ids = yolo_label[:, 0:1]
    output = np.concatenate((class_ids, bbox_geo), axis=1)
    
    return output


def add_obb_to_label_studio_df(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """
    Add YOLO oriented bounding box to Label Studio DataFrame.
    
    Parameters
    ----------
    df: Label Studio DataFrame.
        This should be extracted from the Label Studio "CSV" option.
        
    label_map: Dictionary mapping class names to class IDs.
        Example: {"car": 0, "truck": 1, "bus": 2}
    
    Returns
    -------
    df: Label Studio DataFrame with YOLO oriented bounding box added as a new column named "obb".
    """
    
    def process_row(row):
        try:
            str_label = row["label"]
            labels = eval(str_label)
            obb_list = []
            for label in labels:
                x1 = label['x']
                y1 = label['y']
                width = label['width']
                height = label['height']
                rotation = label['rotation']
                class_name = label['rectanglelabels'][0]
                
                obb = label_studio_csv_to_obb(x1, y1, width, height, rotation, class_name, label_map)
                obb_list.append(obb)
            obb = np.stack(obb_list)
            return obb
        except Exception as e:
            warnings.warn(f"Error processing row: {row}\n{e}")
            return np.zeros((0, 9))
    
    df["obb"] = df.apply(process_row, axis=1)
    return df

@dataclass
class ConfusionMatrix(SVConfusionMatrix):
    """
    Confusion Matrix for Object Detection inspired from `ConfusionMatrix` class in Supervision library.
    
    """
    
    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_obb_tensors(
        cls,
        predictions: List[Float[ndarray, "_ 10"]],
        targets: List[Float[ndarray, "_ 9"]],
        classes: List[str],
        conf_threshold: float,
        iou_threshold: float,
    ) -> "ConfusionMatrix":
        """
        Calculate Confusion Matrix based on Oriented Bounding Box (OBB) predictions and targets.
        
        Parameters
        ----------
        predictions: Each element of the list describes a single image and has bounding boxes in `[class_id, x1, y1, x2, y2, x3, y3, x4, y4, confidence]` format.
        
        targets: Each element of the list describes a single image and has bounding boxes in `[class_id, x1, y1, x2, y2, x3, y3, x4, y4]` format.
        
        classes (List[str]): Model class names.
        
        conf_threshold (float): Detection confidence threshold between `0` and `1`.
            Detections with lower confidence will be excluded.
        
        iou_threshold (float): Detection iou  threshold between `0` and `1`.
            Detections with lower iou will be classified as `FP`.
        """
        
        num_classes = len(classes)
        matrix = np.zeros((num_classes + 1, num_classes + 1))
        for true_batch, detection_batch in zip(targets, predictions):
            matrix += cls.evaluate_detection_obb_batch(
                predictions=detection_batch,
                targets=true_batch,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        return cls(
            matrix=matrix,
            classes=classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        
    @staticmethod
    @jaxtyped(typechecker=beartype)
    def evaluate_detection_obb_batch(
        predictions: Float[ndarray, "n 10"],
        targets: Float[ndarray, "m 9"],
        num_classes: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Float[ndarray, "{num_classes}+1 {num_classes}+1"]:
        """
        Calculate confusion matrix for a batch of obb detections for a single image.

        Parameters:
        -----------
            predictions: Batch prediction. Describes a single image and has format `[class_id, x1, y1, x2, y2, x3, y3, x4, y4, confidence]`.
            
            targets: Batch target. Describes a single image and has format `[class_id, x1, y1, x2, y2, x3, y3, x4, y4]`.
            
            num_classes (int): Number of classes.
            
            conf_threshold (float): Detection confidence threshold between `0` and `1`.
                Detections with lower confidence will be excluded.
                
            iou_threshold (float): Detection iou  threshold between `0` and `1`.
                Detections with lower iou will be classified as `FP`.

        Returns:
            np.ndarray: Confusion matrix based on a single image.
        """
        result_matrix = np.zeros((num_classes + 1, num_classes + 1))

        conf_idx = 9
        confidence = predictions[:, conf_idx]
        detection_batch_filtered = predictions[confidence > conf_threshold]

        class_id_idx = 0
        true_classes = np.array(targets[:, class_id_idx], dtype=np.int16)
        detection_classes = np.array(
            detection_batch_filtered[:, class_id_idx], dtype=np.int16
        )
        true_boxes = targets[:, 1:9]
        detection_boxes = detection_batch_filtered[:, 1:9]

        iou_batch = obb_iou(true_obb=true_boxes, pred_obb=detection_boxes)
        matched_idx = np.asarray(iou_batch > iou_threshold).nonzero()

        if matched_idx[0].shape[0]:
            matches = np.stack(
                (matched_idx[0], matched_idx[1], iou_batch[matched_idx]), axis=1
            )
            matches = ConfusionMatrix._drop_extra_matches(matches=matches)
        else:
            matches = np.zeros((0, 3))

        matched_true_idx, matched_detection_idx, _ = matches.transpose().astype(
            np.int16
        )

        for i, true_class_value in enumerate(true_classes):
            j = matched_true_idx == i
            if matches.shape[0] > 0 and sum(j) == 1:
                result_matrix[
                    true_class_value, detection_classes[matched_detection_idx[j]]
                ] += 1  # TP
            else:
                result_matrix[true_class_value, num_classes] += 1  # FN

        for i, detection_class_value in enumerate(detection_classes):
            if not any(matched_detection_idx == i):
                result_matrix[num_classes, detection_class_value] += 1  # FP

        return result_matrix
    
    @property
    @jaxtyped(typechecker=beartype)
    def true_positives(self) -> Int[ndarray, "{len(self.classes)}"]:
        """
        Calculate True Positives (TP) for each class.
        
        Returns
        -------
        np.ndarray: True Positives for each class.
        """
        return self.matrix.diagonal()[:-1].astype(int)
    
    @property
    @jaxtyped(typechecker=beartype)
    def predicted_positives(self) -> Int[ndarray, "{len(self.classes)}"]:
        """
        Calculate Predicted Positives (PP) for each class.
        
        Returns
        -------
        np.ndarray: Predicted Positives for each class.
        """
        return self.matrix.sum(axis=0)[:-1].astype(int)
    
    @property
    @jaxtyped(typechecker=beartype)
    def false_positives(self) -> Int[ndarray, "{len(self.classes)}"]:
        """
        Calculate False Positives (FP) for each class.
        
        Returns
        -------
        np.ndarray: False Positives for each class.
        """
        return self.predicted_positives - self.true_positives
    
    @property
    @jaxtyped(typechecker=beartype)
    def actual_positives(self) -> Int[ndarray, "{len(self.classes)}"]:
        """
        Calculate Actual Positives (AP) for each class.
        
        Returns
        -------
        np.ndarray: Actual Positives for each class.
        """
        return self.matrix.sum(axis=1)[:-1].astype(int)
    
    @property
    @jaxtyped(typechecker=beartype)
    def false_negatives(self) -> Int[ndarray, "{len(self.classes)}"]:
        """
        Calculate False Negatives (FN) for each class.
        
        Returns
        -------
        np.ndarray: False Negatives for each class.
        """
        return self.actual_positives - self.true_positives
    
    @property
    @jaxtyped(typechecker=beartype)
    def precision(self) -> Float[ndarray, "{len(self.classes)}"]:
        """
        Calculate precision for each class.
        
        Returns
        -------
        np.ndarray: Precision for each class.
        """
        precision = self.true_positives / self.predicted_positives
        return precision
        
    @property
    @jaxtyped(typechecker=beartype)
    def recall(self) -> Float[ndarray, "{len(self.classes)}"]:
        """
        Calculate recall for each class.
        
        Returns
        -------
        np.ndarray: Recall for each class.
        """
        recall = self.true_positives / self.actual_positives
        return recall
    
    @property
    @jaxtyped(typechecker=beartype)
    def f1_score(self) -> Float[ndarray, "{len(self.classes)}"]:
        """
        Calculate F1 score for each class.
        
        Returns
        -------
        np.ndarray: F1 score for each class.
        """
        # f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        # OR more efficiently
        f1_score = 2 * self.true_positives / (self.predicted_positives + self.actual_positives)
        return f1_score
    
    @property
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame.
        
        Returns
        -------
        pd.DataFrame: Summary DataFrame.
        """
        summary_df = pd.DataFrame(columns=self.classes)
        
        summary_df.loc["Actual Positives", self.classes] = self.actual_positives
        summary_df.loc["Predicted Positives", self.classes] = self.predicted_positives
        summary_df.loc["True Positives", self.classes] = self.true_positives
        summary_df.loc["False Positives", self.classes] = self.false_positives
        summary_df.loc["False Negatives", self.classes] = self.false_negatives
        summary_df.loc["Precision", self.classes] = self.precision
        summary_df.loc["Recall", self.classes] = self.recall
        summary_df.loc["F1 Score", self.classes] = self.f1_score
        return summary_df