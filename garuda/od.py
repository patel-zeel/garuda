import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from numpy import ndarray

from dataclasses import dataclass
from supervision.utils.file import list_files_with_extensions, read_yaml_file, read_txt_file
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.dataset.core import DetectionDataset as SVDetectionDataset
from supervision.dataset.formats.yolo import _with_mask, _extract_class_names, yolo_annotations_to_detections
from supervision.metrics.detection import ConfusionMatrix as SVConfusionMatrix
from supervision.metrics.mean_average_precision import MeanAveragePrecision as SVMeanAveragePrecision, MeanAveragePrecisionResult
from supervision.metrics.core import Metric, MetricTarget
from supervision.detection.utils import box_iou_batch, mask_iou_batch
from supervision.detection.core import Detections
from garuda.core import webm_pixel_to_geo, geo_to_webm_pixel, local_to_geo, obb_iou, obb_iou_shapely_batch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from beartype.typing import Union, List, Tuple, Dict
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


# def add_obb_to_label_studio_df(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
#     """
#     Add YOLO oriented bounding box to Label Studio DataFrame.
    
#     Parameters
#     ----------
#     df: Label Studio DataFrame.
#         This should be extracted from the Label Studio "CSV" option.
        
#     label_map: Dictionary mapping class names to class IDs.
#         Example: {"car": 0, "truck": 1, "bus": 2}
    
#     Returns
#     -------
#     df: Label Studio DataFrame with YOLO oriented bounding box added as a new column named "obb".
#     """
    
#     def process_row(row):
#         try:
#             str_label = row["label"]
#             labels = eval(str_label)
#             obb_list = []
#             for label in labels:
#                 x1 = label['x']
#                 y1 = label['y']
#                 width = label['width']
#                 height = label['height']
#                 rotation = label['rotation']
#                 class_name = label['rectanglelabels'][0]
                
#                 obb = label_studio_csv_to_obb(x1, y1, width, height, rotation, class_name, label_map)
#                 obb_list.append(obb)
#             obb = np.stack(obb_list)
#             return obb
#         except Exception as e:
#             warnings.warn(f"Error processing row: {row}\n{e}")
#             return np.zeros((0, 9))
    
#     df["obb"] = df.apply(process_row, axis=1)
#     return df

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
            # print(detection_batch.shape, true_batch.shape)
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
        true_boxes = targets[:, 1:9].reshape(-1, 4, 2)
        detection_boxes = detection_batch_filtered[:, 1:9].reshape(-1, 4, 2)

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
        # fill NaN values with 0
        precision = np.nan_to_num(precision)
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
    
class MeanAveragePrecision(SVMeanAveragePrecision):
    def __init__(
        self,
        metric_target: MetricTarget = MetricTarget.BOXES,
        class_agnostic: bool = False,
    ):
        """
        Initialize the Mean Average Precision metric.

        Args:
            metric_target (MetricTarget): The type of detection data to use.
            class_agnostic (bool): Whether to treat all data as a single class.
        """
        self._metric_target = metric_target
        self._class_agnostic = class_agnostic

        self._predictions_list: List[Detections] = []
        self._targets_list: List[Detections] = []
        
    def _detections_content(self, detections: Detections) -> np.ndarray:
        """Return boxes, masks or oriented bounding boxes from detections."""
        if self._metric_target == MetricTarget.BOXES:
            return detections.xyxy
        if self._metric_target == MetricTarget.MASKS:
            return (
                detections.mask
                if detections.mask is not None
                else self._make_empty_content()
            )
        if self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
            obb = detections.data.get(ORIENTED_BOX_COORDINATES)
            if obb is not None:
                return obb.astype(np.float32)
            return self._make_empty_content()
        raise ValueError(f"Invalid metric target: {self._metric_target}")
    
    def _compute(
        self,
        predictions_list: List[Detections],
        targets_list: List[Detections],
    ) -> MeanAveragePrecisionResult:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        stats = []

        for predictions, targets in zip(predictions_list, targets_list):
            prediction_contents = self._detections_content(predictions)
            target_contents = self._detections_content(targets)

            if len(targets) > 0:
                if len(predictions) == 0:
                    stats.append(
                        (
                            np.zeros((0, iou_thresholds.size), dtype=bool),
                            np.zeros((0,), dtype=np.float32),
                            np.zeros((0,), dtype=int),
                            targets.class_id,
                        )
                    )

                else:
                    if self._metric_target == MetricTarget.BOXES:
                        iou = box_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.MASKS:
                        iou = mask_iou_batch(target_contents, prediction_contents)
                    elif self._metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
                        iou = obb_iou(target_contents, prediction_contents)
                        # iou = obb_iou_shapely_batch(target_contents, prediction_contents)
                    else:
                        raise NotImplementedError(
                            f"Unsupported metric target={self._metric_target} for MeanAveragePrecision"
                        )

                    matches = self._match_detection_batch(
                        predictions.class_id, targets.class_id, iou, iou_thresholds
                    )
                    stats.append(
                        (
                            matches,
                            predictions.confidence,
                            predictions.class_id,
                            targets.class_id,
                        )
                    )

        # Compute average precisions if any matches exist
        if stats:
            concatenated_stats = [np.concatenate(items, 0) for items in zip(*stats)]
            average_precisions, unique_classes = self._average_precisions_per_class(
                *concatenated_stats
            )
            mAP_scores = np.mean(average_precisions, axis=0)
        else:
            mAP_scores = np.zeros((10,), dtype=np.float32)
            unique_classes = np.empty((0,), dtype=int)
            average_precisions = np.empty((0, len(iou_thresholds)), dtype=np.float32)

        return MeanAveragePrecisionResult(
            metric_target=self._metric_target,
            mAP_scores=mAP_scores,
            iou_thresholds=iou_thresholds,
            matched_classes=unique_classes,
            ap_per_class=average_precisions,
        )
        
        
    @staticmethod
    def _average_precisions_per_class(
        matches: np.ndarray,
        prediction_confidence: np.ndarray,
        prediction_class_ids: np.ndarray,
        true_class_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

        Args:
            matches (np.ndarray): True positives.
            prediction_confidence (np.ndarray): Objectness value from 0-1.
            prediction_class_ids (np.ndarray): Predicted object classes.
            true_class_ids (np.ndarray): True object classes.
            eps (float, optional): Small value to prevent division by zero.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Average precision for different
                IoU levels, and an array of class IDs that were matched.
        """
        eps = 1e-16

        sorted_indices = np.argsort(-prediction_confidence)
        matches = matches[sorted_indices]
        prediction_class_ids = prediction_class_ids[sorted_indices]

        unique_classes, class_counts = np.unique(true_class_ids, return_counts=True)
        num_classes = unique_classes.shape[0]

        average_precisions = np.zeros((num_classes, matches.shape[1]))

        for class_idx, class_id in enumerate(unique_classes):
            is_class = prediction_class_ids == class_id
            total_true = class_counts[class_idx]
            total_prediction = is_class.sum()

            if total_prediction == 0 or total_true == 0:
                continue

            false_positives = (1 - matches[is_class]).cumsum(0)
            true_positives = matches[is_class].cumsum(0)
            false_negatives = total_true - true_positives

            recall = true_positives / (true_positives + false_negatives + eps)
            precision = true_positives / (true_positives + false_positives)

            for iou_level_idx in range(matches.shape[1]):
                average_precisions[class_idx, iou_level_idx] = (
                    MeanAveragePrecision._compute_average_precision(
                        recall[:, iou_level_idx], precision[:, iou_level_idx]
                    )
                )

        return average_precisions, unique_classes
        
    @staticmethod
    def _compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Compute the average precision using 101-point interpolation (COCO), given
            the recall and precision curves.

        Args:
            recall (np.ndarray): The recall curve.
            precision (np.ndarray): The precision curve.

        Returns:
            (float): Average precision.
        """
        ########### Area method
        # if len(recall) == 0 and len(precision) == 0:
        #     return 0.0

        # recall_levels = np.linspace(0, 1, 101)
        # precision_levels = np.zeros_like(recall_levels)
        # for r, p in zip(recall[::-1], precision[::-1]):
        #     precision_levels[recall_levels <= r] = p

        # average_precision = (1 / 100 * precision_levels).sum()
        # return average_precision
        
        ############ 101 point method
        extended_recall = np.concatenate(([0.0], recall, [1.0]))
        extended_precision = np.concatenate(([1.0], precision, [0.0]))
        max_accumulated_precision = np.flip(
            np.maximum.accumulate(np.flip(extended_precision))
        )
        interpolated_recall_levels = np.linspace(0, 1, 101)
        interpolated_precision = np.interp(
            interpolated_recall_levels, extended_recall, max_accumulated_precision
        )
        average_precision = np.trapz(interpolated_precision, interpolated_recall_levels)
        # raise NotImplementedError("101 method is not implemented yet.")
        return average_precision

def load_yolo_annotations(
    images_directory_path: str,
    annotations_directory_path: str,
    data_yaml_path: str,
    force_masks: bool = False,
    is_obb: bool = False,
) -> Tuple[List[str], List[str], Dict[str, Detections]]:
    """
    Loads YOLO annotations and returns class names, images,
        and their corresponding detections.

    Args:
        images_directory_path (str): The path to the directory containing the images.
        annotations_directory_path (str): The path to the directory
            containing the YOLO annotation files.
        data_yaml_path (str): The path to the data
            YAML file containing class information.
        force_masks (bool): If True, forces masks to be loaded
            for all annotations, regardless of whether they are present.
        is_obb (bool): If True, loads the annotations in OBB format.
            OBB annotations are defined as `[class_id, x, y, x, y, x, y, x, y]`,
            where pairs of [x, y] are box corners.

    Returns:
        Tuple[List[str], List[str], Dict[str, Detections]]:
            A tuple containing a list of class names, a dictionary with
            image names as keys and images as values, and a dictionary
            with image names as keys and corresponding Detections instances as values.
    """
    image_paths = [
        str(path)
        for path in list_files_with_extensions(
            directory=images_directory_path, extensions=["jpg", "jpeg", "png", "tif"]
        )
    ]
    
    classes = _extract_class_names(file_path=data_yaml_path)
    annotations = {}

    for image_path in image_paths:
        image_stem = Path(image_path).stem
        annotation_path = os.path.join(annotations_directory_path, f"{image_stem}.txt")
        if not os.path.exists(annotation_path):
            annotations[image_path] = Detections.empty()
            continue

        image = cv2.imread(image_path)
        lines = read_txt_file(file_path=annotation_path, skip_empty=True)
        h, w, _ = image.shape
        resolution_wh = (w, h)

        def _with_mask(lines: List[str]) -> bool:
            return any([len(line.split()) > 5 for line in lines])
        with_masks = _with_mask(lines=lines)
        with_masks = force_masks if force_masks else with_masks
        annotation = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=resolution_wh,
            with_masks=with_masks,
            is_obb=is_obb,
        )
        annotations[image_path] = annotation
    return classes, image_paths, annotations


class DetectionDataset(SVDetectionDataset):
    @classmethod
    def from_yolo(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        data_yaml_path: str,
        force_masks: bool = False,
        is_obb: bool = False,
    ) -> "DetectionDataset":
        """
        Creates a Dataset instance from YOLO formatted data.

        Args:
            images_directory_path (str): The path to the
                directory containing the images.
            annotations_directory_path (str): The path to the directory
                containing the YOLO annotation files.
            data_yaml_path (str): The path to the data
                YAML file containing class information.
            force_masks (bool): If True, forces
                masks to be loaded for all annotations,
                regardless of whether they are present.
            is_obb (bool): If True, loads the annotations in OBB format.
                OBB annotations are defined as `[class_id, x, y, x, y, x, y, x, y]`,
                where pairs of [x, y] are box corners.

        Returns:
            DetectionDataset: A DetectionDataset instance
                containing the loaded images and annotations.

        Examples:
            ```python
            import roboflow
            from roboflow import Roboflow
            import supervision as sv

            roboflow.login()
            rf = Roboflow()

            project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            dataset = project.version(PROJECT_VERSION).download("yolov5")

            ds = sv.DetectionDataset.from_yolo(
                images_directory_path=f"{dataset.location}/train/images",
                annotations_directory_path=f"{dataset.location}/train/labels",
                data_yaml_path=f"{dataset.location}/data.yaml"
            )

            ds.classes
            # ['dog', 'person']
            ```
        """
        classes, image_paths, annotations = load_yolo_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path,
            force_masks=force_masks,
            is_obb=is_obb,
        )
        return DetectionDataset(
            classes=classes, images=image_paths, annotations=annotations
        )