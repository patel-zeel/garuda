from jaxtyping import jaxtyped
from beartype import beartype
from beartype.typing import Sequence, Tuple

from garuda.box import BB

@jaxtyped(typechecker=beartype)
def possible_intersection(obb1: BB, obb2: BB) -> bool:
    max_lon1 = obb1.max_lon
    min_lon1 = obb1.min_lon
    max_lat1 = obb1.max_lat
    min_lat1 = obb1.min_lat
    
    max_lon2 = obb2.max_lon
    min_lon2 = obb2.min_lon
    max_lat2 = obb2.max_lat
    min_lat2 = obb2.min_lat
    
    if (max_lat1 < min_lat2) or (min_lat1 > max_lat2):
        return False
    if (max_lon1 < min_lon2) or (min_lon1 > max_lon2):
        return False
    return True

@jaxtyped(typechecker=beartype)
def deduplicate(labels: Sequence[BB], ioa_threshold: float, iou_threshold: float) -> Tuple[Sequence[BB], float, float]:
    pass