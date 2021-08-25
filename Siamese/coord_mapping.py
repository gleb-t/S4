from typing import Tuple, Optional, List

from Siamese.data_types import CoordMap, SHAPE_LEN_NA, PatchDesc, SupportSetDesc


def compute_coord_map(shapeOrigMax: Tuple[int, ...], volumeCrop: Tuple[Optional[slice]], downsampleFactor: int):
    """
    Because we crop and downsample the original volumes, we need to map all coordinates appropriately.
    E.g. coordinates specifying support sets or feature metadata describing locations of
    different features in the data.
    """

    # Compute a mapping for each axis: min/max coords included and scaling.
    axisMaps = []
    for dim in range(SHAPE_LEN_NA):
        left, right, scale = 0, shapeOrigMax[dim], 1
        if volumeCrop:
            s = volumeCrop[dim]
            if s is not None:
                left, right = s.start or left, s.stop or right
        if downsampleFactor != 1 and dim not in [0, 1]:  # Assuming 2D here.
            scale = downsampleFactor

        axisMaps.append(CoordMap(left, right, scale))

    return axisMaps


def map_coords(coords: Tuple[int, ...], patchShape: Tuple[int, ...], dataCoordMaps: List[CoordMap]) -> Tuple[int, ...]:
    assert len(coords) == len(patchShape) == len(dataCoordMaps)

    result = []
    for dim in range(SHAPE_LEN_NA):
        if patchShape[dim] is None:
            # If patch size is missing (using the whole axis), set the coords to zero. (convenience feature)
            result.append(0)
            continue

        left, right, scale = dataCoordMaps[dim]
        if coords[dim] >= right or coords[dim] < left:
            raise RuntimeError("Support set patch coords are invalid: {}".format(coords))
        newCoord = int((coords[dim] - left) / scale)
        result.append(newCoord)

    return tuple(result)


def map_support_sets(supportSets: List[SupportSetDesc], dataCoordMaps: List[CoordMap], patchShape: Tuple[int, ...]):
    """
    Apply the axis mapping to each patch in each support set.
    """
    preparedSets = []
    for supportSet in supportSets:
        patches = list((PatchDesc(patch.memberName,
                                  map_coords(patch.coords, patchShape, dataCoordMaps)), isPositive)
                       for patch, isPositive in supportSet.patches)
        preparedSets.append(SupportSetDesc(supportSet.index, patches))

    return preparedSets


def compute_patch_shape(patchShapeConfig: Tuple[Optional[int], ...],
                        volumeShapeMin: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(patchShapeConfig[dim] or volumeShapeMin[dim] for dim in range(SHAPE_LEN_NA))


def compute_patch_number_max(patchShape: Tuple[int, ...], volumeShapeMax: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(volumeShapeMax[dim] // patchShape[dim] for dim in range(len(volumeShapeMax)))