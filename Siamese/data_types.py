from typing import *


class PatchDesc(NamedTuple):
    memberName: str
    coords: Tuple[int, ...]


TSupportSetPatchList = List[Tuple[PatchDesc, bool]]
TFeatureMatchesStruct = List[List[Tuple[List[int], List[float]]]]

# The type typically used for volume shapes. T+ZYX+A
TupleInt = Tuple[int, ...]

SHAPE_LEN = 5                 # Shape length used all over the project. Time + 3D space + Attributes.
SHAPE_LEN_NA = SHAPE_LEN - 1  # Shape length without the attribute dimension.


class RandomSetSampleDesc(NamedTuple):
    sampleSize: int  # How many support sets to generate.
    featureName: str
    patchNumberPos: int
    patchNumberNeg: int


class CoordMap(NamedTuple):
    left: int
    right: int
    scale: int


class DataPointPatchDesc:

    def __init__(self,
                 index: int,
                 memberName: str,
                 patches: List[PatchDesc]):

        self.index = index
        self.memberName = memberName
        self.patches = patches


class SupportSetDesc:

    def __init__(self,
                 index: int,
                 patches: TSupportSetPatchList,
                 isRandom: bool = False,
                 randomSampleIndex: int = 0):
        self.index = index
        self.patches = patches
        self.isRandom = isRandom
        self.randomSampleIndex = randomSampleIndex

    def __repr__(self) -> str:
        return '{}_{}_{}_{}'.format(self.index, self.patches, self.isRandom, self.randomSampleIndex)

    def get_positive_patches(self) -> List[PatchDesc]:
        return [patch for patch, isPositive in self.patches if isPositive]

    def get_positive_members(self) -> List[str]:
        return list(set(patch.memberName for patch in self.get_positive_patches()))

    def get_pos_neg_count(self):
        positiveCount = sum(1 for p, isPositive in self.patches if isPositive)
        negativeCount = len(self.patches) - positiveCount
        return positiveCount, negativeCount

    def get_patch_count_str(self) -> str:
        positiveCount, negativeCount = self.get_pos_neg_count()
        return '{}+{}-'.format(positiveCount, negativeCount)
