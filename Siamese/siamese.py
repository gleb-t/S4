import argparse
import csv
import datetime
import json
import logging
import os
import pickle
import platform
import random
import shutil

# Import the annoying libs that need global configs:
import keras
import matplotlib

matplotlib.use('Agg')

# Import the rest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pyplant
from beautifultable import beautifultable

from Lib.haikunator import Haikunator
from PythonExtras import numpy_extras as npe, config_tools, keras_extras, file_tools, \
    logging_tools, patching_tools, iterator_tools
from PythonExtras.Normalizer import Normalizer
from PythonExtras.KerasBatchedTrainer import KerasBatchedTrainer
from PythonExtras.KerasCheckpointCallback import KerasCheckpointCallback
from Siamese.layers import *
from Siamese.data_types import *
from Siamese.config import SiameseConfig
from Siamese.feature_processing import EnsembleFeatureMetadata, sample_random_support_sets
from Siamese.vgg_keras import vgg_preprocess
from Siamese.data_loading import load_ensemble_member_metadata, load_member_volume, EnsembleMemberMetadata
from Siamese.feature_matching import compute_feature_matches as compute_feature_matches_func
from Siamese.coord_mapping import compute_coord_map, compute_patch_shape, map_support_sets, map_coords
from Siamese.ranking_metrics import compute_query_ranking_metrics, aggregate_metrics_table

TItem = TypeVar('TItem')


class DataPointDesc:

    def __init__(self, index: int, descLeft: DataPointPatchDesc, descRight: DataPointPatchDesc, isSame: bool):

        assert index == descLeft.index == descRight.index

        self.index = index
        self.descLeft = descLeft
        self.descRight = descRight
        self.isSame = isSame


def sample_subintervals(seq: Sequence[Any], n: int, l: int) -> Generator[Sequence[Any], None, None]:
    """
    Sample 'n' non-overlapping intervals of length 'l' from the given sequence.

    Credit:
    https://stackoverflow.com/questions/18641272/n-random-contiguous-and-non-overlapping-subsequences-each-of-length
    """
    seqLength = len(seq)
    if (n * l) > seqLength:
        # Impossible to sample given the configuration.
        return

    indices = range(seqLength - (l - 1) * n)
    offset = 0
    for i in sorted(random.sample(indices, n)):
        i += offset
        yield seq[i:i+l]
        offset += l - 1


@pyplant.ReactorFunc
def load_member_metadata(pipe: pyplant.Pipework, config: SiameseConfig):

    ensembleMetadata = load_ensemble_member_metadata(config.dataPath,
                                                     config.downsampleFactor,
                                                     config.volumeCrop)

    shapeMin, shapeMax = ensembleMetadata.get_shape_min_max()
    shapeOrigMin, shapeOrigMax = ensembleMetadata.get_orig_shape_min_max()
    attrNumber = ensembleMetadata.get_attr_number()
    # Since we now have the metadata, we can generate the concrete patch shape.
    patchShape = compute_patch_shape(config.patchShape, shapeMin)

    # A sanity check: at least one patch should fit into all members.
    if any((shapeMin[dim] < (config.patchShape[dim] or shapeMin[dim]) for dim in range(SHAPE_LEN_NA))):
        raise RuntimeError("The patch shape {} is too large for the min shape of {}.".format(config.patchShape, shapeMin))

    axisMaps = compute_coord_map(shapeOrigMax, config.volumeCrop, config.downsampleFactor)

    pipe.send('ensemble-metadata', ensembleMetadata)
    pipe.send('data-coord-maps', axisMaps, pyplant.Ingredient.Type.object)
    pipe.send('patch-shape', patchShape)
    pipe.send('attr-number', attrNumber)


@pyplant.ReactorFunc
def prepare_feature_metadata(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)

    dataCoordMaps = yield pipe.receive('data-coord-maps')  # type: List[CoordMap]
    ensembleMetadata = yield pipe.receive('ensemble-metadata')  # type: EnsembleMemberMetadata
    memberNames = ensembleMetadata.memberNames

    # Load the metadata describing which ensemble member has which features.
    metaPath = config.featureMetadataPath or os.path.join(os.path.dirname(config.dataPath), 'feature-metadata.json')
    if os.path.exists(metaPath):
        metadata = EnsembleFeatureMetadata.load_from_json(metaPath)
        # Map the coordinates, since we might crop and scale the original volume data.
        metadata.apply_coord_mapping(dataCoordMaps)
        if set(metadata.memberNames) != set(memberNames):
            logger.warning('Feature metadata has fewer/different members than the ensemble data.')
            # All members in the metadata should be in the data. (Important for sampling random support sets.)
            assert len(set(metadata.memberNames) & set(memberNames)) == len(metadata.memberNames)

        # Make sure that the coord mapping was done correctly. (ideally this would be refactored to be in one place)
        for name in metadata.memberNames:
            shapeInFeatureMeta = metadata.memberShapes[name][:SHAPE_LEN_NA]
            shapeInDataMeta = ensembleMetadata.memberShapes[memberNames.index(name)][:SHAPE_LEN_NA]
            assert shapeInFeatureMeta == shapeInDataMeta
    else:
        logging.warning("No feature metadata is present at '{}'. Will skip related computations.".format(metaPath))
        metadata = None

    pipe.send('feature-metadata-prepared', metadata, pyplant.Ingredient.Type.object)


@pyplant.SubreactorFunc
def build_random_support_set_generator(pipe: pyplant.Pipework, config: SiameseConfig):
    featureMetadata = yield pipe.receive('feature-metadata-prepared')  # type: EnsembleFeatureMetadata
    patchShape = yield pipe.receive('patch-shape')  # type: Tuple[int, ...]
    dataCoordMaps = yield pipe.receive('data-coord-maps')  # type: List[CoordMap]

    # Since a subreactor is already a generator, return a nested generator instead.
    def _generator():
        # Can't generate random sets, if we don't have the metadata.
        if featureMetadata is None:
            return

        spatialCoords = config.featureRandomSetSpatialCoords
        coordMaxOffset = config.featureRandomSetSpatialCoordsMaxOffset
        if spatialCoords is None:
            # If spatial coords are not specified, the patch must cover the whole spatial domain. We set coords to zero.
            assert all(s is None for s in config.patchShape[1:])
            spatialCoords = (0,) * (SHAPE_LEN_NA - 1)
            coordMaxOffset = (0,) * (SHAPE_LEN_NA - 1)
        else:
            # Map the spatial coords into our downscaled/cropped coordinates. Add and then drop a dummy frame index.
            dummyFrameValue = dataCoordMaps[0].left
            spatialCoords = map_coords((dummyFrameValue,) + spatialCoords,
                                       patchShape, dataCoordMaps)[1:]
            coordMaxOffset = map_coords((dummyFrameValue,) + coordMaxOffset,
                                        patchShape, dataCoordMaps)[1:]

        # Use a separate random generator so that we can control the seed and make the sets reproducible.
        randomEngine = random.Random(config.featureRandomSetSeed)

        for iSample, sampleDesc in enumerate(config.featureRandomSetSamples):
            yield from sample_random_support_sets(iSample, sampleDesc, featureMetadata, patchShape,
                                                  spatialCoords=spatialCoords, spatialCoordsMaxOffset=coordMaxOffset,
                                                  randomEngine=randomEngine)

    return _generator()


@pyplant.ReactorFunc
def prepare_support_sets_and_feature_metadata(pipe: pyplant.Pipework, config: SiameseConfig):
    """
    Since the original volume data can be cropped and scaled,
    the coordinates in the support set specification need to be adjusted accordingly.
    """

    dataCoordMaps = yield pipe.receive('data-coord-maps')  # type: List[CoordMap]

    # Build set desc objects. Assign set indices. Map the coordinates of the manually specified support sets.
    manualSets = []
    for iSet, patchList in enumerate(config.featureSupportSets):
        manualSets.append(SupportSetDesc(iSet, patchList))
    preparedSets = map_support_sets(manualSets, dataCoordMaps, config.patchShape)

    # Add the randomly generated sets.
    manualSetNumber = len(preparedSets)
    randomSetGenerator = yield from build_random_support_set_generator(pipe, config)
    for iRandomSet, supportSetDesc in enumerate(randomSetGenerator):
        supportSetDesc.index = manualSetNumber + iRandomSet
        preparedSets.append(supportSetDesc)  # Random support sets are already using mapped coordinates.

    pipe.send('support-sets-prepared', preparedSets, pyplant.Ingredient.Type.object)


@pyplant.ReactorFunc
def write_support_sets(pipe: pyplant.Pipework, config: SiameseConfig):
    supportSets = yield pipe.receive('support-sets-prepared')  # type: List[SupportSetDesc]

    with open(os.path.join(config.outputDirPath, 'support-sets.csv'), 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['index', 'is-random', 'sample-index', 'member-name', 'coords', 'is-positive'])
        for iSet, supportSet in enumerate(supportSets):
            for iPatch, (patchDesc, isPositive) in enumerate(supportSet.patches):
                writer.writerow([
                    supportSet.index, supportSet.isRandom, supportSet.randomSampleIndex,
                    patchDesc.memberName, patchDesc.coords, isPositive
                ])


@pyplant.SubreactorFunc
def render_data_examples(dataXLeft: npe.LargeArray, dataXRight: npe.LargeArray, dataY: npe.LargeArray,
                         metadata: List[DataPointDesc], sampleNumber: int, outputPdfPath: str):
    from PythonExtras.iterator_tools import reservoir_sample
    from matplotlib.backends.backend_pdf import PdfPages

    # # Assuming a 2D dataset here.
    # raise NotImplementedError("Function needs to be adapted to 3D volumes.")

    frameNumber = dataXLeft.shape[2]
    # In case we have very little data.
    sampleNumber = min(sampleNumber, dataXLeft.shape[0])

    if sampleNumber == 0:
        return

    with PdfPages(outputPdfPath) as pdf:
        textFilepath = os.path.join(os.path.dirname(outputPdfPath),
                                    os.path.splitext(os.path.basename(outputPdfPath))[0] + '.txt')

        with open(textFilepath, 'w') as metadataFile:
            sampleIndices = list(reservoir_sample(iter(range(dataXLeft.shape[0])), sampleNumber))
            sampleIndices = list(sorted(sampleIndices))  # HDF arrays require sorted indices.
            trainXLeftSample = dataXLeft[sampleIndices]
            trainXRightSample = dataXRight[sampleIndices]
            trainYSample = dataY[sampleIndices]

            valueMin = min(np.min(trainXLeftSample), np.min(trainXRightSample))
            valueMax = max(np.max(trainXLeftSample), np.max(trainXRightSample))

            for iSample, iData in zip(range(sampleNumber), sampleIndices):

                fig = plt.figure()

                def _draw_frame(axIndex: int, frameImage: np.ndarray):
                    ax = fig.add_subplot(shotNumber + 1, frameNumber, axIndex)
                    ax.imshow(frameImage.transpose(), cmap='viridis', vmin=valueMin, vmax=valueMax)
                    ax.set_axis_off()

                # We show only one random Z-slice for each sample.
                z = random.randrange(0, dataXLeft.shape[3])
                for f in range(frameNumber):

                    shotNumber = dataXLeft.shape[1]  # The shot number is constant during training.
                    for iShot in range(shotNumber):
                        _draw_frame(iShot * frameNumber + f + 1, trainXLeftSample[iSample, iShot, f, z, ..., 0])

                    _draw_frame(shotNumber * frameNumber + f + 1, trainXRightSample[iSample, 0, f, z, ..., 0])

                    yValue = int(trainYSample[iSample])

                leftMembers = [p.memberName for p in metadata[iData].descLeft.patches]
                rightMember = metadata[iData].descRight.patches[0].memberName

                metadataFile.write("======= Sample {:02d} =======\n".format(iSample + 1))
                metadataFile.write("Min: {:6f} Max: {:6f} Label: {}\n".format(valueMin, valueMax, yValue))
                metadataFile.write("==== Left ====\n")
                for m in leftMembers:
                    metadataFile.write(m + "\n")
                metadataFile.write("==== Right ===\n")
                metadataFile.write(rightMember + "\n")
                metadataFile.write("\n")

                fig.suptitle("Z: {} Min: {:6f} Max: {:6f} Label: {}".format(z, valueMin, valueMax, yValue))

                pdf.savefig(fig)
                plt.close(fig)

    yield


@pyplant.SubreactorFunc
def allocate_training_data_arrays(pipe: pyplant.Pipework, config: SiameseConfig,
                                  patchShape: Tuple[int, ...], attrNumber: int) -> Generator[None, None, Tuple[npe.LargeArray, ...]]:
    logger = logging.getLogger(config.loggerName)

    trainPointNumber = int(config.dataPointNumber * (1.0 - config.testSplitRatio))
    testPointNumber = config.dataPointNumber - trainPointNumber
    dataSizeFlat = config.dataPointNumber * int(npe.multiply(patchShape) + 1) * attrNumber * \
                   (config.shotNumber + 1) * np.dtype(config.dtypeModel).itemsize

    logger.info("Estimated data size: {}".format(logging_tools.format_large_number(dataSizeFlat)))

    patchShapeWithAttr = (*patchShape, attrNumber)
    trainXLeft = pipe.allocate('train-x-left', pyplant.specs.HdfArraySpec,
                               shape=(trainPointNumber, config.shotNumber, *patchShapeWithAttr), dtype=config.dtypeModel)
    trainXRight = pipe.allocate('train-x-right', pyplant.specs.HdfArraySpec,
                                shape=(trainPointNumber, 1, *patchShapeWithAttr), dtype=config.dtypeModel)
    trainY = pipe.allocate('train-y', pyplant.specs.HdfArraySpec,
                           shape=(trainPointNumber, 1), dtype=config.dtypeModel)

    testXLeft = pipe.allocate('test-x-left', pyplant.specs.HdfArraySpec,
                              shape=(testPointNumber, config.shotNumber, *patchShapeWithAttr), dtype=config.dtypeModel)
    testXRight = pipe.allocate('test-x-right', pyplant.specs.HdfArraySpec,
                               shape=(testPointNumber, 1, *patchShapeWithAttr), dtype=config.dtypeModel)
    testY = pipe.allocate('test-y', pyplant.specs.HdfArraySpec,
                          shape=(testPointNumber, 1), dtype=config.dtypeModel)

    yield
    return trainXLeft, trainXRight, trainY, testXLeft, testXRight, testY


@pyplant.ReactorFunc
def collect_training_data(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)

    ensembleMetadata = yield pipe.receive('ensemble-metadata')  # type: EnsembleMemberMetadata
    patchShape = yield pipe.receive('patch-shape')  # type: TupleInt
    attrNumber = ensembleMetadata.get_attr_number()
    isSingleMember = ensembleMetadata.is_single_member()
    memberNames = ensembleMetadata.memberNames
    memberNumber = len(memberNames)

    ndim = SHAPE_LEN_NA
    offsetZ = 0 if ensembleMetadata.is_2d() else config.patchMaxOffsetSpace
    offsetMax = (config.patchMaxOffsetTime, offsetZ, config.patchMaxOffsetSpace, config.patchMaxOffsetSpace)

    logger.info("Starting training data collection. Patch shape: {} Max offset: {}".format(patchShape, offsetMax))
    if isSingleMember:
        logger.info("Sampling in single member mode.")

    trainXLeft, trainXRight, trainY, testXLeft, testXRight, testY = \
        yield from allocate_training_data_arrays(pipe, config, patchShape, attrNumber)

    # First, we only compute the metadata and do not extract the patches, to do it more efficiently later.
    pointDescListLeft = []  # type: List[DataPointPatchDesc]
    pointDescListRight = []  # type: List[DataPointPatchDesc]
    isSameList = []  # type: List[Tuple[int, bool]]

    for iPoint in range(config.dataPointNumber):
        memberIndexRight = random.randrange(0, memberNumber)

        isSame = random.randint(0, 1) == 0
        # --- Decide from which member the left shots come from. Positive point = same members, else = different.
        # If we only have a single member, always pick it, only patch locations will be different.
        if isSame or isSingleMember:
            memberIndexLeft = memberIndexRight
        else:
            # Make sure that members cannot be accidentally the same.
            memberIndexLeft = random.randrange(0, memberNumber - 1)
            if memberIndexLeft >= memberIndexRight:
                memberIndexLeft += 1

        memberShapeLeft = ensembleMetadata.memberShapes[memberIndexLeft]
        memberShapeRight = ensembleMetadata.memberShapes[memberIndexRight]

        #  --- Choose the location of the right shot.
        # Add '+ 1' to make sure that when patch size == volume size, there is a valid sample (zero index).
        patchCoordsRight = sample_coords(memberShapeRight, patchShape)
        pointDescRight = DataPointPatchDesc(
            index=iPoint,
            memberName=memberNames[memberIndexRight],
            patches=[PatchDesc(memberNames[memberIndexRight], patchCoordsRight)]
        )

        # --- Choose the location of the left shots.
        #     (If the grouped shots setting is enabled, all patches are sampled around it.)
        commonCoordsLeft = sample_coords(memberShapeLeft, patchShape)
        # todo When we run in single member mode, we'll probably need to make sure we don't sample near the right patch.

        # -- Figure out the patch locations for the 'left' side of the data point.
        pointDescLeft = DataPointPatchDesc(
            index=iPoint,
            memberName=memberNames[memberIndexLeft],
            patches=[]
        )
        for iShot in range(config.shotNumber):
            # noinspection DuplicatedCode
            if isSame or config.alignDifferentMembers:
                # --- Sample from around the same location as the right shot.
                # Do not avoid patch overlap. Just uniformly sample the whole valid range.
                # Clamp the sampling range with max/min to make sure we're not sampling outside the data.
                c = sample_coords_near_patch(memberShapeLeft, patchShape, patchCoordsRight, offsetMax)
            elif config.groupLeftShots:
                # --- Chose some different location (already done) and sample around it.
                c = sample_coords_near_patch(memberShapeLeft, patchShape, commonCoordsLeft, offsetMax)

            else:
                # --- Here we don't care about left patches' positions, just sample all over the member, independently.
                c = sample_coords(memberShapeLeft, patchShape)

            pointDescLeft.patches.append(PatchDesc(memberNames[memberIndexLeft], c))

        # Store the descriptions of the data point.
        pointDescListLeft.append(pointDescLeft)
        pointDescListRight.append(pointDescRight)
        # Technically, we don't have to store point indices (they are ordered), but do it anyway to avoid bugs.
        isSameList.append((iPoint, isSame))

        if iPoint % 1000 == 0:
            progress = (iPoint + 1) / config.dataPointNumber * 100
            logger.info("Generating training data descriptors ... {:.1f}%.".format(progress),
                        extra={'throttlingId': 'collect_training_data'})

    dataPointMetadata = [DataPointDesc(index, left, right, isSame) for left, right, (index, isSame) in
                         zip(pointDescListLeft, pointDescListRight, isSameList)]
    trainMetadata = dataPointMetadata[:trainY.shape[0]]
    testMetadata = dataPointMetadata[trainY.shape[0]:]

    pointsLeftGrouped = iterator_tools.group_by(pointDescListLeft, lambda d: d.memberName)
    pointsRightGrouped = iterator_tools.group_by(pointDescListRight, lambda d: d.memberName)

    logger.info("Loading the left-side training data ...")
    yield from load_training_patches_from_member(pipe, config, pointsLeftGrouped, trainXLeft, testXLeft)
    logger.info("Loading the right-side training data ...")
    yield from load_training_patches_from_member(pipe, config, pointsRightGrouped, trainXRight, testXRight)

    # Prepare the Y data.
    for pointIndex, isSame in isSameList:
        dataY = int(isSame)
        if pointIndex < trainY.shape[0]:
            trainY[pointIndex] = dataY
        else:
            testPointIndex = pointIndex - trainY.shape[0]
            testY[testPointIndex] = dataY

    logger.info("Normalizing the input data.")
    if attrNumber > 1:
        raise NotImplementedError("If we start working with multivar data, we need to normalize attributes separately.")
    normalizerX = Normalizer()
    if config.normalizerCoefs is None:
        normalizerX.fit_and_scale_batched(trainXLeft, axis=None, batchSizeFlat=int(1e9), printFn=logger.info)
    else:
        normalizerX.set_coefs(trainXLeft.dtype, trainXLeft.ndim, config.normalizerCoefs)
        normalizerX.scale_batched(trainXLeft, batchSizeFlat=int(1e9), printFn=logger.info)

    normalizerX.scale_batched(trainXRight, batchSizeFlat=int(1e9), printFn=logger.info)
    normalizerX.scale_batched(testXLeft, batchSizeFlat=int(1e9), printFn=logger.info)
    normalizerX.scale_batched(testXRight, batchSizeFlat=int(1e9), printFn=logger.info)

    # Train on noise. For debugging purposes.
    # trainXRight = np.random.normal(np.mean(trainXRight), np.std(trainXRight), size=trainXRight.shape)

    pipe.send('normalizer', normalizerX, pyplant.Ingredient.Type.object)
    pipe.send('train-data-metadata', trainMetadata)
    pipe.send('test-data-metadata', testMetadata)
    pipe.send('train-x-left', trainXLeft)
    pipe.send('train-x-right', trainXRight)
    pipe.send('train-y', trainY)
    pipe.send('test-x-left', testXLeft)
    pipe.send('test-x-right', testXRight)
    pipe.send('test-y', testY)


def sample_coords_near_patch(memberShapeLeft: TupleInt, patchShape: TupleInt,
                             commonCoordsLeft: TupleInt, offsetMax: TupleInt) -> TupleInt:

    return tuple(random.randrange(max(0, commonCoordsLeft[dim] - patchShape[dim] - offsetMax[dim]),
                                  min(memberShapeLeft[dim] - patchShape[dim],
                                      commonCoordsLeft[dim] + patchShape[dim] + offsetMax[dim]) + 1)
                 for dim in range(SHAPE_LEN_NA))


def sample_coords(memberShapeLeft: TupleInt, patchShape: TupleInt) -> TupleInt:
    return tuple(random.randrange(0, memberShapeLeft[dim] - patchShape[dim] + 1)
                 for dim in range(SHAPE_LEN_NA))


@pyplant.SubreactorFunc
def load_training_patches_from_member(pipe: pyplant.Pipework, config: SiameseConfig,
                                      pointsGroupedByMember: Iterable[Tuple[str, Iterable[DataPointPatchDesc]]],
                                      trainX: np.ndarray, testX: np.ndarray):
    """
    Given descriptions of training patches grouped by which member they come from,
    load the appropriate data into the training data arrays.
    """
    logger = logging.getLogger(config.loggerName)
    # Needed for progress reporting.
    ensembleMetadata = yield pipe.receive('ensemble-metadata')  # type: EnsembleMemberMetadata
    for iMember, (memberName, pointDescGroup) in enumerate(pointsGroupedByMember):
        logger.info("Loading training data ... {:.1f}%.".format(iMember / len(ensembleMetadata.memberNames) * 100),
                    extra={'throttle': True})
        memberData = load_member_volume(config, memberName)

        for pointDesc in pointDescGroup:  # type: DataPointPatchDesc
            assert pointDesc.memberName == memberName  # Patches must be grouped by member.

            patchShapeWithAttr = trainX.shape[2:]  # First two dims are 'batch' and 'patch/shot'
            patchShapeNoAttr = patchShapeWithAttr[:-1]
            dataPoint = np.empty((len(pointDesc.patches), *patchShapeWithAttr), dtype=config.dtypeModel)

            # Cut out each patch.
            for iPatch, patchDesc in enumerate(pointDesc.patches):
                # The patch extraction method will slice first N dimensions, and fetch all the attributes.
                dataPoint[iPatch, ...] = patching_tools.get_patch_from_volume(memberData, patchDesc.coords, patchShapeNoAttr)

            # Data point axes: [patches, time, z, y, x, attr]
            if config.augmentDataFlip:
                # Flip the data randomly.
                if not config.augmentPatchesSeparately:
                    # Flip (or not) all the patches together.
                    if random.randint(0, 1) == 0:
                        dataPoint = dataPoint[:, :, :, ::-1, :, :]
                    if random.randint(0, 1) == 0:
                        dataPoint = dataPoint[:, :, :, :, ::-1, :]
                else:
                    # Decide individually, which patches to flip.
                    # Along Y.
                    choices = np.random.randint(0, 1 + 1, dataPoint.shape[0])
                    dataPoint = np.asarray([patch if choice == 0 else patch[:, :, ::-1, :, :]
                                            for choice, patch in zip(choices, dataPoint)])
                    # Along X.
                    choices = np.random.randint(0, 1 + 1, dataPoint.shape[0])
                    dataPoint = np.asarray([patch if choice == 0 else patch[:, :, :, ::-1, :]
                                            for choice, patch in zip(choices, dataPoint)])

            if config.augmentDataMul:
                # Multiply data randomly.
                if not config.augmentPatchesSeparately:
                    # Scale all patches together.
                    coef = random.uniform(0.5, 1.5)
                else:
                    # Generate a unique coefficient for each patch.
                    coef = np.random.uniform(0.5, 1.5, (dataPoint.shape[0], 1, 1, 1, 1, 1))
                dataPoint = dataPoint * coef

            # Write out the data point. Split into train and test appropriately.
            if pointDesc.index < trainX.shape[0]:
                trainX[pointDesc.index] = dataPoint
            else:
                iTestPoint = pointDesc.index - trainX.shape[0]
                testX[iTestPoint] = dataPoint


@pyplant.ReactorFunc
def train_model(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)

    patchShape = yield pipe.receive('patch-shape')  # type: TupleInt
    attrNumber = yield pipe.receive('attr-number')  # type: int

    trainDataMetadata = yield pipe.receive('train-data-metadata')  # type: List[DataPointDesc]
    testDataMetadata = yield pipe.receive('test-data-metadata')  # type: List[DataPointDesc]
    trainXLeft = yield pipe.receive('train-x-left')    # type: npe.LargeArray
    trainXRight = yield pipe.receive('train-x-right')  # type: npe.LargeArray
    trainY = yield pipe.receive('train-y')             # type: npe.LargeArray
    testXLeft = yield pipe.receive('test-x-left')      # type: npe.LargeArray
    testXRight = yield pipe.receive('test-x-right')    # type: npe.LargeArray
    testY = yield pipe.receive('test-y')               # type: npe.LargeArray

    if config.checkpointToLoad is None:
        logger.info("Building a model.")
        # noinspection PyTupleAssignmentBalance
        model, encoder, classifier = yield from build_model(config, patchShape, attrNumber)
    else:
        logger.info("Loading the model checkpoint from '{}'.".format(config.checkpointToLoad))
        model = keras.models.load_model(config.checkpointToLoad,
                                        custom_objects=CUSTOM_KERAS_LAYERS)  # type: keras.models.Sequential
        # Old models don't have the attribute dimensions, drop it if needed for backward compatibility.
        if config.backCompDropAttr:
            # noinspection PyTypeChecker
            model = yield from edit_model_drop_attr_dim(model, config, patchShape, attrNumber)  # type: keras.models.Model

        encoder = keras_extras.get_layer_recursive(model, 'encoder')
        classifier = keras_extras.get_layer_recursive(model, 'classifier')

    encodingLayer = encoder.get_layer('encoder_out')  # type: keras.layers.Dense
    encodingShape = encodingLayer.output_shape[1:]  # The first dimension is batch.

    encoder.summary(print_fn=logger.info)
    classifier.summary(print_fn=logger.info)
    model.summary(print_fn=logger.info)

    loggerCallback = _get_epoch_logger_callback(config.epochNumber, printFn=logger.info)
    # tbCallback = keras.callbacks.TensorBoard(config.tensorboardDirPath, histogram_freq=1, write_grads=True)
    earlyStoppingCallback = keras.callbacks.EarlyStopping(patience=config.patience)
    checkpointCallback = KerasCheckpointCallback(config.checkpointDirPath,
                                                 minIntervalMinutes=config.checkpointIntervalMinutes,
                                                 metric='val_loss',
                                                 metricDelta=0.0,
                                                 printFunc=logger.info)

    # Keep the debug output here to be as close to model training as possible.
    logger.info("Dumping some samples of the training data.")

    yield from render_data_examples(trainXLeft, trainXRight, trainY, trainDataMetadata, 20,
                                    outputPdfPath=os.path.join(config.outputDirPath, 'train-data.pdf'))
    yield from render_data_examples(testXLeft, testXRight, testY, testDataMetadata, 20,
                                    outputPdfPath=os.path.join(config.outputDirPath, 'test-data.pdf'))

    # If we're just loading a checkpoint, we don't need to train at all.
    history = None
    if config.epochNumber > 0:
        macrobatchSize = int(config.trainMacrobatchSizeBytes / (npe.compute_byte_size(trainXLeft[0]) * 2))

        if config.useCustomTrainer:
            trainer = KerasBatchedTrainer(
                model,
                [trainXLeft, trainXRight], [trainY],
                [testXLeft, testXRight], [testY],
                macrobatchSize=macrobatchSize,
                minibatchSize=config.batchSize,
                minibatchSizeEval=config.batchSizePredict,
                macrobatchPreprocess=None if config.vggTunedMode is None else vgg_preprocess
            )

            history = trainer.train(
                epochNumber=config.epochNumber,
                callbacks=[earlyStoppingCallback, loggerCallback, checkpointCallback],
                printFunc=logger.info
            )
        else:
            history = model.fit(x=[trainXLeft[...], trainXRight[...]], y=[trainY[...]],  # Load into RAM.
                                validation_data=([testXLeft[...], testXRight[...]], [testY[...]]),
                                batch_size=config.batchSize, epochs=config.epochNumber,
                                # callbacks=[earlyStoppingCallback, loggerCallback, tbCallback, checkpointCallback],
                                callbacks=[earlyStoppingCallback, loggerCallback, checkpointCallback],
                                verbose=False)

        # Load the best checkpoint after training.
        # noinspection PyTypeChecker
        model = checkpointCallback.load_best_model(customObjects=CUSTOM_KERAS_LAYERS)

    pipe.send('model-full', model, pyplant.specs.KerasModelSpec)
    pipe.send('model-encoding-shape', encodingShape, pyplant.Ingredient.Type.simple)
    pipe.send('training-history', history, pyplant.Ingredient.Type.object)


@pyplant.SubreactorFunc
def build_model(config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber: int):
    assert len(patchShape) == SHAPE_LEN_NA  # Patch shape doesn't include attributes.
    patchShapeWithAttr = patchShape + (attrNumber,)

    shapeLeft = (config.shotNumber, *patchShapeWithAttr)
    shapeRight = (1, *patchShapeWithAttr)

    if config.vggTunedMode:  # todo Ugly VGG workaround.
        shapeLeft = (config.shotNumber, *patchShapeWithAttr[2:4], 3)  # Drop time, Z and attr. Add RGB.
        shapeRight = (1, *patchShapeWithAttr[2:4], 3)

    inputLeft = keras.layers.Input(shapeLeft, name='input-left')
    inputRight = keras.layers.Input(shapeRight, name='input-right')

    encoderCtor = config.encoderCtor  # type: Callable[[SiameseConfig, TupleInt], keras.models.Sequential]
    encoder = encoderCtor(config, patchShapeWithAttr)
    encoder.name = 'encoder'

    supportPatches = SplitLayer(axis=1, length=config.shotNumber)(inputLeft)
    supportPatches = supportPatches if config.shotNumber > 1 else [supportPatches]  # Force to be a list.
    encodingsLeft = [encoder(s) for s in supportPatches]
    # encodingLeftSum = keras.layers.Add()(encodingsLeft)
    # encodingLeftAvg = MulConstLayer(1 / config.shotNumber)(encodingLeftSum)
    encodingRight = encoder(SplitLayer(axis=1, length=1)(inputRight))  # We only have one slice on the right.

    distanceLayer = L1Layer()
    distances = [distanceLayer([enc, encodingRight]) for enc in encodingsLeft]
    if config.shotNumber > 1:
        distancesMean = MulConstLayer(1 / config.shotNumber)(keras.layers.Add()(distances))
    else:
        distancesMean = distances[0]

    classifier = config.classifierCtor(config, distanceLayer.output_shape[1:])
    classifier.name = 'classifier'
    output = classifier(distancesMean)

    model = keras.models.Model(inputs=[inputLeft, inputRight], outputs=output)
    model.compile(keras.optimizers.Adam(lr=config.learningRate), loss='binary_crossentropy', metrics=['accuracy'])

    return model, encoder, classifier


@pyplant.SubreactorFunc
def edit_model_drop_attr_dim(model: keras.models.Model, config: SiameseConfig,
                             patchShape: Tuple[int, ...], attrNumber: int):
    """
    A unfortunate workaround for older models that don't have the attribute dimension.
    (Used only to reproduce older results.)
    Couldn't find a clean way of inserting a reshape layer into an existing non-sequential model,
    so just rebuild the model using the same code.
    """

    # Prepare a new encoder.
    encoderOld = model.get_layer('encoder')
    encoderNew = keras.models.Sequential(name='encoder')
    encoderNew.add(keras.layers.Lambda(lambda x: x[..., 0], input_shape=encoderOld.input_shape[1:] + (1,)))
    for l in encoderOld.layers:
        encoderNew.add(l)

    # Rebuild the model like in 'build_model' but with the new encoder.
    assert len(patchShape) == SHAPE_LEN_NA  # Patch shape doesn't include attributes.
    assert config.shotNumber > 1
    patchShapeWithAttr = patchShape + (attrNumber,)

    inputLeft = keras.layers.Input((config.shotNumber, *patchShapeWithAttr), name='input-left')
    inputRight = keras.layers.Input((1, *patchShapeWithAttr), name='input-right')

    encoder = encoderNew

    supportPatches = SplitLayer(axis=1, length=config.shotNumber)(inputLeft)
    supportPatches = supportPatches if config.shotNumber > 1 else [supportPatches]  # Force to be a list.
    encodingsLeft = [encoder(s) for s in supportPatches]
    encodingRight = encoder(SplitLayer(axis=1, length=1)(inputRight))  # We only have one slice on the right.

    distanceLayer = L1Layer()
    distances = [distanceLayer([enc, encodingRight]) for enc in encodingsLeft]
    distancesMean = MulConstLayer(1 / config.shotNumber)(keras.layers.Add()(distances))

    classifier = model.get_layer('classifier')
    output = classifier(distancesMean)

    modelNew = keras.models.Model(inputs=[inputLeft, inputRight], outputs=output)
    modelNew.compile(keras.optimizers.Adam(lr=config.learningRate), loss='binary_crossentropy', metrics=['accuracy'])

    return modelNew


@pyplant.ReactorFunc
def test_model(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)

    testDataMetadata = yield pipe.receive('test-data-metadata')  # type: List[DataPointDesc]
    testXLeft = yield pipe.receive('test-x-left')    # type: npe.LargeArray
    testXRight = yield pipe.receive('test-x-right')  # type: npe.LargeArray
    testY = yield pipe.receive('test-y')             # type: npe.LargeArray

    model = yield pipe.receive('model-full')  # type: keras.models.Model
    encoder = keras_extras.get_layer_recursive(model, 'encoder')      # type: keras.models.Model

    # Cut out a chunk of test data, too lazy to implement out-of-core testing.
    # Take the first batch.
    batchStart, batchEnd = next(npe.get_batch_indices(testXLeft.shape, testXLeft.dtype, batchSizeFlat=int(5e8)))
    testXLeft = testXLeft[batchStart:batchEnd]
    testXRight = testXRight[batchStart:batchEnd]
    testY = testY[batchStart:batchEnd]
    testDataMetadata = testDataMetadata[batchStart:batchEnd]

    logger.info("Testing the model on a chunk of test data ({} points) .".format(batchEnd - batchStart))

    if config.vggTunedMode:  # todo Annoying workaround.
        testX, testY = vgg_preprocess([testXLeft, testXRight], testY)
        testXLeft = testX[0]
        testXRight = testX[1]

    # Compute the confusion matrix.
    predictionTest = model.predict(x=[testXLeft, testXRight],
                                   batch_size=config.batchSizePredict,
                                   verbose=False)
    mask = (np.round(predictionTest) != testY).flatten()

    formatFn = logging_tools.format_large_number
    logger.info("Total wrong predictions: {} / {}".format(
        formatFn(np.count_nonzero(mask)), formatFn(mask.size),
    ))
    falsePosNumber = np.count_nonzero((testY == 0)[mask])
    falseNegNumber = np.count_nonzero((testY == 1)[mask])
    truePosNumber = np.count_nonzero(testY == 1) - falseNegNumber
    trueNegNumber = np.count_nonzero(testY == 0) - falsePosNumber

    logger.info("Confusion matrix. True: {} {} False: {} {}".format(
        formatFn(trueNegNumber), formatFn(truePosNumber),
        formatFn(falseNegNumber), formatFn(falsePosNumber),
    ))

    # Check model asymmetry (should be ~0).
    predictionTestInverted = model.predict(x=[testXLeft, testXRight],
                                           batch_size=config.batchSizePredict,
                                           verbose=False)
    logger.info("Model asymmetry check: {}".format(np.mean(np.abs(predictionTest - predictionTestInverted))))

    # --- Sparsity checks.
    encodingsTest = encoder.predict(testXLeft[:, 0], batch_size=config.batchSizePredict, verbose=False)
    sparsity = 1.0 - np.count_nonzero(encodingsTest, axis=0) / encodingsTest.shape[0]
    logger.info("Column sparsity: {}".format(sparsity))
    logger.info("Constantly sparse components: {}".format(np.count_nonzero(sparsity == 1.0)))

    # todo Ugly VGG tuning hack.
    if config.vggTunedMode:
        return

    metadata = [m for i, m in enumerate(testDataMetadata) if mask[i]]
    yield from render_data_examples(testXLeft[mask], testXRight[mask], testY[mask], metadata, sampleNumber=20,
                                    outputPdfPath=os.path.join(config.outputDirPath, 'wrong-predictions.pdf'))


@pyplant.ReactorFunc
def write_model(pipe: pyplant.Pipework, config: SiameseConfig):
    model = yield pipe.receive('model-full')  # type: keras.models.Sequential
    normalizer = yield pipe.receive('normalizer')  # type: Normalizer

    model.save(os.path.join(config.outputDirPath, 'model.hdf'))
    with open(os.path.join(config.outputDirPath, 'normalizer.pcl'), 'wb') as file:
        pickle.dump(normalizer, file)


@pyplant.ReactorFunc
def compute_matches(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)

    patchShape = yield pipe.receive('patch-shape')  # type: Tuple
    attrNumber = yield pipe.receive('attr-number')  # type: int
    normalizer = yield pipe.receive('normalizer')  # type: Normalizer
    model = yield pipe.receive('model-full')  # type: keras.models.Sequential
    encoder = keras_extras.get_layer_recursive(model, 'encoder')  # type: keras.models.Sequential
    encodingShape = yield pipe.receive('model-encoding-shape')  # type: Tuple[int, ...]
    supportSets = yield pipe.receive('support-sets-prepared')  # type: List[SupportSetDesc]
    ensembleMetadata = yield pipe.receive('ensemble-metadata')  # type: EnsembleMemberMetadata

    patchShapeWithAttr = patchShape + (attrNumber,)

    targetPatchData = pipe.allocate('target-patch-data', pyplant.specs.HdfArraySpec,
                                    shape=(0,) + patchShapeWithAttr, maxshape=(None,) + patchShapeWithAttr,
                                    dtype=np.uint8)  # type: h5py.Dataset
    targetPatchEncoding = pipe.allocate('target-patch-encoding', pyplant.specs.HdfArraySpec,
                                        shape=(0,) + encodingShape, maxshape=(None,) + encodingShape,
                                        dtype=np.float32)  # type: h5py.Dataset

    assert encodingShape == encoder.output_shape[-1:]

    patchStride = tuple(max(1, int(x / config.matchPatchStrideFactor)) for x in patchShape)

    # We pulled out the feature matching functionality out of the plant
    # so that we can reuse it in the app and some scripts.
    # todo Do we need an external code dependency feature for PyPlant?
    featureMatchesPerMetric, metricNames, supportPatchesRaw, targetPatchDesc = \
        compute_feature_matches_func(
            config, encoder, normalizer, ensembleMetadata.memberNames, patchShape, attrNumber, supportSets,
            patchStride=patchStride,
            outTargetPatchData=targetPatchData,
            outTargetPatchEncoded=targetPatchEncoding,
            logger=logger
        )

    pipe.send('support-patch-data', supportPatchesRaw, pyplant.Ingredient.Type.array)
    pipe.send('target-patch-desc', targetPatchDesc, pyplant.Ingredient.Type.object)
    pipe.send('target-patch-data', targetPatchData, pyplant.specs.HdfArraySpec)
    pipe.send('target-patch-encoding', targetPatchEncoding, pyplant.specs.HdfArraySpec)
    pipe.send('distance-metric-names', metricNames, pyplant.Ingredient.Type.list)
    pipe.send('feature-matches', featureMatchesPerMetric, pyplant.Ingredient.Type.object)


@pyplant.ReactorFunc
def write_matches(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)

    metricNames = yield pipe.receive('distance-metric-names')  # type: List[str]
    featureMatchesPerMetric = yield pipe.receive('feature-matches')  # type: TFeatureMatchesStruct
    supportSets = yield pipe.receive('support-sets-prepared')  # type: List[SupportSetDesc]
    targetPatchDesc = yield pipe.receive('target-patch-desc')  # type: List[PatchDesc]

    for metricName, matches in zip(metricNames, featureMatchesPerMetric):
        outputPath = os.path.join(config.outputDirPath, 'matches-{}.csv'.format(metricName))
        logger.info("Writing feature matches for metric '{}' to file '{}'.".format(metricName, outputPath))
        with open(outputPath, 'w') as file:
            for iSet, supportSet in enumerate(supportSets):
                indicesSorted, distances = matches[iSet]
                for iRank, sliceIndex in enumerate(indicesSorted):
                    patch = targetPatchDesc[sliceIndex]
                    file.write('{}\t{}\t{:.4f}\t{}\t{}\n'.format(
                        iSet, iRank, distances[sliceIndex], patch.memberName, patch.coords)
                    )


@pyplant.ReactorFunc
def compute_ranking_metrics(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)

    metricNames = yield pipe.receive('distance-metric-names')  # type: List[str]
    featureMatchesPerMetric = yield pipe.receive('feature-matches')  # type: TFeatureMatchesStruct

    targetPatchDesc = yield pipe.receive('target-patch-desc')  # type: List[PatchDesc]
    patchShape = yield pipe.receive('patch-shape')  # type: Tuple
    supportSets = yield pipe.receive('support-sets-prepared')  # type: List[SupportSetDesc]
    metadataFull = yield pipe.receive('feature-metadata-prepared')  # type: EnsembleFeatureMetadata

    precisionRanks = [10, 50, 100]
    resultRows = []

    if not metadataFull:
        logger.warning("Skipping feature metric computation because the metadata is not available.")
    else:
        # todo We haven't adapted the code or the metadata format for the patches.
        #      Still, it's helpful for comparison with the old results.
        logger.info("RANKING METRICS ARE NOT IMPLEMENTED FOR PATCHES. ASSUMING THAT FEATURES COVER THE WHOLE SPACE")

        # Loop over different ranking metrics/algorithms.
        for rankingPerSet, metricName in zip(featureMatchesPerMetric, metricNames):
            # Loop over all the support sets (queries).
            for iSet, (supportSet, ranking) in enumerate(zip(supportSets, rankingPerSet)):
                patchIndices, distValues = ranking

                # === Prepare the support set.
                supportMembersPositive = supportSet.get_positive_members()
                supportFeatureNames = metadataFull.get_support_set_features(supportSet, patchShape)
                if len(supportFeatureNames) == 0:
                    logger.warning("WARNING: Support set members specify non-intersecting feature types.")
                    continue

                # === Prepare the patch descriptions.
                patchesSorted = [targetPatchDesc[i] for i in patchIndices]
                # Filter out patches from members that are not labeled in the metadata,
                # because we don't know which features they have.
                patchesSorted = list(filter(lambda p: p.memberName in metadataFull.memberNames, patchesSorted))
                
                # First, compute the ranking metrics considering all the members.
                metricValuesWithQuery = compute_query_ranking_metrics(supportFeatureNames, patchesSorted,
                                                                      metadataFull, patchShape, precisionRanks)
                # Then, filter out patches from members present in the query and remove the members from the metadata.
                # Makes for a more honest comparison.
                patchesSorted = list(filter(lambda p: p.memberName not in supportMembersPositive, patchesSorted))
                metadataFiltered = metadataFull.build_metadata_subset(supportMembersPositive, allowMissing=False)
                metricValuesWithoutQuery = compute_query_ranking_metrics(supportFeatureNames, patchesSorted,
                                                                         metadataFiltered, patchShape, precisionRanks)
                
                # Add a postfix to the metrics to tell them apart.
                metricValuesWithQuery = {k + '-wq': v for k, v in metricValuesWithQuery.items()}
                metricValuesWithoutQuery = {k + '-nq': v for k, v in metricValuesWithoutQuery.items()}

                # Store how many positive/negative patches are in the query.
                patchesDesc = supportSet.get_patch_count_str()
                
                resultRows.append({
                    'Metric': metricName,
                    'Set': iSet + 1,
                    'Feature': ', '.join(supportFeatureNames),
                    'Rand': supportSet.isRandom,
                    'Sample': supportSet.randomSampleIndex,
                    'Patches': patchesDesc,
                    **metricValuesWithQuery,
                    **metricValuesWithoutQuery
                })

    resultsTable = pandas.DataFrame(resultRows)

    pipe.send('ranking-metrics', resultsTable, pyplant.Ingredient.Type.object)


@pyplant.ReactorFunc
def aggregate_ranking_metrics(pipe: pyplant.Pipework, config: SiameseConfig):
    rankingTable = yield pipe.receive('ranking-metrics')  # type: pandas.DataFrame

    if rankingTable.shape[0] > 0:
        # Remove irrelevant columns, filter out non-random support sets, group by sample index and aggregate.
        tableGrouped = rankingTable.drop(['Set'], axis=1)
        # tableGrouped = tableGrouped.sort_values(['Sample'])  # Sort by the sample index for consistent presentation.
        tableGrouped = tableGrouped[tableGrouped.apply(lambda r: r['Rand'], axis=1)]
        tableGrouped = tableGrouped.groupby(['Metric', 'Sample'])

        # The resulting number of groups should be equal to the number of samples (times metric number).
        assert tableGrouped.ngroups == len(config.featureRandomSetSamples) * \
               len(config.simMetricNames) * len(config.aggFuncNames)

        aggregatedTable = aggregate_metrics_table(tableGrouped)
    else:
        aggregatedTable = pandas.DataFrame([])

    pipe.send('ranking-metrics-aggregated', aggregatedTable, pyplant.Ingredient.Type.object)


@pyplant.ReactorFunc
def print_ranking_metrics(pipe: pyplant.Pipework, config: SiameseConfig):
    logger = logging.getLogger(config.loggerName)
    rankingTable = yield pipe.receive('ranking-metrics')  # type: pandas.DataFrame
    rankingTableAggregated = yield pipe.receive('ranking-metrics-aggregated')  # type: pandas.DataFrame

    if rankingTable.shape[0] > 0:
        printedTable = rankingTable.copy(deep=True)  # type: pandas.DataFrame
        printedTable = printedTable[printedTable['Rand'] == False].reset_index()
        printedTable.drop(['Rand', 'Sample'], axis=1)

        # Manually format floating point columns.
        for col, dtype in zip(printedTable.columns, printedTable.dtypes):
            if np.issubdtype(dtype, np.floating):
                printedTable[col] = printedTable[col].map('{:.1f}'.format)
        
        formattedTable = beautifultable.BeautifulTable(max_width=999, default_alignment=beautifultable.enums.ALIGN_RIGHT)
        formattedTable.column_headers = printedTable.columns.to_list()
        formattedTable.set_style(beautifultable.enums.STYLE_COMPACT)

        for i, row in printedTable.iterrows():
            formattedTable.append_row(row.to_list())

        logger.info('\n' + str(formattedTable))

        outCsvPath = os.path.join(config.outputDirPath, 'feature-match-metrics.csv')
        logger.info("Writing metrics to '{}'.".format(outCsvPath))
        rankingTable.to_csv(outCsvPath, sep='\t', index=False)
    else:
        logger.info("No metrics to print.")

    if rankingTableAggregated.shape[0] > 0:
        # todo Copy-pasting for now, will see what to do with it later.
        # Remove irrelevant columns from the printed table.
        printedTable = rankingTableAggregated.drop(['Rand'], axis=1)
        # Manually format floating point columns.
        for col, dtype in zip(printedTable.columns, printedTable.dtypes):
            if np.issubdtype(dtype, np.floating):
                printedTable[col] = printedTable[col].map('{:.1f}'.format)

        formattedTable = beautifultable.BeautifulTable(max_width=999, default_alignment=beautifultable.enums.ALIGN_RIGHT)
        formattedTable.column_headers = printedTable.columns.to_list()
        formattedTable.set_style(beautifultable.enums.STYLE_COMPACT)
        # Have to add row-by-row to the table formatter.
        for i, row in printedTable.iterrows():
            formattedTable.append_row(row.to_list())

        logger.info('\n' + str(formattedTable))

        outCsvPath = os.path.join(config.outputDirPath, 'feature-match-metrics-agg.csv')
        logger.info("Writing aggregated metrics to '{}'.".format(outCsvPath))
        rankingTableAggregated.to_csv(outCsvPath, sep='\t', index=False)
    else:
        logger.info("No aggregated metrics to print.")


@pyplant.ReactorFunc
def write_config(pipe: pyplant.Pipework, config: SiameseConfig):
    # Store the config for future reference.
    with open(os.path.join(config.outputDirPath, 'config.json'), 'w') as f:
        json.dump({k: str(v) for k, v in config.__dict__.items() if not k.startswith('_')}, f, cls=npe.JsonEncoder)


@pyplant.ReactorFunc
def write_patch_encodings(pipe: pyplant.Pipework, config: SiameseConfig):
    """
    Store the encoded patches and their description for future use.
    """

    targetPatchDesc = yield pipe.receive('target-patch-desc')  # type: List[PatchDesc]
    targetPatchEncoding = yield pipe.receive('target-patch-encoding')  # type: h5py.Dataset

    file_tools.create_dir(os.path.join(config.outputDirPath, 'patches'))
    with open(os.path.join(config.outputDirPath, 'patches', 'patch-desc.pcl'), 'wb') as file:
        pickle.dump(targetPatchDesc, file)

    # We assume that all the encodings fit into RAM.
    np.save(os.path.join(config.outputDirPath, 'patches', 'patch-encoding.npy'), targetPatchEncoding[...])


def _get_epoch_logger_callback(epochNumber, printFn: Callable):
    def log_on_epoch_end(epoch, logs):
        if 'time_total' in logs:
            template = 'Epoch: {}/{} Train: loss = {:8f} acc = {:8f} Test: loss = {:8f} acc = {:8f} ' \
                       '[{:.2f} min. = Load: {:.2f} min. Train: {:.2f} min. Eval: {:.2f} min. ' \
                       'Norm: {:.2f} min. Rest: {:.2f} min.]'
            printFn(template.format(epoch, epochNumber, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc'],
                                    logs['time_total'] / 60, logs['time_load'] / 60,
                                    logs['time_train'] / 60, logs['time_val'] / 60,
                                    logs['time_norm'] / 60, logs['time_rest'] / 60
                                    ))
        else:
            template = 'Epoch: {}/{} Train: loss = {:8f} acc = {:8f} Test: loss = {:8f} acc = {:8f} '
            printFn(template.format(epoch, epochNumber, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))

    return keras.callbacks.LambdaCallback(on_epoch_end=log_on_epoch_end)


def prepare_config(configPath, runsetId: str, runId: str, taskNumber: int, taskId: int):
    # Prepare the default config.
    defaultConfig = SiameseConfig()
    defaultConfig.outputDirPath = os.path.expandvars(r'$DEV_OUT_PATH/{}/{}'.format(runsetId, runId))
    defaultConfig.tensorboardDirPath = os.path.expandvars(r'$DEV_OUT_PATH/tensorboard/{}'.format(runId))
    defaultConfig.checkpointDirPath = os.path.join(defaultConfig.outputDirPath, 'checkpoints')

    # Load the config from disk and apply it on top of the default one.
    config = config_tools.parse_and_apply_config_eval(defaultConfig, configPath)

    # If a 'parameter sweep' config is provided, expand it into config instances and choose the one we should run.
    if config_tools.is_parameter_sweep_config(config):
        sweepConfigs = config_tools.expand_parameter_sweep_config(config)
        assert len(sweepConfigs) == taskNumber  # The number of (cluster) tasks should match the config.

        config = sweepConfigs[taskId]

    return config


def main():

    print("Starting.")

    # Parse command-line arguments.
    parser = argparse.ArgumentParser('S4')
    parser.add_argument('--config-path', required=True, type=str)
    parser.add_argument('--runset-id', required=False, type=str)
    parser.add_argument('--run-id', required=False, type=str)
    parser.add_argument('--task-number', required=False, type=int, default=1)
    parser.add_argument('--task-id', required=False, type=int, default=0)

    args = vars(parser.parse_args())
    configPath = args['config_path']
    configName = os.path.splitext(os.path.basename(configPath))[0]

    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    if args['runset_id']:
        runsetId = args['runset_id']
    else:
        runsetId = '{timestamp}_{hostname}_{config}'.format(
            timestamp=timestamp,
            hostname=platform.node().split('.')[0],
            config=configName
        )

    if args['run_id']:
        runId = args['run_id']
    else:
        runNickname = Haikunator().haikunate(token_length=0)
        runId = '{timestamp}-{nickname}_{config}'.format(
            timestamp=timestamp,
            nickname=runNickname,
            config=configName
        )

    config = prepare_config(configPath, runsetId, runId, args['task_number'], args['task_id'])

    # Create the output dir and copy over the config.
    file_tools.create_dir(config.outputDirPath)
    # todo We also need to export the config name (for loading) and make sure that sub-configs are exported (how?).
    #      And, we currently only export the instantiated sweep params in json.
    shutil.copy(configPath, os.path.join(config.outputDirPath, os.path.basename(configPath)))

    logging_tools.configure_logger('', logToStd=True, logLevel=logging.INFO)  # The root logger logs to std.
    logging_tools.configure_logger(config.loggerName, os.path.join(config.outputDirPath, 'log.log'),
                                   logToStd=False, throttle=True)
    logger = logging.getLogger(config.loggerName)
    plantLogger = logging.getLogger('pyplant')
    logging_tools.configure_logger(plantLogger.name, logLevel=logging.INFO, logToStd=False)
    logging_tools.setup_uncaught_exception_report(logger)

    keras_extras.assert_gpu_is_available()

    with pyplant.Plant(os.path.expandvars(r'$DEV_PYPLANT_PATH/siamese'), plantLogger) as plant:
        kerasSpec = pyplant.specs.KerasModelSpec(customLayers=CUSTOM_KERAS_LAYERS)
        hdfSpec = pyplant.specs.HdfArraySpec()
        plant.warehouse.register_ingredient_specs([kerasSpec, hdfSpec])
        plant.add_all_visible_reactors()
        plant.set_config(config)

        logger.info("Starting run '{}'".format(runId))
        plant.run_reactor(write_config)
        plant.run_reactor(write_support_sets)
        plant.run_reactor(test_model)
        plant.run_reactor(write_model)
        plant.run_reactor(print_ranking_metrics)
        plant.run_reactor(write_matches)
        plant.run_reactor(write_patch_encodings)
        # plant.run_reactor(compute_ensemble_distances)
        # plant.run_reactor(render_projection_video)

        plant.print_execution_history(logger.info)

        # Store the ingredients needed to run the feature matching,
        # so that we can run matching on the backend server.
        logger.info("Exporting the ingredients needed for feature matching.")
        exportDir = os.path.join(config.outputDirPath, 'export')
        os.makedirs(exportDir, exist_ok=True)
        pyplant.utils.store_reactor_inputs_to_dir(plant, compute_matches.__name__, exportDir)


if __name__ == '__main__':
    main()
