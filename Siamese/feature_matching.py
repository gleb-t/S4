import itertools
import logging
from enum import Enum
from typing import *

import h5py
import numpy as np
import ot
import scipy.ndimage
import scipy.stats
import skimage.metrics
import skimage.transform

# Do the annoying global config for Tensorflow before importing keras.
import tensorflow as tf
tfConfig = tf.ConfigProto()
tfConfig.gpu_options.allow_growth = True
session = tf.Session(config=tfConfig)
import keras
keras.backend.set_session(session)

from PythonExtras import logging_tools, patching_tools, numpy_extras as npe
from PythonExtras.Normalizer import Normalizer
from PythonExtras.MultidataPatchExtractor import MultidataPatchExtractor
from PythonExtras.StageTimer import StageTimer
from Siamese.config import SiameseConfig
from Siamese.data_loading import load_member_volume, EnsembleDataLoaderCached
from Siamese.data_types import SupportSetDesc, PatchDesc
from Siamese.vgg_keras import vgg_preprocess


class SimilarityMetricBase:
    """
    A base class representing a method of computing similarity between support sets and target patches.
    For batching reasons, first the supports sets are encoded (if necessary) and stored within the object.
    Then, calls to compute distance to a batch of target patches. Distances are computed from all previously
    encoded support sets.
    """

    def __init__(self, config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber, logger: logging.Logger = None):
        self.config = config
        self.patchShape = patchShape
        self.attrNumber = attrNumber
        self.logger = logging_tools.get_null_logger_if_none(logger)
        self.encodingShape = (0,)
        self.supportSets = None  # type: Optional[List[SupportSetDesc]]
        self.supportPatchesEnc = None  # type: Optional[np.ndarray]
        self.patchBatchEnc = None  # type: Optional[np.ndarray]

    def encode_support_sets(self, supportPatchesRaw: np.ndarray, supportSets: List[SupportSetDesc]) -> np.ndarray:
        self.supportSets = supportSets
        pass

    def compute_patch_dist(self, patchesRaw: Optional[np.ndarray] = None, patchesEnc: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
        """
        :param patchesEnc  Pre-computed encodings can be provided to speed up the computation.

        :return:  Distances to each target patch (ndarray) for each support patch in each set.
        """
        pass


class SimilarityMetricModel(SimilarityMetricBase):

    def __init__(self, config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber: int,
                 normalizer: Normalizer, encoder: keras.models.Sequential,
                 logger: logging.Logger = None):
        super().__init__(config, patchShape, attrNumber, logger)

        self.config = config
        self.normalizer = normalizer
        self.encoder = encoder
        self.encodingShape = self.encoder.output_shape[-1:]
        self.batchSize = config.batchSizePredict

    def encode_support_sets(self, supportPatchesRaw: np.ndarray, supportSets: List[SupportSetDesc]) -> np.ndarray:
        super().encode_support_sets(supportPatchesRaw, supportSets)

        self.supportPatchesEnc = np.zeros(supportPatchesRaw.shape[0:2] + self.encodingShape, dtype=np.float32)

        # Since the support patch array is sparse, we could do two things here:
        # either encode the whole array in a single GPU call (encoding the padding zeros as well),
        # or make a call per set with only the relevant data. We do the latter.
        for iSet, supportSetDesc in enumerate(supportSets):
            patchNumber = len(supportSetDesc.patches)
            inputDataNorm = self.normalizer.scale(supportPatchesRaw[iSet, :patchNumber].astype(np.float32), inPlace=False)

            if self.config.vggTunedMode:  # todo Annoying workaround.
                # Insert a fake batch dimension, and then remove it.
                inputDataNorm = vgg_preprocess([inputDataNorm[np.newaxis, ...]], None)[0][0][0]

            # Compute the patch encoding.
            self.supportPatchesEnc[iSet, :patchNumber] = self.encoder.predict(inputDataNorm, batch_size=patchNumber, verbose=False)

        return self.supportPatchesEnc

    def compute_patch_dist(self, patchesRaw: np.ndarray = None,
                           patchesEnc: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
        if patchesEnc is None:
            # Encode the target patches.
            patchesNorm = self.normalizer.scale(patchesRaw.astype(np.float32), inPlace=False)

            if self.config.vggTunedMode:  # todo Annoying workaround.
                patchesNorm = vgg_preprocess([patchesNorm[np.newaxis, ...]], None)[0][0][0]

            self.patchBatchEnc = self.encoder.predict(patchesNorm, batch_size=self.batchSize, verbose=False)
        else:
            self.patchBatchEnc = patchesEnc

        # Compute the metric.
        distancesPerSetPerPatch = []
        for iSet, supportSet in enumerate(self.supportSets):
            # The model-based distance is an L1 distance between the support patch encoding
            # and the target patch encoding.
            distToSupportPatchModel = []
            for iSupportPatch in range(len(supportSet.patches)):
                # Mean over emb. components. Computes dist to each target patch.
                dist = np.mean(np.abs(self.patchBatchEnc - self.supportPatchesEnc[iSet, iSupportPatch, ...]), axis=1)
                distToSupportPatchModel.append(dist)

            distancesPerSetPerPatch.append(distToSupportPatchModel)

        return distancesPerSetPerPatch


class SimilarityMetricVgg(SimilarityMetricBase):

    def __init__(self, config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber: int,
                 logger: logging.Logger = None):
        from keras.applications.vgg16 import VGG16

        super().__init__(config, patchShape, attrNumber, logger)

        if attrNumber > 1:
            raise NotImplementedError()

        layerName = config.vggPretrainedLayerName
        fullModel = VGG16(weights='imagenet', include_top=True)
        self.vggModel = keras.models.Model(inputs=fullModel.input, outputs=fullModel.get_layer(layerName).output)
        self.vggInputShape = (224, 224)
        # We will flatten all non-batch dimensions, so multiply to compute the flat shape.
        self.encodingShape = (npe.multiply(fullModel.get_layer(layerName).output_shape[1:]),)

    def encode_support_sets(self, supportPatchesRaw: np.ndarray, supportSets: List[SupportSetDesc]) -> np.ndarray:
        super().encode_support_sets(supportPatchesRaw, supportSets)

        self.supportPatchesEnc = np.zeros(supportPatchesRaw.shape[0:2] + self.encodingShape, dtype=np.float32)

        # Since the support patch array is sparse, we could do two things here:
        # either encode the whole array in a single GPU call (encoding the padding zeros as well),
        # or make a call per set with only the relevant data. We do the latter.
        for iSet, supportSetDesc in enumerate(supportSets):
            patchNumber = len(supportSetDesc.patches)
            # Compute the patch encoding.
            self.supportPatchesEnc[iSet, :patchNumber] = self._encode_patches_vgg(
                supportPatchesRaw[iSet, :patchNumber].astype(np.float32)
            )

        return self.supportPatchesEnc

    def compute_patch_dist(self, patchesRaw: np.ndarray = None,
                           patchesEnc: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
        assert patchesRaw is not None  # Don't support pre-encoded patches.

        # Encode the target patches.
        self.patchBatchEnc = self._encode_patches_vgg(patchesRaw.astype(np.float32))

        # Compute the metric.
        distancesPerSetPerPatch = []
        for iSet, supportSet in enumerate(self.supportSets):
            # The model-based distance is an L1 distance between the support patch encoding
            # and the target patch encoding.
            distToSupportPatchModel = []
            for iSupportPatch in range(len(supportSet.patches)):
                # Using Mse for the distance in feature space.
                # dist = np.mean(np.abs(self.patchBatchEnc - self.supportPatchesEnc[iSet, iSupportPatch, ...]), axis=1)
                dist = np.mean(np.square(self.patchBatchEnc - self.supportPatchesEnc[iSet, iSupportPatch, ...]), axis=1)
                distToSupportPatchModel.append(dist)

            distancesPerSetPerPatch.append(distToSupportPatchModel)

        return distancesPerSetPerPatch

    def _encode_patches_vgg(self, patches):
        from keras.applications.vgg16 import preprocess_input

        # todo Using only the first frame for now. Ideally should use all.
        patches = patches[:, 0, 0, ..., 0]  # Take the first frame, drop the Z axis and the attribute axis.
        patchesResized = np.empty((patches.shape[0], *self.vggInputShape), dtype=np.float32)

        for iPatch in range(patches.shape[0]):
            patchesResized[iPatch] = skimage.transform.resize(patches[iPatch], self.vggInputShape)

        vggInput = preprocess_input(patchesResized[..., np.newaxis].repeat(3, -1))
        # Encode and flatten out spatial dims (if using conv/pool layers).
        return self.vggModel.predict(vggInput, batch_size=32, verbose=False).reshape((vggInput.shape[0], -1))


class SimilarityMetricMse(SimilarityMetricBase):

    def __init__(self, config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber: int, logger: logging.Logger = None):
        super().__init__(config, patchShape, attrNumber, logger)

        # We could say that MSE 'encodes' the support patch with identity,
        # but let's be more explicit and create a separate field.
        self.supportPatchesRaw = None  # type: Optional[np.ndarray]

    def encode_support_sets(self, supportPatchesRaw: np.ndarray, supportSets: List[SupportSetDesc]) -> np.ndarray:
        super().encode_support_sets(supportPatchesRaw, supportSets)
        self.supportPatchesRaw = supportPatchesRaw

        return self.supportPatchesEnc

    def compute_patch_dist(self, patchesRaw: np.ndarray = None,
                           patchesEnc: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
        super().compute_patch_dist(patchesRaw)

        distancesPerSetPerPatch = []
        for iSet, supportSet in enumerate(self.supportSets):
            distPerPatch = []
            for iSupportPatch in range(len(supportSet.patches)):
                dist = np.mean(np.square(patchesRaw - self.supportPatchesRaw[iSet, iSupportPatch]),
                               axis=tuple(range(1, patchesRaw.ndim)))  # Mean over space-time.
                distPerPatch.append(dist)

            distancesPerSetPerPatch.append(distPerPatch)

        return distancesPerSetPerPatch


class SimilarityMetricWasserstein(SimilarityMetricBase):

    def __init__(self, config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber: int,
                 logger: logging.Logger = None):
        super().__init__(config, patchShape, attrNumber, logger)
        if attrNumber > 1 or patchShape[1] > 1:
            raise NotImplementedError()

        self.supportPatchesRaw = None  # type: Optional[np.ndarray]

        self.downsamplingFactors = [0.25, 0.25]
        downsampledShape = scipy.ndimage.interpolation.zoom(np.zeros(patchShape[2:]),  # Take only the first frame.
                                                            self.downsamplingFactors,
                                                            order=1, mode='nearest').shape

        # Generate a patch-shaped array of coordinates.
        patchCellCoords = np.stack(np.meshgrid(*tuple(np.arange(x) for x in downsampledShape), indexing='ij'), axis=-1)
        # Reshape into a flat list of coordinate vectors.
        patchCellCoordsFlat = patchCellCoords.reshape((-1, patchCellCoords.shape[-1]))
        # Compute the distances needed for Wasserstein.
        self.patchCellDist = ot.dist(patchCellCoordsFlat, patchCellCoordsFlat, metric='euclidean')

    def encode_support_sets(self, supportPatchesRaw: np.ndarray, supportSets: List[SupportSetDesc]) -> np.ndarray:
        super().encode_support_sets(supportPatchesRaw, supportSets)
        self.supportPatchesRaw = supportPatchesRaw

        return self.supportPatchesEnc

    def compute_patch_dist(self,
                           patchesRaw: Optional[np.ndarray] = None,
                           patchesEnc: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
        super().compute_patch_dist(patchesRaw, patchesEnc)
        assert patchesRaw is not None

        distancesPerSetPerPatch = []
        for iSet, supportSet in enumerate(self.supportSets):
            distPerSupPatch = []
            for iSupportPatch in range(len(supportSet.patches)):
                # Take only the first frame, drop the Z axis and the attribute axis.
                supportPatch = self.supportPatchesRaw[iSet, iSupportPatch, 0, 0, :, :, 0]
                # Downsample the patch, we can't handle the full size.
                supportPatch = scipy.ndimage.interpolation.zoom(supportPatch, self.downsamplingFactors,
                                                                order=1, mode='nearest')
                supportPatchFlat = supportPatch.flatten()
                sumSupport = np.sum(supportPatch)
                distances = np.zeros(patchesRaw.shape[0], np.float32)
                for iTarget in range(patchesRaw.shape[0]):
                    # Take only the first frame, drop the Z axis and the attribute axis.
                    targetPatch = patchesRaw[iTarget, 0, 0, :, :, 0]
                    targetPatch = scipy.ndimage.interpolation.zoom(targetPatch, self.downsamplingFactors,
                                                                   order=1, mode='nearest')

                    sumTarget = np.sum(targetPatch)
                    targetPatch = (targetPatch / sumTarget * sumSupport).astype(np.uint8)

                    distances[iTarget] = ot.unbalanced.sinkhorn_unbalanced2(supportPatchFlat, targetPatch.flatten(),
                                                                            self.patchCellDist, 0.1, 1.0)

                distPerSupPatch.append(distances)
            distancesPerSetPerPatch.append(distPerSupPatch)

        return distancesPerSetPerPatch


class SimilarityMetricHist(SimilarityMetricBase):

    class HistMetric(Enum):
        none = 0
        l1 = 1
        wasserstein = 2

    def __init__(self, config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber: int, metric: HistMetric,
                 logger: logging.Logger = None):
        super().__init__(config, patchShape, attrNumber, logger)

        if attrNumber > 1:
            raise NotImplementedError()

        self.metric = metric
        self.binNumber = config.metricHistBinNumber

    def encode_support_sets(self, supportPatchesRaw: np.ndarray, supportSets: List[SupportSetDesc]) -> np.ndarray:
        assert supportPatchesRaw.dtype == np.uint8
        super().encode_support_sets(supportPatchesRaw, supportSets)

        # We will store a histogram per set per patch.
        self.supportPatchesEnc = np.zeros((*supportPatchesRaw.shape[0:2], self.binNumber), dtype=np.int32)
        for iSet, supportSet in enumerate(supportSets):
            for iPatch, patch in enumerate(supportSet.patches):
                self.supportPatchesEnc[iSet, iPatch, ...] = self._compute_hist(supportPatchesRaw[iSet, iPatch, ...])

        return self.supportPatchesEnc

    def compute_patch_dist(self,
                           patchesRaw: Optional[np.ndarray] = None,
                           patchesEnc: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
        super().compute_patch_dist(patchesRaw, patchesEnc)
        assert patchesRaw is not None

        distancesPerSetPerPatch = []
        for iSet, supportSet in enumerate(self.supportSets):
            distPerSupPatch = []
            for iSupportPatch in range(len(supportSet.patches)):
                supportPatchHist = self.supportPatchesEnc[iSet, iSupportPatch]

                distances = np.zeros(patchesRaw.shape[0], np.float32)
                for iTarget in range(patchesRaw.shape[0]):
                    targetPatch = patchesRaw[iTarget]
                    targetPatchHist = self._compute_hist(targetPatch)
                    if self.metric == SimilarityMetricHist.HistMetric.l1:
                        distances[iTarget] = np.sum(np.abs(supportPatchHist - targetPatchHist))
                    elif self.metric == SimilarityMetricHist.HistMetric.wasserstein:
                        distances[iTarget] = scipy.stats.wasserstein_distance(supportPatchHist, targetPatchHist)
                    else:
                        raise ValueError("Unknown histogram metric: {}".format(self.metric))

                distPerSupPatch.append(distances)
            distancesPerSetPerPatch.append(distPerSupPatch)

        return distancesPerSetPerPatch

    def _compute_hist(self, patch: np.ndarray):
        return np.histogram(patch, bins=self.binNumber, range=(0, 255))[0]


class SimilarityMetricSsim(SimilarityMetricBase):

    def __init__(self, config: SiameseConfig, patchShape: Tuple[int, ...], attrNumber: int, logger: logging.Logger = None):
        super().__init__(config, patchShape, attrNumber, logger)

        if attrNumber > 1 or patchShape[1] > 1:
            raise NotImplementedError()

        self.supportPatchesRaw = None  # type: Optional[np.ndarray]
        self.supportMaxVal = 0.0

    def encode_support_sets(self, supportPatchesRaw: np.ndarray, supportSets: List[SupportSetDesc]) -> np.ndarray:
        super().encode_support_sets(supportPatchesRaw, supportSets)
        self.supportPatchesRaw = supportPatchesRaw
        self.supportMaxVal =  np.max(supportPatchesRaw)

        return self.supportPatchesEnc

    def compute_patch_dist(self,
                           patchesRaw: Optional[np.ndarray] = None,
                           patchesEnc: Optional[np.ndarray] = None) -> List[List[np.ndarray]]:
        super().compute_patch_dist(patchesRaw, patchesEnc)
        assert patchesRaw is not None

        dataMaxVal = max(self.supportMaxVal, np.max(patchesRaw))

        distancesPerSetPerPatch = []
        for iSet, supportSet in enumerate(self.supportSets):
            distPerSupPatch = []
            for iSupportPatch in range(len(supportSet.patches)):
                supportPatch = self.supportPatchesRaw[iSet, iSupportPatch]
                distances = np.zeros(patchesRaw.shape[0], np.float32)
                for iTarget in range(patchesRaw.shape[0]):
                    targetPatch = patchesRaw[iTarget]

                    distSum = 0
                    for f in range(supportPatch.shape[0]):
                        distSum += skimage.metrics.structural_similarity(supportPatch[f, 0, :, :, 0],
                                                                         targetPatch[f, 0, :, :, 0],
                                                                         win_size=11,
                                                                         data_range=(dataMaxVal - 0))

                    # Invert the value, since SSIM is large for similar images.
                    distances[iTarget] = 1.0 - distSum / (supportPatch.shape[0] * targetPatch.shape[0])

                distPerSupPatch.append(distances)
            distancesPerSetPerPatch.append(distPerSupPatch)

        return distancesPerSetPerPatch


def compute_feature_matches(config: SiameseConfig, encoder: keras.models.Sequential, normalizer: Normalizer,
                            memberNames: List[str], patchShape: Tuple[int, ...], attrNumber: int,
                            supportSets: List[SupportSetDesc], patchStride: Optional[Tuple[int, ...]] = None,
                            outTargetPatchData: Optional[h5py.Dataset] = None,
                            outTargetPatchEncoded: Optional[h5py.Dataset] = None,
                            logger: Optional[logging.Logger] = None):

    logger = logging_tools.get_null_logger_if_none(logger)

    # By default, do patching with no overlap.
    patchStride = patchStride or patchShape
    supportSetNumber = len(supportSets)
    patchShapeWithAttr = patchShape + (attrNumber,)

    # ========== Initialize all the similarity metrics. ==========
    similarityMetrics = build_similarity_metrics(config.simMetricNames, config, patchShape, attrNumber,
                                                 normalizer, encoder, logger)
    aggFuncs = build_agg_funcs(config.aggFuncNames)
    rankingScores = list(itertools.product(similarityMetrics, aggFuncs))
    scoreNames = list(map(lambda names: '-'.join(names),
                          itertools.product(config.simMetricNames, config.aggFuncNames)))

    # ========== Prepare the support set data. ==========
    # These two arrays are partially empty, because support set sizes could be different and we allocate for max.
    supportPatchesRaw = load_encode_support_sets(similarityMetrics, config, patchShape, attrNumber, supportSets,
                                                 logger=logger)

    # ========== Loop through all the patches and compute distances. ==========
    # The patch size is aproximate, not including the index nad the dummy y-patch.
    patchingBatchSize = int(config.matchBatchSizeBytes / (npe.multiply(patchShape) * np.dtype(np.uint8).itemsize))
    patchBuffer = np.empty((patchingBatchSize, *patchShapeWithAttr), dtype=np.uint8)
    indexBuffer = np.empty((patchingBatchSize, len(patchShape)), dtype=np.uint64)
    # Patch extractor also fetches the 'y-patch', which we don't need here.
    dummyBuffer = np.empty((patchingBatchSize, 1, 1, 1, 1, 1), dtype=np.uint8)

    # Use a generator to feed volumes to the patch extractor.
    volumeIterable = (load_member_volume(config, name) for name in memberNames)
    volumeNumber = len(memberNames)
    patchExtractor = MultidataPatchExtractor(
        volumes=volumeIterable,
        patchXSize=patchShape,
        patchStride=patchStride,
        logger=logging_tools.configure_child(logger, 'extractor', logging.INFO),
        inputBufferSize=int(5e8),

        # Default 'unused' settings.
        patchYSize=(1, 1, 1, 1), patchInnerStride=(1, 1, 1, 1),
        predictionDelay=0,  # Set to zero to make patch support == patch-x size (relevant near borders.)
        detectEmptyPatches=False, emptyValue=66, emptyCheckFeature=0,
        undersamplingProbAny=1.0, undersamplingProbEmpty=1.0, undersamplingProbNonempty=1.0,
    )
    targetPatchDesc = []

    logger.info("Computing feature matches using following scores: {}.".format(', '.join(scoreNames)))
    logger.info("Starting volume patch matching.")
    # todo This is getting insane, refactor.
    # For each metric, for each support set, construct a list of matches.
    distancesPerScorePerSet = [[[] for _ in supportSets] for _ in rankingScores]
    while patchExtractor.has_next_batch():
        patchesExtracted = patchExtractor.extract_next_batch(patchingBatchSize, outX=patchBuffer,
                                                             outY=dummyBuffer, outIndices=indexBuffer)

        if patchesExtracted == 0 or patchesExtracted > patchingBatchSize:
            raise RuntimeError("Invalid number of patches extracted: {}.".format(patchesExtracted))

        # Get the patch contents.
        patchesRaw = patchBuffer[:patchesExtracted].astype(np.float32)
        # Get the metadata for each patch.
        patchDescBatch = [PatchDesc(memberNames[patchExtractor.datasetIndex], tuple(int(x) for x in index))
                          for index in indexBuffer[:patchesExtracted]]

        for iScore, (metric, aggFunc) in enumerate(rankingScores):
            # Compute the similarity metric between all support set patches and all the target patches.
            distPerSetPerPatch = metric.compute_patch_dist(patchesRaw)
            # Aggregate the similarity metric into a ranking score.
            for iSet, supportSet in enumerate(supportSets):
                distance = aggFunc(distPerSetPerPatch[iSet], supportSet)
                distancesPerScorePerSet[iScore][iSet].extend(distance)

        # todo We should pre-compute the total number of patches and pre-allocate the arrays.
        # Store the raw patch data.
        if outTargetPatchData is not None:
            patchNumberStored = outTargetPatchData.shape[0]
            outTargetPatchData.resize(patchNumberStored + patchesExtracted, axis=0)
            outTargetPatchData[patchNumberStored:] = patchBuffer[:patchesExtracted]
        if outTargetPatchEncoded is not None:
            # This is a bit of a hack, but since the model is a special metric,
            # we break the interface to fetch the encodings, which will be needed for the app.
            metricModel = next((m for m in similarityMetrics if isinstance(m, SimilarityMetricModel)), None)
            if metricModel is not None:
                patchNumberStored = outTargetPatchEncoded.shape[0]
                outTargetPatchEncoded.resize(patchNumberStored + patchesExtracted, axis=0)
                outTargetPatchEncoded[patchNumberStored:] = metricModel.patchBatchEnc

        targetPatchDesc.extend(patchDescBatch)

        progress = patchExtractor.get_progress(volumeNumber)
        logger.info("Loading and matching volume patches ... {:.1f}%".format(progress),
                    extra={'throttlingId': 'compute_feature_matches'})

    patchExtractor.destruct()

    # ========== Use the distances to sort the patches. ==========
    featureMatchesPerMetric = []
    for iMetric, metricName in enumerate(scoreNames):

        featureMatches = []
        for iSet in range(supportSetNumber):
            distances = distancesPerScorePerSet[iMetric][iSet]
            indicesSorted = np.argsort(np.asarray(distances)).tolist()
            featureMatches.append((indicesSorted, distances))

        featureMatchesPerMetric.append(featureMatches)

    return featureMatchesPerMetric, scoreNames, supportPatchesRaw, targetPatchDesc


def compute_feature_matches_model_preencoded(
        metricModel: SimilarityMetricModel,
        targetPatchEnc: npe.LargeArray,
        supportSets: List[SupportSetDesc],
        logger: Optional[logging.Logger] = None):
    """
    Same as the original function, but uses precomputed target patch encodings
    to speed up the matching.
    """
    logger = logging_tools.get_null_logger_if_none(logger)

    supportSetNumber = len(supportSets)

    # Compute the similarity metric between all support set patches and all the target patches.
    distPerSetPerPatch = metricModel.compute_patch_dist(patchesEnc=targetPatchEnc)
    # Prepare the aggregation function. Always using mean in this optimized function.
    aggFunc = build_agg_funcs(['mean'])[0]
    # Aggregate the similarity metric into a ranking score.
    distancesPerSet = []
    for iSet, supportSet in enumerate(supportSets):
        distances = aggFunc(distPerSetPerPatch[iSet], supportSet)
        distancesPerSet.append(distances)

    # ========== Use the distances to sort the patches. ==========
    featureMatchesPerSet = []
    for iSet in range(supportSetNumber):
        distances = distancesPerSet[iSet]
        indicesSorted = np.argsort(np.asarray(distances)).tolist()
        featureMatchesPerSet.append((indicesSorted, distances))

    return featureMatchesPerSet


def load_encode_support_sets(similarityMetrics: List[SimilarityMetricBase],
                             config: SiameseConfig,
                             patchShape: Tuple[int, ...],
                             attrNumber: int,
                             supportSets: List[SupportSetDesc],
                             preloadedData: Optional[EnsembleDataLoaderCached] = None,
                             logger: Optional[logging.Logger] = None):

    logger = logging_tools.get_null_logger_if_none(logger)
    timer = StageTimer()
    timer.start_stage('init')

    supportSetNumber = len(supportSets)
    supportPatchNumberMax = max(len(s.patches) for s in supportSets)
    patchShapeWithAttr = patchShape + (attrNumber,)

    supportPatchesRaw = np.zeros((supportSetNumber, supportPatchNumberMax, *patchShapeWithAttr), dtype=np.uint8)
    timer.start_stage('load')
    for iSet, supportSetDesc in enumerate(supportSets):
        logger.info("Loading support set {}/{}.".format(iSet + 1, supportSetNumber), extra={'throttle': True})

        for iSupportPatch, (patchDesc, isPositive) in enumerate(supportSetDesc.patches):
            if not preloadedData:
                supportMemberData = load_member_volume(config, patchDesc.memberName)
                # We use re-mapped support set specs, so we can use the coordinates directly.
                supportPatch = patching_tools.get_patch_from_volume(supportMemberData, patchDesc.coords, patchShape)
            else:
                supportPatch = preloadedData.load_patch_data(patchDesc)

            supportPatchesRaw[iSet, iSupportPatch, ...] = supportPatch

    # Compute the support set encodings for all similarity metrics.
    timer.start_stage('encode')
    for metric in similarityMetrics:
        metric.encode_support_sets(supportPatchesRaw, supportSets)

    timer.end()
    logger.info("Support set encoding performance: {}".format(timer.get_total_report()))

    return supportPatchesRaw


def build_similarity_metrics(metricNames: List[str],
                             config: SiameseConfig,
                             patchShape: Tuple[int, ...],
                             attrNumber: int,
                             normalizer: Normalizer,
                             encoder: keras.models.Sequential,
                             logger: logging.Logger) -> List[SimilarityMetricBase]:
    simMetrics = []
    for name in metricNames:
        if name == 'model':
            simMetrics.append(SimilarityMetricModel(config, patchShape, attrNumber, normalizer, encoder, logger=logger))
        elif name == 'vgg':
            simMetrics.append(SimilarityMetricVgg(config, patchShape, attrNumber, logger=logger))
        elif name == 'mse':
            simMetrics.append(SimilarityMetricMse(config, patchShape, attrNumber, logger=logger))
        elif name == 'wasserstein':
            simMetrics.append(SimilarityMetricWasserstein(config, patchShape, attrNumber, logger=logger))
        elif name == 'ssim':
            simMetrics.append(SimilarityMetricSsim(config, patchShape, attrNumber, logger=logger))
        elif name == 'hist-l1':
            simMetrics.append(SimilarityMetricHist(config, patchShape, attrNumber, SimilarityMetricHist.HistMetric.l1, logger=logger))
        elif name == 'hist-wasserstein':
            simMetrics.append(SimilarityMetricHist(config, patchShape, attrNumber, SimilarityMetricHist.HistMetric.wasserstein, logger=logger))
        else:
            raise ValueError("Unknown distance metric: '{}'".format(name))

    return simMetrics


def build_agg_funcs(aggFuncNames: List[str]) -> List[Callable]:
    aggFuncs = []
    for aggName in aggFuncNames:
        if aggName == 'mean':
            aggFuncs.append(_agg_mean)
        elif aggName == 'min':
            aggFuncs.append(_agg_min)
        else:
            raise ValueError("Unknown aggregation function: '{}'".format(aggName))

    return aggFuncs


def _agg_mean(distToSupportPatches: List[np.ndarray], supportSet: SupportSetDesc) -> np.ndarray:
    targetPatchNumber = distToSupportPatches[0].shape[0]
    distances = np.zeros(targetPatchNumber, dtype=np.float64)
    for iSupportPatch, (patch, isPositive) in enumerate(supportSet.patches):
        sign = 1 if isPositive else -1
        distances += sign * distToSupportPatches[iSupportPatch]

    return distances / len(supportSet.patches)


def _agg_min(distToSupportPatches: List[np.ndarray], supportSet: SupportSetDesc) -> np.ndarray:
    targetPatchNumber = distToSupportPatches[0].shape[0]

    def _dist_to_first_matching_or_zeros(shouldBePos):
        for d, (p, isPos) in zip(distToSupportPatches, supportSet.patches):
            if isPos == shouldBePos:
                return d.copy()

        return np.zeros(targetPatchNumber, dtype=np.float64)

    # Initialize the mins as the distance to the first pos/neg patch or just zeros, if none are present.
    # This way, for example if there are no negative patches, we will not affect the result.
    distancesMinPos = _dist_to_first_matching_or_zeros(shouldBePos=True)
    distancesMinNeg = _dist_to_first_matching_or_zeros(shouldBePos=False)

    for iSupportPatch, (patch, isPositive) in enumerate(supportSet.patches):
        distArray = distancesMinPos if isPositive else distancesMinNeg
        # Write the values, don't just reference a new array.
        distArray[...] = np.minimum(distArray, distToSupportPatches[iSupportPatch])

    # Minimize min distance to positives, maximize min distance to negatives.
    return distancesMinPos - distancesMinNeg

