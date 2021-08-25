import itertools
import os
import pickle
import random
import re
import sys
import warnings
from pathlib import Path
from typing import *

import numpy as np
import pandas
import pyplant

from Siamese.config import SiameseConfig
from Siamese.data_loading import load_ensemble_member_metadata, load_member_volume
from PythonExtras import numpy_extras as npe, config_tools, file_tools, patching_tools, cairo_extras, logging_tools
from Siamese.data_types import *
from Siamese.scripts import ensemble_figure_tools as tools
from Siamese.scripts import paper_aliases


class MatchRecord(NamedTuple):
    setIndex: int
    rank: int
    dist: float
    memberName: str
    coords: Tuple[int, ...]


def parse_tuple_string(string: str, valueParser: Callable[[str], Any]=None, valueType: type = None) -> Tuple:
    if string[0] != '(' or string[-1] != ')':
        raise RuntimeError("Cannot parse a tuple from string '{}'".format(string))

    withoutParenthesis = string[1:-1]
    valueStrings = [s.strip() for s in withoutParenthesis.split(',')]

    if valueParser is not None:
        return tuple([valueParser(s) for s in valueStrings])
    elif valueType is not None:
        return tuple([valueType(s) for s in valueStrings])
    else:
        return tuple(valueStrings)


def load_run_matches(metrics, runPath) -> Dict[str, List[MatchRecord]]:
    matchesPerMetric = {}
    for metric in metrics:
        matches = []
        # Backward compatibility, matches export has been moved.
        path = os.path.join(runPath, 'feature-matches-{}'.format(metric), 'matches.csv')
        if not os.path.exists(path):
            path = os.path.join(runPath, 'matches-{}.csv'.format(metric))

        with open(path, 'r') as file:

            for line in file:
                # todo The exported format was meant for reading, not parsing, and needs to be changed.
                reMatch = re.match(r'(\d+)\s+(\d+)\s+([\d.-]+)\s+([^\s]+)\s+(\(.*\)).*', line)
                iSet = int(reMatch.group(1))
                rank = int(reMatch.group(2))
                dist = float(reMatch.group(3))
                memberName = reMatch.group(4)
                coords = parse_tuple_string(reMatch.group(5), valueType=int)

                matches.append(MatchRecord(iSet, rank, dist, memberName, coords))

        matchesPerMetric[metric] = matches
    return matchesPerMetric


def render_feature_matches_figure(supportPatchData: npe.LargeArray, matchPatchData: npe.LargeArray,
                                  matchRanks: List[List[int]],
                                  supportSets: List[SupportSetDesc], supportSetFeatures: List[Tuple[str, str]],
                                  figConfig: SiameseConfig.FigureConfig, outputFilePath, frameIndex: int):
    matchesToRender = matchPatchData.shape[1]

    supportSetNumber = supportPatchData.shape[0]
    maxSupportPatchNumber = supportPatchData.shape[1]
    margin = figConfig.margin
    blockWidth, blockHeight = figConfig.blockWidth, figConfig.blockHeight
    impostorPadding = 0
    impostorTf = tools.get_ensemble_tf(figConfig.ensembleTfName)

    matrixWidth = matchesToRender * blockWidth + len(matchRanks) * figConfig.groupPadding
    matrixHeight = supportSetNumber * blockHeight
    # topLabelsHeight = blockHeight
    topLabelsHeight = 30
    leftLabelsWidth = 80
    supportSetPanelWidth = maxSupportPatchNumber * blockWidth + margin
    supportSetPanelMargin = blockWidth

    figureSize = (leftLabelsWidth + supportSetPanelWidth + supportSetPanelMargin + matrixWidth,
                  topLabelsHeight + matrixHeight)

    # Matches are grouped into sections of ranks.
    # Compute their global indices.
    matchIndices = range(sum(len(l) for l in matchRanks))
    # Compute the indices of their group.
    groupIndices = list(itertools.chain(*([i for _ in l] for i, l in enumerate(matchRanks))))
    matchRanksFlat = list(itertools.chain(*matchRanks))

    # Prepare the transfer functions.
    # plasmaTf = plt.get_cmap('plasma')
    # allDistanceMatrices = [matchTuple[1] for matchTuple in featureMatchesPerSet]
    # distanceMin, distanceMax = min((np.min(mat) for mat in allDistanceMatrices)), \
    #                            max((np.max(mat) for mat in allDistanceMatrices))
    # distanceTfModel = lambda value: plasmaTf((value - distanceMin) / (distanceMax - distanceMin))

    imageRenderer = tools.ImageRenderer()
    file_tools.create_dir(os.path.dirname(outputFilePath))
    with cairo_extras.CairoFileSurface(outputFilePath, figureSize) as surface:
        cr = surface.get_context()

        # Paint the white background.
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()
        cr.set_font_size(12)

        # -- Render the top labels.
        cr.set_source_rgb(0, 0, 0)
        cr.translate(leftLabelsWidth, 0)
        cr.translate(supportSetPanelWidth + supportSetPanelMargin, 0)
        for iMatch, iGroup in zip(matchIndices, groupIndices):
            rank = matchRanksFlat[iMatch]

            cairo_extras.cairo_show_text_centered(cr, str(rank + 1),
                                                  (iMatch + 0.5) * blockWidth + iGroup * figConfig.groupPadding,
                                                  topLabelsHeight / 2)
        cr.identity_matrix()
        cr.translate(leftLabelsWidth, topLabelsHeight)

        # --- Render the support patch panel.
        for iSet, supportSet in enumerate(supportSets):
            blockY = iSet * blockHeight
            featureName, patchCount = supportSetFeatures[iSet]
            if featureName in paper_aliases.featureAliases:
                featureName = paper_aliases.featureAliases[featureName]
            labelText = f'{featureName} {patchCount}'
            cairo_extras.cairo_show_text_centered(cr, labelText, -leftLabelsWidth / 2, blockY + blockHeight / 2)

            for iPatch, (patch, isPositive) in enumerate(supportSet.patches):
                imageHash = 'support_set_{}_patch_{}'.format(iSet, iPatch)
                patchImage = imageRenderer.render_image(supportPatchData[iSet, iPatch, frameIndex],
                                                        imageHash, impostorTf)
                blockX = iPatch * blockWidth
                _, _, imageRight, imageBot = cairo_extras.render_image_into_cairo(cr, patchImage,
                                                                                  blockX, blockY,
                                                                                  blockWidth, blockHeight)
                # cr.set_source_rgba(1, 0, 0, 1)
                # cr.rectangle(blockX, blockY, blockWidth, blockHeight)
                # cr.fill()

                iconSize = figConfig.iconSize
                lineWidth = iconSize / 3
                iconX, iconY = imageRight - iconSize, imageBot - iconSize
                render_plus_minus_icon(cr, iconSize, lineWidth, iconX, iconY, isPositive)

        cr.translate(supportSetPanelWidth + supportSetPanelMargin, 0)

        # --- Render the matches..
        for iSet, supportSet in enumerate(supportSets):
            for iMatch, iGroup in zip(matchIndices, groupIndices):
                rank = matchRanksFlat[iMatch]

                blockX, blockY = iMatch * blockWidth + iGroup * figConfig.groupPadding, \
                                 iSet * blockHeight

                # Render the member data.
                imageHash = 'match_patch_{}_{}'.format(iSet, rank)
                patchImage = imageRenderer.render_image(matchPatchData[iSet, iMatch, frameIndex], imageHash, impostorTf)
                cairo_extras.render_image_into_cairo(cr, patchImage,
                                                     blockX + impostorPadding,
                                                     blockY + impostorPadding,
                                                     blockWidth - 2 * impostorPadding,
                                                     blockHeight - 2 * impostorPadding)
                # cairo_extras.cairo_show_text_centered(cr, str(rank), blockX, blockY)


def render_plus_minus_icon(cr, iconSize, lineWidth, iconX, iconY, isPositive):
    cr.set_source_rgba(1, 1, 1, 1)
    cr.rectangle(iconX, iconY, iconSize + 1, iconSize + 1)
    cr.fill()
    cr.set_source_rgba(0, 0, 0, 1)
    cr.set_line_width(lineWidth)
    # The horizontal line (minus).
    cr.move_to(iconX, iconY + iconSize / 2)
    cr.line_to(iconX + iconSize, iconY + iconSize / 2)
    cr.stroke()
    if isPositive:
        # Make it a plus, if positive.
        cr.move_to(iconX + iconSize / 2, iconY)
        cr.line_to(iconX + iconSize / 2, iconY + iconSize)
        cr.stroke()


def load_match_data_and_render(runPath: str,
                               configPath: str,
                               ranksToRender: List[List[int]],
                               outDirPath: str,
                               supportSetIndices: Optional[List[int]] = None,
                               allowRandomSets: bool = False,
                               metrics: Optional[List[str]] = None):
    frameIndex = 0
    metrics = metrics or ['model-mean', 'mse-mean']
    config = config_tools.parse_and_apply_config_eval(SiameseConfig(), configPath)
    figConfig = config.figureConfig
    figConfig.margin = 0

    # todo This is brittle and should be better exported.
    with open(os.path.join(runPath, 'export', 'support-sets-prepared.pcl'), 'rb') as file:
        import sys
        # Backward compatibility hack for pickle, after renaming the module.
        sys.modules['Siamese.types'] = sys.modules['Siamese.data_types']

        supportSets = pickle.load(file)
        if not allowRandomSets:
            supportSets = [s for s in supportSets if not s.isRandom]
        # We rely on the index field, which used to be only valid for non-random sets.
        # We're overwriting the index here, so the following code can use it.
        for i, s in enumerate(supportSets):
            if i != s.index:
                warnings.warn("Support set #{}'s index doesn't match (got {}).".format(i, s.index))
            s.index = i

        if supportSetIndices:
            supportSets = [supportSets[i] for i in supportSetIndices]
        else:
            supportSetIndices = [s.index for s in supportSets]

    ingredientPath = os.path.join(runPath, 'export', 'patch-shape.pcl')
    if os.path.exists(ingredientPath):
        with open(ingredientPath, 'rb') as file:
            patchShape = pickle.load(file)
    else:
        # Older versions exported patch shape as a 'simple' type, not object.
        with open(os.path.join(runPath, 'export', 'simple.pyplant.pcl'), 'rb') as file:
            patchShape = pickle.load(file)['patch-shape']

    setMetrics = pandas.read_csv(os.path.join(runPath, 'feature-match-metrics.csv'), sep='\t')
    # For each support set. Find it in the table by index. (Take first, could be multiple metrics with same set),
    # get the description of the feature and patch count contained in the set. ("query intent")
    supportSetFeatures = [tuple(setMetrics.loc[setMetrics['Set'] == i + 1].iloc[0][['Feature', 'Patches']])
                          for i in supportSetIndices]  # type: List[Tuple[str, str]]

    print("Parsing match CSVs.")
    matchesPerMetric = load_run_matches(metrics, runPath)

    # Get the max rank, it's the same as the total number of patches/matches.
    matchesTotalNumber = max(r.rank for r in next(iter(matchesPerMetric.values())))
    matchesRenderNumber = sum(len(l) for l in ranksToRender)
    # Convert any negative values into the actual indices.
    ranksToRender = [[r if r >= 0 else matchesTotalNumber + r for r in l] for l in ranksToRender]

    maxSupportSetSize = max(len(s.patches) for s in supportSets)
    supportPatchData = np.zeros((len(supportSets), maxSupportSetSize, *patchShape), dtype=np.uint8)
    matchDataPerMetric = {m: np.zeros((len(supportSets), matchesRenderNumber, *patchShape), dtype=np.uint8)
                          for m in metrics}

    print("Loading the patch data.")
    for iSet, supportSet in enumerate(supportSets):

        for iPatch, (patch, isPos) in enumerate(supportSet.patches):
            volumeData = load_member_volume(config, patch.memberName)
            supportPatchData[iSet, iPatch, ...] = patching_tools.get_patch_from_volume(
                volumeData, patch.coords, patchShape
            )[..., 0]  # Take the first attribute.

        for metric in metrics:
            matchData = matchDataPerMetric[metric]

            allMatches = list(filter(lambda m: m.setIndex == supportSet.index, matchesPerMetric[metric]))
            matchesToRender = [allMatches[i] for i in itertools.chain(*ranksToRender)]

            assert all(m.rank == i for i, m in zip(itertools.chain(*ranksToRender), matchesToRender))
            assert len(matchesToRender) == matchesRenderNumber

            for iMatch, match in enumerate(matchesToRender):
                volumeData = load_member_volume(config, match.memberName)
                matchData[iSet, iMatch, ...] = \
                patching_tools.get_patch_from_volume(volumeData, match.coords, patchShape)[..., 0]  # Take the first attribute.

    for metric in metrics:
        outPath = os.path.join(outDirPath, 'matches-{}.pdf'.format(metric))
        render_feature_matches_figure(
            supportPatchData=supportPatchData,
            matchPatchData=matchDataPerMetric[metric],
            matchRanks=ranksToRender,
            supportSets=supportSets,
            supportSetFeatures=supportSetFeatures,
            figConfig=config.figureConfig,
            outputFilePath=outPath,
            frameIndex=frameIndex
        )


def main():

    ranksToRender = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [45, 46, 47, 48, 49],
        [95, 96, 97, 98, 99],
        [995, 996, 997, 998, 999],
        [-5, -4, -3, -2, -1]
    ]
    outRootPath = Path(os.environ['DEV_OUT_PATH'])
    configRootPath = Path(os.environ['DEV_SIAMESE_CONFIG_PATH'])
    outPath = outRootPath / 'paper'
    outPath.mkdir(exist_ok=True)

    if '--recompute' not in sys.argv:
        print("Using precomputed runset.")
        # --- TVCG 2020 --- Cylinder ---
        runPath = str(outRootPath /
                      '20200817-153710_cluster-1125_200817_cylinder-300-2-basic_all-metrics' /
                      '200817-153723-crimson-pond_200817_cylinder-300-2-basic_all-metrics')
    else:
        print("Using recomputed runset data.")
        runPath = str(outRootPath / 'recomputed-runset' / 'recomputed-run')

    load_match_data_and_render(
        runPath=runPath,
        configPath=str(configRootPath / '200414_cylinder-300-2-basic-paper.py'),
        supportSetIndices=[0, 2, 3, 5],
        ranksToRender=ranksToRender,
        outDirPath=str(outPath / 'cylinder')
    )


if __name__ == '__main__':
    main()
