from typing import Set, List

import numpy as np
import pandas

from Siamese.data_types import PatchDesc, TupleInt
from Siamese.feature_processing import EnsembleFeatureMetadata


def compute_query_ranking_metrics(supportFeatureNames: Set[str], patchesSorted: List[PatchDesc],
                                  metadataFiltered: EnsembleFeatureMetadata, patchShape: TupleInt,
                                  precisionRanks: List[int]):
    # Computing feature overlap.
    featureCoverageExact, patchesForCoverageExact = metadataFiltered.get_total_feature_coverage(
        supportFeatureNames, patchShape, patchesSorted, isExactNotPartial=True
    )
    featureCoveragePartial, patchesForCoveragePartial = metadataFiltered.get_total_feature_coverage(
        supportFeatureNames, patchShape, patchesSorted, isExactNotPartial=False
    )

    # For all target patches (ranked matches), check if it matches the feature from the support set.
    matches = []
    for patch in patchesSorted:
        targetFeatureNames = metadataFiltered.get_patch_features(patch, patchShape)

        if targetFeatureNames == supportFeatureNames:
            matches.append('exact')
        elif len(targetFeatureNames & supportFeatureNames) > 0:
            matches.append('partial')
        else:
            matches.append('false')

    # Compute the precision and the recall at various ranking cutoff values (top 10, top 50, etc.).
    exactPrecisions, exactRecalls = [], []
    partialPrecisions, partialRecalls = [], []

    def _compute_precisions(rank):
        # This can happen when we compute precision-at-coverage, with no exact coverage possible.
        if rank == 0:
            return 0, 0

        countExact = sum(1 for match in matches[:rank] if match == 'exact')
        countPartial = sum(1 for match in matches[:rank] if match in ['exact', 'partial'])

        exact = countExact / rank * 100
        partial = countPartial / rank * 100

        return exact, partial

    for precisionRank in precisionRanks:
        pExact, pPartial = _compute_precisions(precisionRank)
        exactPrecisions.append(pExact)
        partialPrecisions.append(pPartial)

    precisionAtCoverageExact, _ = _compute_precisions(patchesForCoverageExact)
    _, precisionAtCoveragePartial = _compute_precisions(patchesForCoveragePartial)

    return {
        'CnE': patchesForCoverageExact,
        'CnP': patchesForCoveragePartial,
        'C%E': featureCoverageExact,
        'C%P': featureCoveragePartial,
        'P@CnE': precisionAtCoverageExact,
        'P@CnP': precisionAtCoveragePartial,
        **dict(zip(['P@{}E'.format(p) for p in precisionRanks], exactPrecisions)),
        **dict(zip(['P@{}P'.format(p) for p in precisionRanks], partialPrecisions)),
    }


def aggregate_metrics_table(tableGrouped):
    aggregatedRows = []
    for keys, group in tableGrouped:  # type: pandas.DataFrame
        textFields = group.select_dtypes(exclude=np.number).iloc[0]  # These are the same for all rows. # todo check?
        dataToAggregate = group.select_dtypes(include=np.number)
        means = dataToAggregate.mean().rename(lambda col: col + '-m')
        stds = dataToAggregate.std().rename(lambda col: col + '-s')

        row = pandas.concat((textFields, means, stds))
        aggregatedRows.append(row)
    aggregatedTable = pandas.DataFrame(aggregatedRows)

    return aggregatedTable
