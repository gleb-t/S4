import operator
import os
import sys
from pathlib import Path
from typing import *

import pandas

from Siamese.scripts.paper_aliases import *


def load_run_metrics(path: str):
    table = pandas.read_csv(os.path.expandvars(os.path.join(path, 'feature-match-metrics.csv')), sep='\t')
    # I was accidentally including the index into the csv, which needs to be filtered out.
    if 'Unnamed: 0' in table.columns:
        table = table.drop(['Unnamed: 0'], axis=1)

    return table


def export_run_manual_query_metrics(runPath: str,
                                    outputPath: str,
                                    includeSupportMembers: bool = True,
                                    metricsToShow: Optional[List[str]] = None,
                                    setIndicesToExport: Optional[List[int]] = None):
    """
    Export ranking metrics for the manual queries into a csv file to be used in the paper.
    :return:
    """
    keyColumns = ['Metric', 'Sample']
    labelCols = ['Feature', 'Patches']
    if includeSupportMembers:
        evalCols = ['C%P-wq', 'P@10P-wq', 'P@50P-wq', 'P@100P-wq']
    else:
        evalCols = ['C%P-nq', 'P@10P-nq', 'P@50P-nq', 'P@100P-nq']

    table = load_run_metrics(runPath)
    table = table[~table['Rand']]

    if setIndicesToExport:
        # If set filter is specified, keep results only for the requested support sets.
        table = table[table.apply(lambda r: r['Set'] in setIndicesToExport, axis=1)]

    table.reset_index()

    # Use the set index as the sample index (each manual set is unique).
    # This helps to reuse the grouping code below and handle things more uniformly.
    table['Sample'] = table['Set']
    # Drop out the irrelevant columns.
    table = table[keyColumns + labelCols + evalCols]

    # Rename some table values for presentation purposes.
    table = table.rename(columns=columnAliases)
    evalColsMapped = [columnAliases[c] if c in columnAliases else c for c in evalCols]
    table['Metric'] = table.apply(lambda r: metricAliases[r['Metric']], axis=1)
    table['Feature'] = table.apply(lambda r: featureAliases[r['Feature']], axis=1)

    metricNames = list(sorted(table['Metric'].unique(), key=lambda m: metricOrder.index(m)))
    if metricsToShow:
        # Restrict the list of metrics.
        newMetricNames = [metricAliases[n] for n in metricsToShow]
        assert all(n in metricNames for n in newMetricNames)  # All must exist.
        metricNames = newMetricNames

    metricSubtables = []
    for metric in metricNames:
        subtable = table[table.apply(lambda r: r['Metric'] == metric, axis=1)].copy().reset_index(drop=True)
        metricSubtables.append(subtable)

    assert all(metricSubtables[0].shape[0] == t.shape[0] for t in metricSubtables)  # All the same length.

    # Apply latex formatting to highlight the better results.
    formatCell = lambda val, other: f'\\textbf{{{val:.1f}}}' if val >= other else f'{val:.1f}'
    for colName in evalColsMapped:
        for iRow in range(metricSubtables[0].shape[0]):
            values = [t.loc[iRow, colName] for t in metricSubtables]
            bestValueIndex, bestValue = max(enumerate(values), key=operator.itemgetter(1))
            for subtable in metricSubtables:
                subtable.loc[iRow, colName] = formatCell(subtable.loc[iRow, colName], bestValue)

    subtablesWithNames = zip(metricSubtables, metricNames)
    finalTable = next(subtablesWithNames)[0].drop('Metric', 1)
    for subtable, metricName in subtablesWithNames:
        finalTable = finalTable.merge(subtable[['Sample'] + evalColsMapped], on='Sample', suffixes=('', f'-{metricName}'))

    print(finalTable)

    finalTable.to_csv(outputPath, sep=',', float_format='%.1f', index=False)


def main():
    outRootPath = Path(os.environ['DEV_OUT_PATH'])
    outPath = outRootPath / 'paper'
    outPath.mkdir(exist_ok=True)

    if '--recompute' not in sys.argv:
        print("Using precomputed runset.")
        # --- TVCG 2020 ---
        runPath = str(outRootPath /
                         '20200817-153710_cluster-1125_200817_cylinder-300-2-basic_all-metrics' /
                         '200817-153723-crimson-pond_200817_cylinder-300-2-basic_all-metrics')
    else:
        print("Using recomputed runset data.")
        runPath = str(outRootPath / 'recomputed-runset' / 'recomputed-run')

    export_run_manual_query_metrics(
        runPath=runPath,
        outputPath=str(outPath / 'cylinder.csv'),
        setIndicesToExport=[
            1, 2, 3, 4, 5, 6,       # Turbulent.
            13, 14, 15, 16, 17, 18  # Laminar.
        ]
    )


if __name__ == '__main__':
    main()
