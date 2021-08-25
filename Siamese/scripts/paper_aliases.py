

metricAliases = {
        'model-mean': 'Ours',
        'mse-mean': 'MSE',
        'vgg-mean': 'VGG',
        'wasserstein-mean': 'EMD',
        'ssim-mean': 'SSIM',
        'hist-l1-mean': 'Hist-L1',
        'hist-wasserstein-mean': 'Hist-EMD',
        'VGG-tune-l': 'VGG-tune-l',
        'VGG-tune-a': 'VGG-tune-a',
}
# Add value->value identity records, so that the mapping is safe to apply several times.
metricAliases = {**metricAliases, **dict(zip(metricAliases.values(), metricAliases.values()))}

featureAliases = {
    't': 'turb',
    'l, t': 'turb',
    't, l': 'turb',
    'l': 'lam',
    # In the hotroom datasets we plot only metrics for 'edit' feature, i.e. mixed, not for a single object type.
    'circle, edit': 'object',
    'edit, circle': 'object',
    'line, edit': 'object',
    'edit, line': 'object',
    'edit': 'object'
}
columnAliases = {
    'C%P': 'C',
    'P@10P': 'P10',
    'P@50P': 'P50',
    'P@100P': 'P100',

    'C%P-wq': 'C',
    'P@10P-wq': 'P10',
    'P@50P-wq': 'P50',
    'P@100P-wq': 'P100',

    'C%P-nq': 'C',
    'P@10P-nq': 'P10',
    'P@50P-nq': 'P50',
    'P@100P-nq': 'P100'
}

paramAliases = {
    'patchMaxOffsetSpace': 'o_s',
    'patchMaxOffsetTime': 'o_t'
}

# From https://personal.sron.nl/~pault/#sec:qualitative
# The original map for 7 categories.
metricColormap = {
    metricAliases['model-mean']: '#EE8866',
    metricAliases['mse-mean']: '#44BB99',
    metricAliases['vgg-mean']: '#BBCC33',
    'VGG-tune-a': '#77AADD',
    'VGG-tune-l': '#99DDFF',
    metricAliases['wasserstein-mean']: '#AAAA00',
    metricAliases['ssim-mean']: '#EEDD88',
    metricAliases['hist-l1-mean']: '#FFAABB',
    metricAliases['hist-wasserstein-mean']: '#DDDDDD',
}

metricOrder = [
    metricAliases['model-mean'],
    'VGG-tune-a',
    'VGG-tune-l',
    metricAliases['vgg-mean'],
    metricAliases['hist-wasserstein-mean'],
    metricAliases['hist-l1-mean'],
    metricAliases['mse-mean'],
    metricAliases['ssim-mean'],
    metricAliases['wasserstein-mean'],
]