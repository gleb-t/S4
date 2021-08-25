import json
import os
import pickle
import tempfile
import time
import sys

import flask
import matplotlib.colors
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pyplant
from pyplant.specs import KerasModelSpec

from PythonExtras import config_tools, logging_tools
from PythonExtras.StageTimer import StageTimer
from Siamese.config import SiameseConfig
from Siamese import feature_matching
from Siamese.data_loading import load_ensemble_member_metadata, EnsembleDataLoaderCached
from Siamese.feature_matching import SimilarityMetricModel
from Siamese.layers import CUSTOM_KERAS_LAYERS
from Siamese.data_types import PatchDesc, TSupportSetPatchList, SupportSetDesc

app = flask.Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
# Importantly, we run the server with --no-reload option.
# I was having issues with Tensorflow being loaded twice using the 'stat' reloader.

# Backward compatibility hack for pickle, after renaming the module.
sys.modules['Siamese.types'] = sys.modules['Siamese.data_types']

renderedImageDirPath = os.path.expandvars(r'${DEV_PYPLANT_PATH}\siamese-app')
patchesToSend = None  # Send all.

# siameseRunPath = r'T:\out\siamese\cylinder-basic\200211-161408-silent-smoke_cylinder-basic'
# siameseConfigPath = os.path.expandvars(r'${DEV_SIAMESE_CONFIG_PATH}\cylinder-basic.py')
# ensembleName = 'cylinder-25'

# Requires PyPlant 0.4.3
# siameseRunPath = os.path.expandvars(r'${DEV_OUT_PATH}/siamese/200318-001328_cluster-913_cylinder-300-2-basic-count-same\200318-001347-dawn-star_cylinder-300-2-basic-count-same')
# siameseConfigPath = os.path.expandvars(r'${DEV_SIAMESE_CONFIG_PATH}\cylinder-300-2-basic-count-same.py')
# ensembleName = 'cylinder-300-2'

siameseRunPath = os.path.expandvars(r'${DEV_OUT_PATH}\siamese\droplet-splash-debug\200316-190837-odd-water_droplet-splash-debug')
siameseConfigPath = os.path.expandvars(r'${DEV_SIAMESE_CONFIG_PATH}\droplet-splash-debug.py')
ensembleName = 'droplet-splash'

# siameseRunPath = os.path.expandvars(r'${DEV_OUT_PATH}\siamese\200331-154800_cluster-928_200331_droplet-splash-nonspatial\200331-154826-ancient-sunset_200331_droplet-splash-nonspatial')
# siameseConfigPath = os.path.expandvars(r'${DEV_SIAMESE_CONFIG_PATH}\200331_droplet-splash-nonspatial.py')
# siameseRunPath = os.path.expandvars(r'${DEV_OUT_PATH}\siamese\200330-182759_cluster-927_200327_droplet-splash-1000\200330-182814-aged-base_200327_droplet-splash-1000')
# siameseConfigPath = os.path.expandvars(r'${DEV_SIAMESE_CONFIG_PATH}\200327_droplet-splash-1000.py')
# ensembleName = 'droplet-splash-1000'

logger = logging_tools.configure_logger('siamese-server', logToStd=True)

# We load both the 'original' Python object config and it's expanded JSON version,
# because we need the expanded output paths.
config = config_tools.parse_and_apply_config_eval(SiameseConfig(), siameseConfigPath)

memberMetadata = load_ensemble_member_metadata(config.dataPath, config.downsampleFactor, config.volumeCrop)
memberMetadataJson = {name: list(shape) for name, shape in zip(memberMetadata.memberNames, memberMetadata.memberShapes)}

# Grab a reference to the TF graph into which we're going to load the model.
graph = tf.get_default_graph()

# Load the ingredients needed to run the feature matching method.
ingredientExportPath = os.path.join(siameseRunPath, 'export')
ingredients = pyplant.utils.load_ingredients_from_dir(ingredientExportPath, logger=logger,
                                                      customSpecs=[KerasModelSpec(CUSTOM_KERAS_LAYERS)])

# Load the pre-encoded target patches.
with open(os.path.join(siameseRunPath, 'patches', 'patch-desc.pcl'), 'rb') as file:
    targetPatchDesc = pickle.load(file)
targetPatchEnc = np.load(os.path.join(siameseRunPath, 'patches', 'patch-encoding.npy'))

# As a dumb optimization for the sake of easing the IO bottleneck, preload the whole ensemble data into memory.
logger.info("Preloading the raw ensemble data into memory.")
preloadedEnsembleData = EnsembleDataLoaderCached(config, ingredients['patch-shape'])

# Prepare the colormap.
colormap = plt.get_cmap('plasma')
colormapJson = []
for i in range(colormap.N):
    colormapJson.append(matplotlib.colors.rgb2hex(colormap(i)[:3]))

# todo This is a hack to get old models with attr dimension to work. Need to reproduce old results.
with graph.as_default():
    import keras
    encoder = ingredients['model-full'].get_layer('encoder')  # type: keras.models.Sequential
    newEnc = keras.models.Sequential()
    newEnc.add(keras.layers.Lambda(lambda x: x[..., 0], input_shape=encoder.input_shape[1:] + (1,)))
    for l in encoder.layers:
        newEnc.add(l)

    encoder = newEnc


logger.info("Server initialized.")


@app.route('/')
def main():

    return flask.render_template('app.html',
                                 json=json,
                                 patchShape=list(ingredients['patch-shape']),
                                 memberMetadata=memberMetadataJson,
                                 colormap=colormapJson,
                                 ensembleName=ensembleName,
                                 patchesToSend=patchesToSend)


@app.route('/image/<path:path>')
def get_rendered_image(path):
    imagePath = flask.safe_join(renderedImageDirPath, path)

    if not os.path.exists(imagePath):
        flask.abort(404)

    return flask.send_file(imagePath, mimetype='image/png')


@app.route('/get-matches', methods=['post'])
def get_matches():
    timer = StageTimer()
    timer.start_stage('init')

    dataJson = flask.request.get_json(force=True)

    # Format mimics TSupportSetPatchList
    def _map_patch_list_json(patchTupleJson):
        patchDescJson = patchTupleJson[0]
        isPositive = patchTupleJson[1]

        memberName = patchDescJson['memberName']
        coords = tuple(patchDescJson['coords'])

        return (PatchDesc(memberName, coords), isPositive)

    if len(dataJson) == 0:
        flask.abort(404)
        return  # Just for clarity.

    patchList = list(map(_map_patch_list_json, dataJson))  # type: TSupportSetPatchList
    supportSetDesc = SupportSetDesc(0, patchList)

    timer.start_stage('encode')
    # Use the graph into which we've loaded the model. Otherwise TF uses a different graph per thread.
    with graph.as_default():
        # encoder = ingredients['model-full'].get_layer('encoder')
        metricModel = SimilarityMetricModel(
            config, ingredients['patch-shape'], 1, ingredients['normalizer'], encoder, logger=logger
        )

        feature_matching.load_encode_support_sets(
            [metricModel], config, ingredients['patch-shape'], 1, [supportSetDesc],
            preloadedData=preloadedEnsembleData,
            logger=logger
        )

        timer.start_stage('compute')
        matchesPerSet = feature_matching.compute_feature_matches_model_preencoded(
            metricModel, targetPatchEnc, [supportSetDesc], logger
        )

    timer.start_stage('respond')
    featureMatches = matchesPerSet[0]  # We provided only one set.
    patchIndices, metricValues = featureMatches

    distanceMin, distanceMax = min(metricValues), max(metricValues)
    matches = []
    for iPatch in patchIndices[:patchesToSend]:
        patchDesc = targetPatchDesc[iPatch]
        distance = metricValues[iPatch]
        matches.append({
            'memberName': patchDesc.memberName,
            'coords': patchDesc.coords,
            'distance': distance
        })
    timer.end_pass()
    logger.info("Finished computing matches. Timings: {}".format(timer.get_pass_report()))

    return flask.jsonify({
        'matches': matches,
        'distanceMin': distanceMin,
        'distanceMax': distanceMax
    })
