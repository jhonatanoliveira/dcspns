"""
Script for running experiments.

Example
-------
>>> python run_experiment.py --name example --seed 1234 --width 64 --height 64
--channels 1 --output-dir outputs --database-path databases/olivetti
--learning-rate 0.01 --minibatch-size 64 --epochs 3 --valid-amount 50
--first-sum-channels 12 --model-type tree --tree-model-size 16
--tree-model-alt-size 32 --tree-model-alt-amt 100 --leaf-components 4
--complete-side left --training-type nll --inference-type mpe
"""
import os
import gc
import datetime
import logging
import argparse
import numpy as np

from dcspn.utilities import Database, TicTac, plot_cost_graph

from dcspn.factory import single_branches, multiple_branches, concat_rec


# Execution arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
                    help="A name for the experiment.")
parser.add_argument('--seed', type=int, default=1234,
                    help="Seed for random generators.")
parser.add_argument('--width', type=int,
                    help="Image width (if smaller/greater than given dataset,\
                          will reshape).")
parser.add_argument('--height', type=int,
                    help="Image height (if smaller/greater than given dataset,\
                          will reshape).")
parser.add_argument('--channels', type=int,
                    help="Image channels.")
parser.add_argument('--output-dir', type=str,
                    help="Directory for output content.")
parser.add_argument('--database-path', type=str,
                    help="Path to image database folder.")
parser.add_argument('--database-name', type=str,
                    help="Name of SK Learn dataset.")
parser.add_argument('--learning-rate', type=float,
                    help="Learning rate.")
parser.add_argument('--minibatch-size', type=int, default=64,
                    help="Size of mini-batch during training.")
parser.add_argument('--epochs', type=int,
                    help="Epochs.")
parser.add_argument('--valid-amount', type=int,
                    help="Amount of valid instances.")
parser.add_argument('--first-sum-channels', type=int,
                    help="Amount of channels for first sum layer.")
parser.add_argument('--tree-model-size', type=int, default=2,
                    help="Pooling layer window size of main layer nodes.")
parser.add_argument('--tree-model-alt-size', type=int, default=2,
                    help="Pooling layer window size of alternative \
                    layer nodes.")
parser.add_argument('--tree-model-alt-amt', type=int, default=90,
                    help="Amount of layers for alternating tree model.")
parser.add_argument('--leaf-components', type=int,
                    help="Amount of components for leaf layer.")
parser.add_argument('--complete-side', type=str, choices=[
                    'left', 'bottom', 'center'],
                    help="Side to complete.")
parser.add_argument('--training-type', type=str,
                    choices=['nll', 'adversarial'],
                    help="Side to complete.")
parser.add_argument('--inference-type', type=str,
                    choices=['mpe', 'sampling'],
                    help="Side to complete.")
parser.add_argument('--model-type', type=str,
                    choices=['onion', 'tree', 'concat'],
                    help="Structure type.")
parser.add_argument('--use-post-processing-filter', type=float, default=0,
                    help="Gaussian filter as post-processing.")
parser.add_argument('--comet-api', type=str, default=None,
                    help="Comet API key.")
parser.add_argument('--comet-project', type=str, default=None,
                    help="Comet project name.")
parser.add_argument('--comet-workspace', type=str, default=None,
                    help="Comet workspace name.")
args = parser.parse_args()

# Set arguments to local variables
SEED = args.seed
save_folder = args.output_dir
database_path = args.database_path
database_name = args.database_name
if database_path is not None and database_name is not None:
    raise ValueError("Choose either a data path or name, not both")
exp_name = args.name
log_file = "exp.{}.log".format(exp_name)

# Tree model arguments
tree_model_size = args.tree_model_size
tree_model_alt_size = args.tree_model_alt_size
tree_model_alt_amt = args.tree_model_alt_amt

# Capturing other input arguments
img_height = args.height
img_width = args.width
img_channel = args.channels

learning_rate = args.learning_rate
minibatch_size = args.minibatch_size
epochs = args.epochs

first_sum_channel = args.first_sum_channels
valid_amount = args.valid_amount
leaf_components = args.leaf_components
input_shape = [img_height, img_width, img_channel]

model_type = args.model_type
complete_side = args.complete_side
training_type = args.training_type
inference_type = args.inference_type
use_filter = args.use_post_processing_filter

comet_api = args.comet_api
comet_project = args.comet_project
comet_workspace = args.comet_workspace
use_comet = comet_api and comet_project and comet_workspace


# Utility
# -------
def _create_dir(dir_name, inside_sav_fold=True):
    dir_name = "{}/{}".format(
        save_folder, dir_name) if inside_sav_fold else dir_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


# EXPERIMENT SETUP
# ----------------
timers = TicTac()
# Start global timer
timers.tic()
# Root output log file
_create_dir(save_folder, inside_sav_fold=False)
save_folder = "{}/exp_{:%Y_%m_%d__%H_%M}".format(
    save_folder, datetime.datetime.now())
_create_dir(save_folder, inside_sav_fold=False)
_create_dir("completions")
_create_dir("samplings")
_create_dir("models")


def _path_sav_fold(f):
        return "{}/{}".format(save_folder, f)


# Logging
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)
# Log to file
fh = logging.FileHandler(_path_sav_fold(log_file), mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
# Log to screen
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


# DATABASE
# ---------
timers.tic()
logger.info("Load and setup database.")
if not use_filter == 0:
    logger.info(
        "Post-processing with low-pass filter activated: {}".format(use_filter)
        )
database = Database(
    dataset_name="olivetti", normalize=False, seed=SEED) \
    if database_name else Database(
    data_path=database_path, normalize=False, seed=SEED)
database.resize(img_height, img_width)
database.normalize()

batch_size = database.batch_size

# Training hyper-parameters
poss_train_perc = int((1 / 3) * batch_size)
valid_amt = valid_amount if poss_train_perc > valid_amount else poss_train_perc
train_percentage = 1 - (valid_amt / batch_size)

train_data, eval_data = database.split(train_percentage=train_percentage,
                                       nhwc_format=True)

leaf_means, leaf_stds = Database.means_stds_quantiles(
    train_data, leaf_components)

# Part of the image to complete
train_data_marg = None
eval_data_marg = None
if complete_side == "left":
    train_data_marg = Database.marginalize(
        train_data, ver_idxs=[[0, img_height]],
        hor_idxs=[[int(img_width / 2), img_width]])
    eval_data_marg = Database.marginalize(
        eval_data, ver_idxs=[[0, img_height]],
        hor_idxs=[[0, int(img_width / 2)]])
elif complete_side == "bottom":
    train_data_marg = Database.marginalize(
        train_data, ver_idxs=[[int(img_height / 2), img_height]],
        hor_idxs=[[0, img_width]])
    eval_data_marg = Database.marginalize(
        eval_data, ver_idxs=[[int(img_height / 2), img_height]],
        hor_idxs=[[0, img_width]])
elif complete_side == "center":
    train_data_marg = Database.marginalize(
        train_data, ver_idxs=[[int(img_height / 4), 3 * int(img_height / 4)]],
        hor_idxs=[[int(img_width / 4), 3 * int(img_width / 4)]])
    eval_data_marg = Database.marginalize(
        eval_data, ver_idxs=[[int(img_height / 4), 3 * int(img_height / 4)]],
        hor_idxs=[[int(img_width / 4), 3 * int(img_width / 4)]])

logger.info("--> Duration: {:.4f}".format(timers.tac()))


# BUILD MODEL
# -------------
timers.tic()
logger.info("Building model")

if model_type == "onion":
    spn_def = {
        "input_shape": input_shape,
        "leaf": {
            "type": "gaussian",
            "num_leaf_components": leaf_components
        },
        "branches": [
            {
                "sum_layers": {
                    "channel_method": "constant",
                    "first_sum_channel": first_sum_channel
                },
                "product_layers": {
                    "pooling_method": "alternate",
                    "pooling_windows": [(1, 2), (2, 1)]
                }
            }
        ]
    }
    spn = single_branches(spn_def)
elif model_type == "tree":
    spn_def = {
        "input_shape": input_shape,
        "leaf":
        {
            "type": "gaussian",
            "num_leaf_components": leaf_components
        },
        "sum_layers": {
            "channel_method": "constant",
            "first_sum_channel": first_sum_channel,
            "hard_inference": False,
            "share_parameters": False,
            "initializer": "uniform"
        },
        "product_layers": {
            "pooling_method": {
                "type": "alternate",
                "amount": tree_model_alt_amt
            },
            "pool_windows": [(1, tree_model_size), (tree_model_size, 1)],
            "alt_pool_win": [(tree_model_alt_size, tree_model_alt_size),
                             (tree_model_alt_size, tree_model_alt_size)],
            "sum_pooling": True
        }
    }
    spn = multiple_branches(spn_def)
elif model_type == "concat":
    spn_def = {
        "input_shape": input_shape,
        "leaf":
        {
            "type": "gaussian",
            "num_leaf_components": leaf_components
        },
        "sum_layers": {
            "channel_method": "constant",
            "first_sum_channel": first_sum_channel,
            "hard_inference": False,
            "share_parameters": False,
            "initializer": "uniform"
        },
        "product_layers": {
            "pooling_method": {
                "type": "alternate",
                "amount": tree_model_alt_amt
            },
            "pool_windows": [(1, tree_model_size), (tree_model_size, 1),
                             (4, 1), (1, 4), (8, 1), (1, 8), (16, 1)],
            "alt_pool_win": None,  # Model type without alternation
            "sum_pooling": True
        }
    }
    spn = concat_rec(spn_def, save_folder)

logger.info("Built SPN with {} layers".format(len(spn.layers)))
logger.info("--> Duration: {:.4f}".format(timers.tac()))


# COMET ML
# ---------
if use_comet:
    from comet_ml import Experiment
    experiment = None
    experiment = Experiment(api_key=comet_api,
                            project_name=comet_project,
                            workspace=comet_workspace)


mult_params = {
    "seed": SEED,
    "name": exp_name,
    "batch_size": batch_size,
    "img_height": img_height,
    "img_width": img_width,
    "img_channel": img_channel,
    "leaf_components": leaf_components,
    "first_sum_channel": first_sum_channel,
    "learning_rate": learning_rate,
    "minibatch_size": minibatch_size,
    "valid_amount": valid_amount,
    "epochs": epochs,
    "model_type": model_type,
    "complete_side": complete_side,
    "training_type": training_type,
    "inference_type": inference_type,
    # Specific for Tree models
    "pool_layer_amt": spn_def["product_layers"]["pooling_method"]["amount"],
    "tree_model_size": tree_model_size,
    "tree_model_alt_size": tree_model_alt_size,
}
if use_comet:
    experiment.log_multiple_params(mult_params)

# Log hyper-parameters
logger.info("Hyper-parameters:")
logger.info(mult_params)


# NLL TRAINNING
# -------------

forward = None
backward_masks = None

if training_type == "nll":
    timers.tic()
    logger.info("Compiling model.")

    forward = spn.compile(learning_rate=learning_rate,
                          optimizer="adam")

    logger.info("--> Duration: {:.4f}".format(timers.tac()))

    timers.tic()
    logger.info("Fitting.")

    feed_means_stds = {
        spn.leaf_layer.means: leaf_means,
        spn.leaf_layer.stds: leaf_stds}
    costs = []
    costs = spn.fit(train_data=train_data, epochs=epochs,
                    add_to_feed=feed_means_stds, minibatch_size=minibatch_size,
                    save_after=10, save_path=_path_sav_fold("models"))

    plot_cost_name = "{}_nll_costs".format(exp_name)
    costs = [float(c) for c in costs]
    plot_cost_graph(
        plot_cost_name, costs, "{}.png".format(_path_sav_fold(plot_cost_name)))

    logger.info("--> Duration: {:.4f}".format(timers.tac()))

    # Comet.ml
    if use_comet:
        for cost in costs:
            experiment.log_metric("cost", cost)

elif training_type == "adversarial":
    timers.tic()
    logger.info("Compiling adversarial model.")

    forward, backward_masks = spn.compile_adversarial(
        learning_rate=learning_rate)

    logger.info("--> Duration: {:.4f}".format(timers.tac()))

    timers.tic()
    logger.info("Fitting.")

    feed_means_stds = {
        spn.leaf_layer.means: leaf_means,
        spn.leaf_layer.stds: leaf_stds}
    disc_costs, gen_costs = spn.fit_adversarial(train_data=train_data,
                                                train_marg=train_data_marg,
                                                epochs=epochs,
                                                add_to_feed=feed_means_stds,
                                                minibatch_size=minibatch_size,
                                                save_after=10,
                                                save_path=_path_sav_fold(
                                                    "models"))
    plot_disc_cost_name = "{}_disc_costs".format(exp_name)
    plot_gen_cost_name = "{}_gen_costs".format(exp_name)
    disc_costs = [float(c) for c in disc_costs]
    gen_costs = [float(c) for c in gen_costs]
    plot_cost_graph(
        plot_disc_cost_name, disc_costs, "{}.png".format(
            _path_sav_fold(plot_disc_cost_name)))
    plot_cost_graph(
        plot_gen_cost_name, gen_costs, "{}.png".format(
            _path_sav_fold(plot_gen_cost_name)))

    logger.info("--> Duration: {:.4f}".format(timers.tac()))

    # Comet.ml
    if use_comet:
        for i in range(len(disc_costs)):
            experiment.log_metric("disc_cost", disc_costs[i])
            experiment.log_metric("gen_cost", gen_costs[i])

# Running Garbage Collector
collected = gc.collect()
logger.info("Garbage collecting after fitting: {}".format(collected))

# INFERENCE
# ----------

# MPE
if inference_type == "mpe":

    timers.tic()
    logger.info("MPE inference.")

    spn_input = spn.inputs_marg
    if backward_masks is None:
        spn_input = spn.inputs
        backward_masks = spn.build_backward_masks(forward)
    mpe_leaf = spn.build_mpe_leaf(backward_masks, replace_marg_vars=spn_input)

    feed_means_stds = {
        spn.leaf_layer.means: leaf_means,
        spn.leaf_layer.stds: leaf_stds}

    root_values = spn.forward_inference(
        forward, eval_data_marg, add_to_feed=feed_means_stds,
        alt_input=spn_input)

    root_value = -1.0 * np.mean(root_values)
    print('{"metric": "Val NLL", "value": %f}' % (root_value))

    mpe_assignment = spn.mpe_inference(
        mpe_leaf, eval_data_marg, add_to_feed=feed_means_stds,
        alt_input=spn_input)

    logger.info("--> Duration: {:.4f}".format(timers.tac()))

    # Running Garbage Collector
    collected = gc.collect()
    logger.info("Garbage collecting after MPE inference: {}".format(collected))

    # De-normalize results
    unorm_eval_data = eval_data
    if database.has_normalized:
        mpe_assignment = database.de_normalize(mpe_assignment)
        unorm_eval_data = database.de_normalize(eval_data)

    # apply gaussian filter with sigma = use_filter if it's not zero
    if not use_filter == 0:
        logger.info(
            "Post-processing with low-pass filter activated: {}".format(
                use_filter))
        from skimage import filters
        if complete_side == "left":
            mpe_assignment[:, :, :int(img_width / 2)] = filters.gaussian(
                mpe_assignment[:, :, :int(img_width / 2)], sigma=use_filter)

    # MSE
    timers.tic()
    logger.info("Computing Mean Square Error")

    save_imgs_path = _path_sav_fold("completions")
    mse = Database.mean_square_error(
        mpe_assignment, unorm_eval_data, save_imgs_path=save_imgs_path)

    logger.info("MSE: {}".format(mse))
    logger.info("--> Duration: {:.4f}".format(timers.tac()))
    print('{"metric": "MSE", "value": %f}' % (mse))

    # Comet.ml
    if use_comet:
        experiment.log_metric("mse", mse)
        for img_idx in range(unorm_eval_data.shape[0]):
            experiment.log_image("{}/{}.png".format(save_imgs_path, img_idx))

# SAMPLING
elif inference_type == "sampling":

    timers.tic()
    logger.info("Sampling inference.")

    spn_input = spn.inputs_marg
    if backward_masks is None:
        spn_input = spn.inputs
        backward_masks = spn.build_backward_masks(
            forward, sampling_amt=valid_amount)
    sampling_leaf = spn.build_sampling_leaf(backward_masks)

    feed_means_stds = {
        spn.leaf_layer.means: leaf_means,
        spn.leaf_layer.stds: leaf_stds}

    mpe_assignment = spn.sampling_inference(
        sampling_leaf, add_to_feed=feed_means_stds)

    logger.info("--> Duration: {:.4f}".format(timers.tac()))

    # De-normalize results
    if database.has_normalized:
        mpe_assignment = database.de_normalize(mpe_assignment)

    # apply gaussian filter with sigma = use_filter if it's not zero
    if use_filter > 0:
        logger.info(
            "Post-processing with low-pass filter activated: {}".format(
                use_filter))
        from skimage import filters
        mpe_assignment = filters.gaussian(mpe_assignment, sigma=use_filter)

    save_imgs_path = _path_sav_fold("samplings")
    for img_idx in range(mpe_assignment.shape[0]):
        Database.save_image("{}/{}.png".format(
            save_imgs_path, img_idx), mpe_assignment[img_idx])

    # Running Garbage Collector
    collected = gc.collect()
    logger.info("Garbage collecting after sampling inference: {}".format(
        collected))

    # Comet.ml
    if use_comet:
        for img_idx in range(mpe_assignment.shape[0]):
            experiment.log_image("{}/{}.png".format(save_imgs_path, img_idx))
