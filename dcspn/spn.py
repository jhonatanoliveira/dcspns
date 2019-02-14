"""Main SPN implementation."""
import tensorflow as tf
import numpy as np

from dcspn.layers import SumLayer, ProductLayer, LeafLayer
from dcspn.utilities import random_mini_batches, kahn_topsort
from dcspn import MARG_VAR_VAL, NEG_INF


class SumProductNetwork:
    """
    Base class for an SPN.

    Implement SPNs as Convolutional Neural Networks
    """

    def __init__(self, input_shape, seed=1234, name="SPN"):
        """
        Maintain a rooted graph of tensors.

        Both connection directions (top-down and bottom-up) are maintained
        for faster access.
        """
        self.layers_graph = {"children": {}, "parents": {}}
        self.layers = []
        self.root_layer = None
        self.leaf_layer = None
        # Arguments
        self.input_shape = input_shape
        self.seed = seed
        self.model_name = name
        # Tensorflow variables
        self.sess = tf.Session()
        self.inputs = tf.placeholder(
            name="Placeholder_Inputs",
            dtype=tf.float32,
            shape=[None, self.input_shape[0], self.input_shape[1],
                   self.input_shape[2]])
        self.saver = None
        self.inputs_labels = tf.placeholder(
            name="Placeholder_Inputs_Labels",
            dtype=tf.float32,
            shape=[None, self.input_shape[0], self.input_shape[1],
                   self.input_shape[2]])
        # Set random seeds
        tf.set_random_seed(self.seed)
        # Variables that can be created internally
        self.loss = None
        self.optimizer = None
        # Variables use din adversarial training
        self.mpe_leaf = None
        self.generator_loss = None
        self.inputs_marg = tf.placeholder(
            name="Placeholder_Inputs_Marg",
            dtype=tf.float32,
            shape=[None, self.input_shape[0], self.input_shape[1],
                   self.input_shape[2]])

    def set_leaf_layer(self, layer):
        """
        Set the leaf layer of the SPN.

        Parameters
        ----------
        layer: SPN layer

        See also
        --------
        Layer
        """
        self.leaf_layer = layer
        # Add edge from leaf layer to inputs.
        self.layers_graph["children"][self.leaf_layer] = [self.inputs]
        self.layers_graph["parents"][self.inputs] = [self.leaf_layer]

    def set_root_layer(self, layer):
        """
        Set the root layer of the SPN.

        Parameters
        ----------
        layer: SPN layer

        See also
        --------
        Layer
        """
        self.root_layer = layer

    def add_layer(self, layer):
        """
        Add an edge in the tensor graph.

        Parameters
        ----------
        layer: SPN layer

        See also
        --------
        Layer
        """
        self.layers.append(layer)

    def add_forward_layer_edge(self, layer_1, layer_2):
        """
        Add a direct edge in the rooted tensor graph.

        For instance, if layer_2 is after layer_1 in the forward
        computation phase, then the direct edge is from layer_2 to layer_1.
        If layers do no exist in the graph, they are added first.

        Parameters
        ----------
        layer_1: SPN layer
        layer_2: SPN layer

        See also
        --------
        Layer
        """
        if layer_1 not in self.layers:
            self.add_layer(layer_1)
        if layer_2 not in self.layers:
            self.add_layer(layer_2)

        if layer_2 not in self.layers_graph["children"]:
            self.layers_graph["children"][layer_2] = [layer_1]
        else:
            self.layers_graph["children"][layer_2].append(layer_1)

        if layer_1 not in self.layers_graph["parents"]:
            self.layers_graph["parents"][layer_1] = [layer_2]
        else:
            self.layers_graph["parents"][layer_1].append(layer_2)

    def build_forward(self, forward_input, reuse=False):
        """
        Forward inference phase in SPN.

        Multiple branches of tensors are allowed.

        Parameters
        ----------
        forward_input: Tensor
            Input for constructing the forward tensor graph.
            Usually a placeholder.
        reuse: Boolean
            If true, reuse weights from previous constructed
            forward graph.
        """
        if self.leaf_layer is None or self.root_layer is None:
            raise ValueError("Leaf and Root layers must be set in SPN \
                before building forward pass.")
        # Initialize forward propagation with inputs (placeholder)
        forward = {
            forward_input: {
                "output_shape": self.input_shape,
                "output": forward_input
            }
        }
        # Perform forward inference
        rev_top_sort = reversed(kahn_topsort(self.layers_graph["children"]))
        for curr_layer in rev_top_sort:
            # Joining child layers depending on the current layer type
            # in order to maintain completeness and decomposability
            children_output = []
            children_output_shape = []
            if isinstance(curr_layer, SumLayer):
                children_out_channels = 0
                for cidx, child in enumerate(
                        self.layers_graph["children"][curr_layer]):
                    # Add channels
                    child_ch_amt = forward[child]["output_shape"][2]
                    children_output_shape = forward[
                        child]["output_shape"]
                    children_out_channels += child_ch_amt
                    children_output.append(forward[
                        child]["output"])
                # Concatenate children outputs in the
                # out channel axis (from behind) for
                # satisfying completeness
                children_output = tf.concat(children_output, axis=3) \
                    if len(children_output) > 1 else children_output[0]
                # Only channel (depth) is altered when joining children
                # layers in a SumLayer
                children_output_shape[2] = children_out_channels
            elif isinstance(curr_layer, ProductLayer):
                children_out_width = 0
                for child in self.layers_graph["children"][curr_layer]:
                    # OBS: current implementation only allows
                    # sideways connectivity so the height of all
                    # children should be the same.
                    child_shape = forward[
                        child]["output_shape"]
                    if len(children_output_shape) > 0 and(
                            (children_output_shape[0] != child_shape[0]) or
                            (children_output_shape[2] != child_shape[0])):
                        raise ValueError("Children of product layer should\
                            have only different width.")
                    # Add width
                    children_output_shape = child_shape
                    children_out_width += child_shape[1]
                    children_output.append(forward[
                        child]["output"])
                # Concatenate children outputs in the width channel
                # (sideways) for decomposability maintenance.
                children_output = tf.concat(children_output, axis=2) \
                    if len(children_output) > 1 else children_output[0]
                # Only the width axis is altered
                children_output_shape[1] = children_out_width
            elif isinstance(curr_layer, LeafLayer):
                # OBS: Currently, leaf layers should have only 1 input
                # layer, but future versions might allow multiples.
                children_output_shape = self.input_shape
                children_output = forward_input
            # Process current layer
            curr_layer.build(children_output_shape, reuse=reuse)
            curr_layer_output = curr_layer.call(children_output)
            curr_layer_output_shape = curr_layer.compute_output_shape(
                children_output_shape)
            # Save current layer in forward computational graph
            forward[curr_layer] = {}
            forward[
                curr_layer]["output_shape"] = curr_layer_output_shape
            forward[
                curr_layer]["output"] = curr_layer_output
        return forward

    def build_backward_masks(self, forward, sampling_amt=None):
        """
        Backward propagation for 1's and 0's masks.

        This process selects which node is active or inactive in the MPE
        inference pass.

        Parameters
        ----------
        forward: dict
            Dictionary containing the output of forward propagation
            with all tensors built.
        sampling_amt: int
            If sampling, instead of MPE, this is the amount of samples.
        """
        is_sampling = False if sampling_amt is None else True
        # Initialize backward mask propagation with
        # root layer receiving mask all ones.
        mask_amount = sampling_amt if is_sampling else tf.shape(
            forward[self.root_layer]["output"])[0]
        backward_masks = {
            None: {
                "output_mask": tf.ones(shape=[mask_amount, 1, 1, 1])
            }
        }
        # Perform forward inference
        top_sort = kahn_topsort(self.layers_graph["children"])
        for curr_layer in top_sort:
            # Joining parents masks depending on the parent layer type
            parents_masks = None
            # Captures the root (no parent)
            # Assuming only one (and unique) root layer
            if curr_layer not in self.layers_graph["parents"] or\
                    len(self.layers_graph["parents"][curr_layer]) == 0:
                parents_masks = backward_masks[
                    None]["output_mask"]
            else:
                # Each child has a slice from its parent's mask
                # Here, we find the current layer slice by searching
                # and considering each other children.
                # For SumLayers this slice is in the channel axis while
                # for ProductLayers it is in the width axis.
                for parent in self.layers_graph["parents"][curr_layer]:
                    if isinstance(parent, SumLayer):
                        slice_start_pos = 0
                        for sibling in self.layers_graph[
                                "children"][parent]:
                            if sibling != curr_layer:
                                # the output shape has
                                # format [height, width, channel]
                                slice_start_pos += forward[
                                    sibling]["output_shape"][2]
                            else:
                                break
                        slice_stop_pos = slice_start_pos + forward[
                            curr_layer]["output_shape"][2]
                        parent_mask = backward_masks[
                            parent]["output_mask"]
                        slice_mask = parent_mask[
                            :, :, :, slice_start_pos:slice_stop_pos]
                        # Adding masks can simulate a logical-OR operation
                        # since greater than 0 means 1
                        parents_masks = parents_masks + slice_mask\
                            if parents_masks is not None else slice_mask
                    elif isinstance(parent, ProductLayer):
                        slice_start_pos = 0
                        for sibling in self.layers_graph[
                                "children"][parent]:
                            if sibling != curr_layer:
                                # the output shape has
                                # format [height, width, channel]
                                slice_start_pos += forward[
                                    sibling]["output_shape"][1]
                            else:
                                break
                        slice_stop_pos = slice_start_pos + forward[
                            curr_layer]["output_shape"][1]
                        parent_mask = backward_masks[
                            parent]["output_mask"]
                        slice_mask = parent_mask[
                            :, :, slice_start_pos:slice_stop_pos, :]
                        # Adding masks can simulate a logical-OR operation
                        # since greater than 0 means 1
                        parents_masks = parents_masks + slice_mask\
                            if parents_masks is not None else slice_mask
            parents_masks = tf.where(
                tf.greater(parents_masks, 0),
                tf.ones(shape=tf.shape(parents_masks)),
                parents_masks)
            # Process current layer mask
            curr_layer_output_mask = None
            # Leaf layers do not have children, thus can not
            # compute mask to next (previous) layers
            if isinstance(curr_layer, LeafLayer):
                curr_layer_output_mask = parents_masks
            else:
                if is_sampling:
                    curr_layer_output_mask = curr_layer.sampling_call(
                        parents_masks)
                else:
                    curr_layer_output_mask = curr_layer.masks_call(
                        parents_masks)
            # Save current layer children masks
            backward_masks[curr_layer] = {}
            backward_masks[
                curr_layer]["output_mask"] = curr_layer_output_mask
        return backward_masks

    def build_mpe_leaf(self, backward_masks, replace_marg_vars=None):
        """
        Find MPE assignment of the leaf layer, given masks propagation.

        Parameters
        ----------
        backward_masks: dict
            Output from building backward mask. It's a dictionary
            containing the masks tensors built.
        replace_marg_vars: Tensor
            Input tensor for filling non marginalized variables.
        """
        leaf_masks = backward_masks[self.leaf_layer]["output_mask"]
        leaf_mpe = self.leaf_layer.mpe_values()
        poss_assignments = tf.multiply(leaf_mpe, leaf_masks)
        # Only one leaf in the channel axis should have a value, the others
        # should be zero. Summing over them gets this only value.
        mpe_assignment = tf.reduce_sum(
            poss_assignments, axis=3, keepdims=True)
        if replace_marg_vars is not None:
            mpe_assignment = tf.where(
                tf.equal(replace_marg_vars, MARG_VAR_VAL),
                mpe_assignment, replace_marg_vars)
        return mpe_assignment

    def build_sampling_leaf(self, backward_masks):
        """
        Find sampling assignment for leaf layer, given masks propagation.

        Parameters
        ----------
        backward_masks: dict
            Output from building backward mask. It's a dictionary
            containing the masks tensors built.
        """
        leaf_masks = backward_masks[self.leaf_layer]["output_mask"]
        leaf_sampling = self.leaf_layer.sampling_values(
            tf.shape(leaf_masks)[0])
        poss_assignments = tf.multiply(leaf_sampling, leaf_masks)
        # Only one leaf in the channel axis should have a value, the others
        # should be zero. Summing over them gets this only value.
        sampling_assignment = tf.reduce_sum(
            poss_assignments, axis=3, keepdims=True)
        return sampling_assignment

    def compile(self, optimizer="adam", learning_rate=0.001):
        """
        Compile convolutional SPN.

        Parameters
        ----------
        loss_type: "nll" | "mse" | "adversarial"
            Builds Negative LogLikelihood (NLL), Mean Square Error,
            or Adversarial loss function.
        optimizer: "adam"
            Type of tensorflow optimizer
        learning_rate: float
            Learning rate for training

        """
        # Loss function
        forward = self.build_forward(self.inputs)
        root_value = forward[self.root_layer]["output"]
        self.loss = tf.reduce_mean(-1.0 * root_value, axis=0)
        # Optimizer
        if optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.loss)
        elif optimizer == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate).minimize(self.loss)
        else:
            raise ValueError("Invalid optimizer.")

        return forward

    def fit(self, train_data, train_labels=None, epochs=50, minibatch_size=64,
            logger="logger", add_to_feed=None,
            keep_learning=False, save_after=None, save_path=None):
        """
        Fit data to compiled network.

        Parameters
        ----------
        train_data: numpy array
            Data to be fitted
        train_labels: numpy array
            Data for learning with error
        epochs: int
            Amount of epochs in learning
        minibatch_size: int
            Size of minibatch
        """
        import logging

        logger = logging.getLogger(logger)
        logger.setLevel(logging.DEBUG)
        logger.info("Fitting: start")

        # Settings
        num_instances = train_data.shape[0]
        if not minibatch_size:
            _poss_size = int(num_instances / 4)
            minibatch_size = _poss_size if _poss_size > 0 else num_instances

        # Initialization
        if not keep_learning:
            logger.info("Fitting: Initializing weights")
            init = tf.global_variables_initializer()
            self.sess.run(init)

        costs = []
        _seed = self.seed
        # Do the training loop
        for epoch in range(epochs):

            logger.info("Fitting: running epoch {}".format(epoch))

            minibatch_cost = 0.
            # number of minibatches of size minibatch_size in the train set
            num_minibatches = int(num_instances / minibatch_size)
            num_minibatches = num_minibatches if num_minibatches > 0 else 1
            _seed += 1
            minibatches = random_mini_batches(
                train_data, train_labels, minibatch_size, _seed)

            for i, minibatch in enumerate(minibatches):
                # Run the session to execute the optimizer and the cost,
                # the feedict should contain a minibatch for (train_data,Y).
                minibatch_X, minibatch_Y = minibatch
                _feed_dict = {self.inputs: minibatch_X}
                if add_to_feed is not None:
                    _feed_dict.update(add_to_feed)
                # Run training iteration
                _, temp_cost = self.sess.run(
                    [self.optimizer, self.loss],
                    feed_dict=_feed_dict)

                minibatch_cost += temp_cost / num_minibatches
            # Save model after amount iteration
            if save_after is not None and save_path is not None\
                    and np.mod(epoch, save_after) == 0:
                self.save(save_path, epoch)
            # Print the cost every n epoch
            print('{"metric": "NLL",\
                "value": %f}' % (minibatch_cost))
            costs.append(minibatch_cost)
            logger.info("Fitting: cost computed {}".format(
                float(minibatch_cost)))
        # Save last one
        if save_path is not None:
            self.save(save_path, epochs)

        return costs

    def compile_adversarial(self, learning_rate=0.001):
        """Compile adversarial."""
        # Generator
        forward_mpe = self.build_forward(self.inputs_marg)
        backward_masks = self.build_backward_masks(forward_mpe)
        self.mpe_leaf = self.build_mpe_leaf(backward_masks,
                                            replace_marg_vars=self.inputs_marg)
        # Discriminator Real
        forward_real = self.build_forward(self.inputs, reuse=True)
        root_value_real = forward_real[self.root_layer]["output"]
        # Discriminator Fake
        forward_fake = self.build_forward(self.mpe_leaf, reuse=True)
        root_value_fake = forward_fake[self.root_layer]["output"]

        # Loss functions
        self.disc_loss_real = tf.reduce_mean(-1.0 * root_value_real, axis=0)
        self.disc_loss_fake = tf.reduce_mean(
            NEG_INF - 1.0 * root_value_fake, axis=0)
        self.discriminator_loss = self.disc_loss_real + self.disc_loss_fake
        self.generator_loss = tf.reduce_mean(-1.0 * root_value_fake, axis=0)
        # Optimizer
        self.d_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.discriminator_loss)
        self.g_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.generator_loss)

        return forward_mpe, backward_masks

    def fit_adversarial(self, train_data, train_marg, epochs=50,
                        minibatch_size=64, logger="logger", add_to_feed=None,
                        keep_learning=False,
                        save_after=None, save_path=None):
        """Fit data to compiled adversarial network."""
        import logging

        logger = logging.getLogger(logger)
        logger.setLevel(logging.DEBUG)
        logger.info("Fitting: Adversarial")

        # Settings
        num_instances = train_data.shape[0]
        if not minibatch_size:
            _poss_size = int(num_instances / 4)
            minibatch_size = _poss_size if _poss_size > 0 else num_instances

        # Initialization
        if not keep_learning:
            logger.info("Fitting: Initializing weights")
            init = tf.global_variables_initializer()
            self.sess.run(init)

        disc_real_costs = []
        disc_fake_costs = []
        disc_costs = []
        gen_costs = []
        _seed = self.seed
        # Do the training loop
        for epoch in range(epochs):

            logger.info("Fitting: running epoch {}".format(epoch))

            disc_real_minibatch_cost = 0.
            disc_fake_minibatch_cost = 0.
            disc_minibatch_cost = 0.
            gen_minibatch_cost = 0.
            # number of minibatches of size minibatch_size in the train set
            num_minibatches = int(num_instances / minibatch_size)
            num_minibatches = num_minibatches if num_minibatches > 0 else 1
            _seed += 1
            minibatches = random_mini_batches(
                train_data, train_marg, minibatch_size, _seed)

            for i, minibatch in enumerate(minibatches):
                # Run the session to execute the optimizer and the cost,
                # the feedict should contain a minibatch for (train_data,Y).
                minibatch_X, minibatch_Y = minibatch

                _feed_dict = {self.inputs: minibatch_X,
                              self.inputs_marg: minibatch_Y}
                if add_to_feed is not None:
                    _feed_dict.update(add_to_feed)
                if minibatch_Y is not None:
                    _feed_dict[self.inputs_labels] = minibatch_Y

                # Train discriminator
                _, disc_temp_cost, disc_real, disc_fake = self.sess.run(
                    [self.d_optim, self.discriminator_loss,
                     self.disc_loss_real, self.disc_loss_fake],
                    feed_dict=_feed_dict)

                # Train generator
                _, gen_temp_cost, mpe_assignment = self.sess.run(
                    [self.g_optim, self.generator_loss, self.mpe_leaf],
                    feed_dict=_feed_dict)

                disc_real_minibatch_cost += disc_real / num_minibatches
                disc_fake_minibatch_cost += disc_fake / num_minibatches
                disc_minibatch_cost += disc_temp_cost / num_minibatches
                gen_minibatch_cost += gen_temp_cost / num_minibatches
            # Save model after amount iteration
            if save_after is not None and save_path is not None\
                    and np.mod(epoch, save_after) == 0:
                self.save(save_path, epoch)
            # Print the cost every n epoch
            print('{"metric": "Disc Real Cost",\
                "value": %f}' % (disc_real_minibatch_cost))
            print('{"metric": "Disc Fake Cost",\
                "value": %f}' % (disc_fake_minibatch_cost))
            print('{"metric": "Discriminator Cost",\
                "value": %f}' % (disc_minibatch_cost))
            print('{"metric": "Generator Cost",\
                "value": %f}' % (gen_minibatch_cost))
            disc_real_costs.append(disc_minibatch_cost)
            disc_fake_costs.append(disc_minibatch_cost)
            disc_costs.append(disc_minibatch_cost)
            gen_costs.append(gen_minibatch_cost)
            logger.info("Fitting: disc real cost computed {}".format(
                disc_real_minibatch_cost))
            logger.info("Fitting: disc fake cost computed {}".format(
                disc_fake_minibatch_cost))
            logger.info("Fitting: disc cost computed {}".format(
                disc_minibatch_cost))
            logger.info("Fitting: gen cost computed {}".format(
                gen_minibatch_cost))
        # Save last one
        if save_path is not None:
            self.save(save_path, epochs)

        return disc_costs, gen_costs

    def forward_inference(self, forward, data, add_to_feed=None,
                          alt_input=None):
        """
        Perform forward inference.

        Parameters
        ----------
        data: numpy array
            Inference data with shape [N, H, W, C]
        """
        spn_input = self.inputs if alt_input is None else alt_input
        _feed_dict = {spn_input: data}
        if add_to_feed:
            _feed_dict.update(add_to_feed)
        root_output = forward[self.root_layer]["output"]
        root_value = self.sess.run(root_output, feed_dict=_feed_dict)
        return root_value

    def mpe_inference(self, mpe_leaf, data, add_to_feed=None, alt_input=None):
        """
        Perform MPE inference and return MPE assignment at leaf layer.

        Parameters
        ----------
        data: numpy array
            Inference data with shape [N, H, W, C]
        """
        spn_input = self.inputs if alt_input is None else alt_input
        _feed_dict = {spn_input: data}
        if add_to_feed:
            _feed_dict.update(add_to_feed)
        mpe_assignment = self.sess.run(
            mpe_leaf, feed_dict=_feed_dict)
        return mpe_assignment

    def sampling_inference(self, sampling_leaf, add_to_feed=None):
        """
        Perform sampling inference and return a assignment at leaf layer.

        Parameters
        ----------
        data: numpy array
            Inference data with shape [N, H, W, C]
        """
        _feed_dict = {}
        if add_to_feed:
            _feed_dict.update(add_to_feed)
        sampling_assignment = self.sess.run(
            sampling_leaf, feed_dict=_feed_dict)
        return sampling_assignment

    def draw_conv_spn(self, path):
        """Print the DCSPN graph in a dot file."""
        import networkx as nx
        from dcspn.layers import Layer
        graph = nx.DiGraph()

        edges = []
        for parent in self.layers_graph["children"]:
            for child in self.layers_graph["children"][parent]:
                if isinstance(child, Layer):
                    edges.append((parent, child))
                else:
                    edges.append(
                        (parent, "Placeholder {}".format(self.input_shape)))
        graph.add_edges_from(edges)

        nx.drawing.nx_pydot.write_dot(graph, path)

    def fit_adversarial_context_perception(self, feed_mask,
                                           eval_data_marg, forward, n_iter,
                                           add_to_feed=None,
                                           lam=0.1, lr=0.01,
                                           momentum=0.01):
        """
        Find better input for MPE completion.

        IMPORTANT NOTE: Will leave this for now - for future investigations,
        but this function does *not* work properly. Computed gradients are
        always zero (since we're computing them from self.inputs).
        """
        if self.mpe_leaf is None or self.generator_loss is None:
            raise Exception("Compile adversarial must be performed first.")
        # Compile loss function
        mask = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.input_shape[0], self.input_shape[1],
                   self.input_shape[2]])
        # Context
        contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(mask, self.mpe_leaf) -
                       tf.multiply(mask, self.inputs))), axis=1)
        # Perception
        # perceptual_loss = self.generator_loss
        complete_loss = contextual_loss

        # Fit
        grad_complete_loss = tf.gradients(complete_loss, self.inputs)
        zhats = np.copy(eval_data_marg)
        v = 0
        for i in range(n_iter):
            _feed_dict = {
                self.inputs: eval_data_marg,
                mask: feed_mask
            }
            if add_to_feed is not None:
                _feed_dict.update(add_to_feed)
            loss, g, G_imgs = self.sess.run(
                [complete_loss, grad_complete_loss, self.mpe_leaf],
                feed_dict=_feed_dict)
            # Gradient descendant
            v_prev = np.copy(v)
            v = momentum * v - lr * g[0]
            delta = -momentum * v_prev + (1 + momentum) * v
            zhats += delta
            zhats = np.clip(zhats, -1, 1)

        return zhats

    def save(self, checkpoint_dir, step):
        """Save variables from TF graph in directory."""
        import os.path
        self.saver = tf.train.Saver(max_to_keep=1) if self.saver is None\
            else self.saver
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        """Load variables to TF graph from directory."""
        self.saver = tf.train.Saver(max_to_keep=1) if self.saver is None\
            else self.saver
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Fix address (in case load check point is diff from saving one)
            addr = ckpt.model_checkpoint_path[
                ckpt.model_checkpoint_path.rfind("/"):]
            full_addr = checkpoint_dir + addr
            self.saver.restore(self.sess, full_addr)
            return True
        return False
