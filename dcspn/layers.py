"""Layer implementations for SPN."""
from dcspn import EPSILON, MARG_VAR_VAL, NEG_INF

import numpy as np

import tensorflow as tf


class Layer:
    """
    Base layer class.

    See Also
    --------
    SumLayer
    ProductLayer
    LeafLayer
    """

    layer_counter = 0

    def __init__(self, name=None):
        """
        Constructor.

        Parameters
        ----------
        name : str | int
            A unique name for this layer
        """
        self.name = name if name else "Layer_{}".format(Layer.layer_counter)
        Layer.layer_counter += 1

    def build(self, input_shape, reuse=False):
        """
        Setup method to build things necessary for running the layer.

        For instance, the weights.
        This method must set self.built = True at the end,
        which can be done by calling super([Layer], self).build().

        Parameters
        ----------
        input_shape : tuple
            Format (height, width, channel)
        """
        self.built = True

    def call(self, input_tensor):
        """
        Logic of the layer.

        Receives the input tensor from previous computation.

        Parameters
        ----------
        input_tensor : tensorflow tensor
            Shape [batch, height, width, channel]
        """
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        """
        General formula for the output shape of the layer.

        In case the layer modifies the shape of the input tensor,
        compute here the new shape.

        Parameters
        ----------
        input_shape : tuple
            Format [height, width, channel]
        """
        raise NotImplementedError

    def masks_call(self, input_mask_tensor):
        """
        Compute output mask given mask for this layer.

        A mask can be propagated backward in the tensor graph
        in order to simulate a Viterbi backtrack. In MPE, this is done
        by following the maximum value sum child.

        Parameters
        ----------
        input_mask_tensor : tensor
            Shape [N, H, W, C] with mask for current layer
        """
        raise NotImplementedError

    def sampling_call(self, input_mask_tensor):
        """
        Compute output mask randomly.

        A mask can be propagated backward in the tensor graph
        in order to simulate a Viterbi backtrack. In sampling,
        we do this in a random fashion.

        Parameters
        ----------
        input_mask_tensor : tensor
            Shape [N, H, W, C] with mask for current layer
        """
        raise NotImplementedError

    def __repr__(self):
        """For printing the layer."""
        extra = None
        layer_type = self.__class__.__name__
        extra = ""
        return "{}-{}-{}".format(self.name, layer_type, extra)


class SumLayer(Layer):
    """
    Sum layer implements convolutions with or without sharing parameters.

    Convolution filters are 1 by 1, then only elements (nodes) of the same
    scope are involved.
    Thus, this layer maintain completeness of the SPN.
    Computation are done in log-space.
    """

    def __init__(self, out_channels, hard_inference=False,
                 share_parameters=False, initializer="uniform", **kwargs):
        """
        Constructor for sum layer.

        Parameters
        ----------
        out_channels : int
            Depth of output tensor
        hard_inference: boolean
            If true, then the value of sum node is the max child,
            instead of the sum of its children
        share_parameters: boolean
            If true, then use a Conv2D implementation;
            else, each sum node has its own set of parameters.
        initializer: "glorot" | "random"
            Glorot uniform or random uniform between 0 and 1
        """
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.share_parameters = share_parameters
        self.initializer = initializer
        self.hard_inference = hard_inference
        # Variables created during execution
        self.weights = None
        self.linears = None

    def build(self, input_shape, reuse=False):
        """
        Build the sum layer weights.

        Shape format as [height, width, in_channels, out_channels]
        """
        chosen_init = None
        if self.initializer == "glorot":
            chosen_init = tf.glorot_uniform_initializer()
        elif self.initializer == "uniform":
            chosen_init = tf.initializers.random_uniform(minval=0, maxval=1)

        _var_shape = None
        if self.share_parameters:
            _var_shape = [1, 1, input_shape[2], self.out_channels]
        else:
            _var_shape = [input_shape[0], input_shape[1],
                          input_shape[2], self.out_channels]
        with tf.variable_scope("SumLayer", reuse=reuse):
            self.weights = tf.get_variable(
                "Weights_{}".format(self.name),
                shape=_var_shape,
                initializer=chosen_init)

        self.input_shape = input_shape

        super().build(input_shape)

    def call(self, input_tensor):
        """
        Compute sum layer value using two possible ways.

        First, using a normal convolutional approach with sharing parameters.
        Second, each element (sum node) has its own set of parameters.
        """
        # If negative weights, project them to zero
        # but a small constant is added to avoid zero weights.
        projected_weights = tf.add(
            tf.nn.relu(tf.subtract(self.weights, EPSILON)), EPSILON)
        normalization_const = tf.reduce_sum(
            projected_weights, axis=2, keepdims=True)
        # Duplicates weights in the in_channel axis
        # to avoid wrong broadcasting
        # R.I.P. FULANO
        normalization_consts = tf.tile(
            normalization_const, [1, 1, tf.shape(projected_weights)[2], 1])
        normalized_weights = tf.divide(projected_weights, normalization_consts)

        log_weights = tf.log(normalized_weights)

        # Sharing parameters uses a normal convolution 2D implementation with
        # a sliding filter passing through the input tensor per output channel.
        # Non sharing parameter uses only one filter with the same input
        # tensor size per output channel.
        output_tensor = None
        self.linears = []
        if self.share_parameters:
            strides = [1, 1, 1, 1]
            padding = "VALID"
            # Custom implementation of a Conv2D output
            # from stackoverflow
            # //:questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
            # We use a custom implementation instead of the tensorflow one
            # so the linear part (used in MPE later on) can be saved
            # Besides, the weights and tensor values are in log-space
            # meaning Conv2D should add instead of multiply
            filter_height = 1
            filter_width = 1
            in_channels = self.input_shape[2]
            out_channels = self.out_channels
            ix_height = int(input_tensor.shape[1])
            ix_width = int(input_tensor.shape[2])
            ix_channels = int(input_tensor.shape[3])
            flat_w = tf.reshape(
                log_weights,
                shape=[filter_height * filter_width * in_channels,
                       out_channels])
            patches = tf.extract_image_patches(
                input_tensor,
                ksizes=[1, filter_height, filter_width, 1],
                strides=strides,
                rates=[1, 1, 1, 1],
                padding=padding
            )
            patches_reshaped = tf.reshape(
                patches,
                shape=[-1, ix_height, ix_width,
                       filter_height * filter_width * ix_channels])
            feature_maps = []
            for i in range(out_channels):
                linear = tf.add(flat_w[:, i], patches_reshaped)
                self.linears.append(linear)
                feature_map = None
                feature_map = tf.reduce_max(linear, axis=3, keepdims=True)\
                    if self.hard_inference else tf.reduce_logsumexp(
                        linear, axis=3, keepdims=True)
                feature_maps.append(feature_map)
            output_tensor = tf.concat(feature_maps, axis=3)
        else:
            feature_maps = []
            for ch in range(self.out_channels):
                ch_weights = log_weights[:, :, :, ch]
                linear = tf.add(input_tensor, ch_weights)
                self.linears.append(linear)
                feature_map = tf.reduce_max(linear, axis=3, keepdims=True)\
                    if self.hard_inference else tf.reduce_logsumexp(
                        linear, axis=3, keepdims=True)
                feature_maps.append(feature_map)
            output_tensor = tf.concat(feature_maps, axis=3) \
                if len(feature_maps) > 1 else feature_maps[0]

        return output_tensor

    def compute_output_shape(self, input_shape):
        """
        Sum layers should only change the output channel of tensors.

        Otherwise, completeness of an SPN can be compromised.
        Height and width are maintained.
        """
        return [input_shape[0], input_shape[1], self.out_channels]

    def masks_call(self, input_mask_tensor):
        """Sum layer mask logic."""
        # Max child of a sum node is chosen
        layer_output_mask = None
        is_first = True
        for i, linear in enumerate(self.linears):
            chosen_child = tf.argmax(
                linear, axis=3, output_type=tf.int32)
            possible_mask = tf.one_hot(chosen_child,
                                       tf.shape(linear)[3], axis=-1)
            slice_in_mask = tf.reshape(
                input_mask_tensor[:, :, :, i],
                shape=(tf.shape(input_mask_tensor)[0],
                       tf.shape(input_mask_tensor)[1],
                       tf.shape(input_mask_tensor)[2],
                       1))
            real_mask = tf.multiply(possible_mask, slice_in_mask)
            if is_first:
                layer_output_mask = real_mask
                is_first = False
            else:
                # Adding mask values simulate a logical OR operation
                layer_output_mask = tf.add(layer_output_mask, real_mask)
        # To create a 0's and 1's tensor, a division by the amount
        # of assigned mask per sum child.
        output_mask_tensor = tf.where(
            tf.equal(layer_output_mask, 0),
            layer_output_mask,
            layer_output_mask / layer_output_mask)

        return output_mask_tensor

    def sampling_call(self, input_mask_tensor):
        """Randomly select a sum child."""
        # Random child of a sum node is chosen
        layer_output_mask = None
        is_first = True
        for i in range(self.out_channels):
            dist = tf.distributions.Categorical(
                probs=self.weights[:, :, :, i])
            chosen_child = dist.sample(sample_shape=[
                tf.shape(input_mask_tensor)[0]])
            possible_mask = tf.one_hot(chosen_child,
                                       self.input_shape[2], axis=-1)
            slice_in_mask = tf.reshape(
                input_mask_tensor[:, :, :, i],
                shape=(tf.shape(input_mask_tensor)[0],
                       tf.shape(input_mask_tensor)[1],
                       tf.shape(input_mask_tensor)[2],
                       1))
            real_mask = tf.multiply(possible_mask, slice_in_mask)
            if is_first:
                layer_output_mask = real_mask
                is_first = False
            else:
                # Adding mask values simulate a logical OR operation
                layer_output_mask = tf.add(layer_output_mask, real_mask)
        # To create a 0's and 1's tensor, a division by the amount
        # of assigned mask per sum child.
        output_mask_tensor = tf.where(
            tf.equal(layer_output_mask, 0),
            layer_output_mask,
            layer_output_mask / layer_output_mask)

        return output_mask_tensor

    def __repr__(self):
        """For printing the layer."""
        extra = None
        layer_type = self.__class__.__name__
        extra = self.out_channels
        return "{}-{}-{}".format(self.name, layer_type, extra)


class ProductLayer(Layer):
    """
    Product layer with computation done in log-space.

    Pooling window does not intersect each other when sliding
    in order to maintain decomposability of the SPN.
    """

    def __init__(self, pooling_size, sum_pooling=True,
                 batch_normalization=False, **kwargs):
        """Product layer constructor."""
        super().__init__(**kwargs)
        self.pooling_size = pooling_size
        self.sum_pooling = sum_pooling
        self.batch_normalization = batch_normalization

    def build(self, input_shape, reuse=False):
        """
        Save input shape for possible backward masks propagation.

        See also
        --------
        masks_call
        """
        self.input_shape = input_shape

        super().build(input_shape)

    def call(self, input_tensor):
        """
        Compute value of product nodes.

        This can be done by using a average pooling layer.
        In reality, a sum pooling layer is the correct way, but
        this can be done by multiplying (optionally) by a constant.
        """
        output_tensor = tf.nn.avg_pool(
            input_tensor,
            ksize=[1, self.pooling_size[0], self.pooling_size[1], 1],
            strides=[1, self.pooling_size[0], self.pooling_size[1], 1],
            padding="VALID")

        # Multiplying by the same amount the average-pooling operation
        # divided can simulate a sum-pooling layer.
        if self.sum_pooling:
            output_tensor = output_tensor * tf.constant(
                self.pooling_size[0] * self.pooling_size[1], dtype=tf.float32)

        # Batch normalization
        if self.batch_normalization:
            mean, var = tf.nn.moments(output_tensor, axes=[0])
            output_tensor = tf.nn.batch_normalization(output_tensor, mean=mean,
                                                      variance=var,
                                                      offset=None, scale=None,
                                                      variance_epsilon=EPSILON)

        return output_tensor

    def compute_output_shape(self, input_shape):
        """
        Product layer does not change the channel of input tensor.

        But height and width can be modified.
        Assumes non-intersecting pooling windows, i.e., strides
        are the same as pooling size.
        General output shape formula:
        [floor( (H - PH) / PH + 1 ), floor( (W - PW) / PW + 1 ), C]
        where PH and PW are pooling layer height and width, respectively.
        """
        return [int((input_shape[0] - self.pooling_size[0]) /
                    self.pooling_size[0] + 1),
                int((input_shape[1] - self.pooling_size[1]) /
                    self.pooling_size[1] + 1),
                input_shape[2]]

    def masks_call(self, input_mask_tensor):
        """
        Propagated to all children its input mask.

        This process is simulated by a resize operation
        using the nearest neighbour method.
        """
        output_mask_tensor = tf.image.resize_images(
            input_mask_tensor,
            size=[self.input_shape[0], self.input_shape[1]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return output_mask_tensor

    def sampling_call(self, input_mask_tensor):
        """Redirect to mask call, since it is the same process."""
        return self.masks_call(input_mask_tensor)

    def __repr__(self):
        """For printing the layer."""
        extra = None
        layer_type = self.__class__.__name__
        extra = self.pooling_size
        return "{}-{}-{}".format(self.name, layer_type, extra)


class LeafLayer(Layer):
    """Base implementation for  different types of leaves in an SPN."""

    def __init__(self, num_leaf_components, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.num_leaf_components = num_leaf_components

    def call(self, input_tensor):
        """
        Common logic for leaf implementations.

        Marginalize variables by setting leaf value to 1.

        See Also
        --------
        Layer
        """
        # Reshape input tensor for 2D only with shape [N, H, W]
        input_tensor = tf.reshape(input_tensor, shape=(
            tf.shape(input_tensor)[0],
            tf.shape(input_tensor)[1],
            tf.shape(input_tensor)[2]))

        # Call logic specific for leaf implementation
        leaf_output = self.leaf_call(input_tensor)

        # Repeats input tensor for shape [N, H, W, K]
        # where K is the number of leaf components
        reap_input_tensor = tf.stack(
            [input_tensor for _ in range(self.num_leaf_components)], axis=3)

        # Marginalize variables from SPN, whenever input
        # contains the MARG_VAR_VAL constant, by setting
        # correspondent leaf value to 1.
        layer_actv = tf.where(
            tf.equal(reap_input_tensor, MARG_VAR_VAL),
            tf.ones(shape=tf.shape(reap_input_tensor)),
            leaf_output)

        # Works as an activation function: all in log-space.
        layer_output = tf.log(layer_actv)
        _constant = tf.fill(
            dims=tf.shape(layer_output), value=tf.constant(
                NEG_INF, dtype=tf.float32))
        layer_output = tf.where(
            tf.equal(layer_actv, 0),
            _constant,
            layer_output)

        return layer_output

    def leaf_call(self, input_tensor):
        """
        Logic specific for Leaf Layers.

        See Also
        --------
        Layer
        """
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        """
        All leaf implementations create shape [H, W, K].

        Here, K is the number of leaf components.

        See Also
        --------
        Layer
        """
        return [input_shape[0], input_shape[1], self.num_leaf_components]

    def mpe_values(self):
        """Return MPE value for leaf implementation."""
        raise NotImplementedError


class GaussianLeafLayer(LeafLayer):
    """
    Leaves are represented as Gaussian distributions.

    Here, means and standard deviations are set using data from
    K components of the percentile of input data.
    """

    # User static variables for reusing means and standard deviation
    # placeholders
    saved_means = None
    saved_stds = None

    def build(self, input_shape, reuse=False):
        """
        Create placeholders for means and standard deviations.

        These placeholders must be set before training/inference.
        """
        if reuse:
            self.means = GaussianLeafLayer.saved_means
            self.stds = GaussianLeafLayer.saved_stds
        else:
            self.means = tf.placeholder(
                dtype=tf.float32,
                shape=[input_shape[0], input_shape[1],
                       self.num_leaf_components])
            self.stds = tf.placeholder(
                dtype=tf.float32,
                shape=[input_shape[0], input_shape[1],
                       self.num_leaf_components])
            # Save them for reuse
            GaussianLeafLayer.saved_means = self.means
            GaussianLeafLayer.saved_stds = self.stds

    def leaf_call(self, input_tensor):
        """Apply each component mean and std through input data."""
        leaf_values = []
        for leaf_ch in range(self.num_leaf_components):
            slice_means = self.means[:, :, leaf_ch]
            slice_stds = self.stds[:, :, leaf_ch]
            # Normal distribution
            leaf_value = tf.exp(
                -0.5 * (input_tensor - slice_means)**2 / (slice_stds**2)) / \
                ((2 * np.pi * (slice_stds**2))**0.5)
            leaf_values.append(leaf_value)
        output_tensor = tf.stack(leaf_values, axis=3)
        return output_tensor

    def mpe_values(self):
        """MPE value of a Gaussian is its mean."""
        return self.means

    def sampling_values(self, amt):
        """Sampling value of a Gaussian."""
        dist = tf.distributions.Normal(loc=self.means, scale=self.stds)
        return dist.sample(sample_shape=[amt])


class MultivariateGaussianLeafLayer(LeafLayer):
    """
    Leaves are represented as Multivariate Gaussian distributions.

    Here, means and standard deviations are set using data from
    K components of a Gaussian mixture model.
    """

    # User static variables for reusing means and standard deviation
    # placeholders
    saved_means = None
    saved_stds = None

    def __init__(self, num_leaf_components, cardinality, **kwargs):
        """Constructor."""
        super().__init__(num_leaf_components, **kwargs)
        self.cardinality = cardinality

    def build(self, input_shape, reuse=False):
        """
        Create placeholders for means and standard deviations.

        These placeholders must be set before training/inference.
        """
        if reuse:
            self.means = GaussianLeafLayer.saved_means
            self.stds = GaussianLeafLayer.saved_stds
        else:
            self.means = tf.placeholder(
                dtype=tf.float32,
                shape=[input_shape[0], input_shape[1],
                       self.num_leaf_components,
                       self.cardinality])
            self.stds = tf.placeholder(
                dtype=tf.float32,
                shape=[input_shape[0], input_shape[1],
                       self.num_leaf_components,
                       self.cardinality])
            # Save them for reuse
            GaussianLeafLayer.saved_means = self.means
            GaussianLeafLayer.saved_stds = self.stds

    def leaf_call(self, input_tensor):
        """Apply each component mean and std through input data."""
        multi_gauss = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.means,
            scale_diag=self.stds
        )
        output_tensor = multi_gauss.prob(input_tensor)
        return output_tensor

    def mpe_values(self):
        """MPE value of a Gaussian is its mean."""
        return self.means

    def sampling_values(self, amt):
        """Sampling value of a Gaussian."""
        dist = tf.distributions.Normal(loc=self.means, scale=self.stds)
        return dist.sample(sample_shape=[amt])
