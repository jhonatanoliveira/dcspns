"""Models for DCSPN"""
from dcspn.spn import SumProductNetwork
from dcspn.layers import SumLayer, ProductLayer, GaussianLeafLayer

from collections import deque

import networkx as nx
import numpy as np


class Tensor:
    __dict_tensors = {}

    tensor_id = -1

    def __init__(self, _ih, _iw, _region_r, _region_c, n_sum):
        # to avoid using another file like the Parameters
        self.image_height = _ih
        self.image_width = _iw
        self.channel = n_sum
        
        # size of regions grouped by this tensor
        self.baseRegion_h = _region_r
        self.baseRegion_w = _region_c
        # layer of the graph in which this tensor is
        self.at_layer = self.baseRegion_h + self.baseRegion_w - 2
        # if it's already part of another tensor
        self.concatened = False
        # if a tensor groups more tensors of the same type
        self.concat = False
        # if the operation of convolution was already applied to this tensor 
        self.convoluted = False
        # how many times the operation of polling was already applied to this tensor 
        self.polled = 0
        # tensor of type [] or ()
        self.tensor_type = None

        # self.id = id_num
        Tensor.tensor_id+=1
        self.id = Tensor.tensor_id
        Tensor.__dict_tensors[self.id] = self

        # generate regions of this tensor
        self._gen_regions()
        # C11 = np.random.rand(N,int(ih/r),int(iw/c),n_sum_leaf)

    def __str__(self):
        return 'Tensor : <%d,%d,%d>' % \
              (self.regions_ids.shape[0], self.regions_ids.shape[1], self.regions_ids.shape[2])

    def _gen_regions(self):
        # this numpy array can store the regions ids(import class Region)
        self.height = int(self.image_height/self.baseRegion_h)
        self.width = int(self.image_width/self.baseRegion_w)

        self.regions_ids = np.ndarray((self.height, self.width, self.channel))

    @staticmethod
    def getTensorId(_ih, _iw, _r, _c, _n_sum): # must do better to distinguish between equal tensors
        # using the cantor pairing function 3 times to encode the 4 numbers in 1
        first = 0.5*(_ih+_r)*(_ih+_r+1)+_r
        second = 0.5*(first+_iw)*(first+_iw+1)+_iw
        id_num = 0.5*(second+_c)*(second+_c+1)+_c
        if id_num - int(id_num) != 0.0:
            print('----------WARNING: FLOAT NUMBER AS ID')#it's not supposed to happen(cantor definition)
        id_num = int(id_num)
        # if tensor doesn't exist, create one
        if id_num not in Tensor.__dict_tensors:
            Tensor.__dict_tensors[id_num] = Tensor(id_num, _ih, _iw, _r, _c, _n_sum)
        return id_num

    @staticmethod
    def getTensor(id_num):
        return Tensor.__dict_tensors[id_num]

    @staticmethod
    def reset_tensors_dict():
        Tensor.__dict_tensors = {}
    
    @staticmethod
    def get_tensors_dict():
        return Tensor.__dict_tensors

def conv2d( _similar_tensors_ids, _tensor, _graph):
    curr_tensor = _tensor
    graph = _graph
    similar_tensors_ids = _similar_tensors_ids
    #the _weight_channel should be the number of tensors of the same type(k_h, k_w) of _tensor, which will be the _tensor depth(prev_sum_channel) - they were concat before
    #the channel of the concat tensor
    bef_concat_channel = curr_tensor.channel
    
    #new tensor has the same H and W of input tensor - channel will be 1 because of reduce sum
    new_tensor = Tensor(curr_tensor.image_height, curr_tensor.image_width, curr_tensor.baseRegion_h, curr_tensor.baseRegion_w, 1)   
    new_tensor.tensor_type = '[]'

    #node on graph to represent the C{r,c} new tensor
    graph.add_node(new_tensor.id,type='S', regions_size=(new_tensor.baseRegion_h,new_tensor.baseRegion_w),label='S_{} scWin:({}.{}): sh({},{},{}={})'.format(new_tensor.id,new_tensor.baseRegion_h,new_tensor.baseRegion_w,new_tensor.height,new_tensor.width,new_tensor.channel,bef_concat_channel))
    
    #add edge of convolution(includes concatenation)
    #add edge of concatenation 
    for _id in similar_tensors_ids:
        graph.add_edge(_id,new_tensor.id,label='Concat&Conv{}'.format(''))
    
    curr_tensor.convoluted = True
    return new_tensor

def poll(_window = (1,1), _tensor = None, _graph=None):
    curr_tensor = _tensor
    graph = _graph
    
    #output tensor
    if curr_tensor.height%_window[0] == 0 and curr_tensor.width%_window[1]==0:
        out_height = curr_tensor.height/_window[0]
        out_width = curr_tensor.width/_window[1]
        out_r = _window[0]*curr_tensor.baseRegion_h
        out_c = _window[1]*curr_tensor.baseRegion_w
        
        #new tensor changes its H and W regarding to the pooling applied - channel is the same
        new_tensor = Tensor(curr_tensor.image_height, curr_tensor.image_width, out_r, out_c, curr_tensor.channel)
        new_tensor.tensor_type = '()'
        
        #adding node
        graph.add_node(new_tensor.id, type='P',window=_window, regions_size=(new_tensor.baseRegion_h,new_tensor.baseRegion_w), label='P_{} scWin:({}.{}): sh({},{},{})'.format(new_tensor.id,new_tensor.baseRegion_h,new_tensor.baseRegion_w,new_tensor.height,new_tensor.width,new_tensor.channel))

        #adding edge
        graph.add_edge(curr_tensor.id,new_tensor.id,label='w{}x{}'.format(_window[0],_window[1]))

        curr_tensor.polled +=1
        return new_tensor
    else:
        print('polling w{}{} not applied to S{}{}'.format(_window[0],_window[1],_tensor.baseRegion_h,_tensor.baseRegion_w))
        return None


def multiple_branches(spn_def):
    """
    Many branches per tensor.

    Example
    -------
    spn_def = {
        "input_shape": input_shape,
        "leaf":
        {
            "type": "gaussian",
            "num_leaf_components": 4
        },
        "sum_layers": {
            "channel_method": "constant",
            "first_sum_channel": 8,

            "hard_inference": False,
            "share_parameters": False,
            "initializer": "uniform"
        },
        "product_layers": {
            "pooling_method": {
                "type": "alternate",
                "amount": 1000
            },
            "pool_windows": [(1, 2), (2, 1)],
            "alt_pool_win": [(2, 4), (4, 2)],
            "sum_pooling": True
        }
    }
    """
    # Arguments
    input_shape = spn_def["input_shape"]
    leaf_components = spn_def["leaf"]["num_leaf_components"]
    first_sum_channel = spn_def["sum_layers"]["first_sum_channel"]
    channel_method = spn_def["sum_layers"]["channel_method"]
    hard_inference = spn_def["sum_layers"]["hard_inference"]
    share_parameters = spn_def["sum_layers"]["share_parameters"]
    initializer = spn_def["sum_layers"]["initializer"]

    pooling_method = spn_def["product_layers"]["pooling_method"]["type"]
    pooling_amt = spn_def["product_layers"]["pooling_method"]["amount"]
    pool_windows = spn_def["product_layers"]["pool_windows"]
    alt_pool_win = spn_def["product_layers"]["alt_pool_win"]
    sum_pooling = spn_def["product_layers"]["sum_pooling"]
    # Create SPN
    spn = SumProductNetwork(input_shape=input_shape)

    # Build leaf layer
    leaf_layer = GaussianLeafLayer(num_leaf_components=leaf_components)
    spn.add_layer(leaf_layer)
    spn.set_leaf_layer(leaf_layer)

    # Multiple branches
    to_process = deque([(leaf_layer,
                        [input_shape[0], input_shape[0], 1])])

    curr_sum_channel = first_sum_channel
    connect_to_root = []

    std_pooling_windows = pool_windows
    num_layers = 0
    is_alternating = False
    while len(to_process) > 0:
        curr_layer, curr_tensor_size = to_process.popleft()

        pool_amt_not_working = 0
        if alt_pool_win is not None:
            if pooling_method == "reduce_at_end":
                if num_layers > pooling_amt:
                    pool_windows = alt_pool_win
            elif pooling_method == "reduce_at_start":
                if num_layers < pooling_amt:
                    pool_windows = alt_pool_win
                else:
                    pool_windows = std_pooling_windows
            elif pooling_method == "alternate":
                if num_layers > 0 and num_layers % pooling_amt == 0:
                    is_alternating = False if is_alternating else True
                pool_windows = alt_pool_win \
                    if is_alternating else std_pooling_windows
        for pool_window in pool_windows:
            if curr_tensor_size[0] >= pool_window[0] and \
                    curr_tensor_size[1] >= pool_window[1]:

                if channel_method == "double":
                    curr_sum_channel = 2 * curr_sum_channel
                elif channel_method == "constant":
                    curr_sum_channel = curr_sum_channel

                sum_layer = SumLayer(out_channels=curr_sum_channel,
                                     hard_inference=hard_inference,
                                     share_parameters=share_parameters,
                                     initializer=initializer)
                spn.add_forward_layer_edge(curr_layer, sum_layer)

                prev_layer = sum_layer

                product_layer = ProductLayer(pooling_size=pool_window,
                                             sum_pooling=sum_pooling)
                spn.add_forward_layer_edge(prev_layer, product_layer)
                prev_layer = product_layer

                prod_layer_shape = product_layer.compute_output_shape(
                    curr_tensor_size)
                new_tensor_size = [
                    prod_layer_shape[0], prod_layer_shape[1], 1]
                to_process.append((product_layer, new_tensor_size))
                num_layers += 1
            else:
                pool_amt_not_working += 1

        if pool_amt_not_working == len(pool_windows):
            if curr_tensor_size[0] != 1 or curr_tensor_size[1] != 1:
                product_layer = ProductLayer(
                    pooling_size=(curr_tensor_size[0], curr_tensor_size[1]))
                prod_layer_shape = product_layer.compute_output_shape(
                    curr_tensor_size)
                spn.add_forward_layer_edge(curr_layer, product_layer)
                curr_layer = product_layer
            connect_to_root.append(curr_layer)
            num_layers += 1

    # Build root layer
    root_layer = SumLayer(out_channels=1)
    spn.set_root_layer(root_layer)
    for to_root in connect_to_root:
        spn.add_forward_layer_edge(to_root, root_layer)

    #[TODO: REMOVE]
    #spn.draw_conv_spn("spn.dot")
    
    return spn

def concat_rec(spn_def,save_dir):
    """
    Inspiring in the regions and its cuts(P&D). Test for concatenation over the channel axis of all children of a Sum_layer, therefore they are followed by a Sum_layer operation.
    
    Example
    -------
    
    """
    
    # Arguments
    input_shape = spn_def["input_shape"]
    leaf_components = spn_def["leaf"]["num_leaf_components"]
    first_sum_channel = spn_def["sum_layers"]["first_sum_channel"]
    channel_method = spn_def["sum_layers"]["channel_method"]
    hard_inference = spn_def["sum_layers"]["hard_inference"]
    share_parameters = spn_def["sum_layers"]["share_parameters"]
    initializer = spn_def["sum_layers"]["initializer"]

    pooling_method = spn_def["product_layers"]["pooling_method"]["type"]
    pooling_amt = spn_def["product_layers"]["pooling_method"]["amount"]
    pool_windows = spn_def["product_layers"]["pool_windows"]
    alt_pool_win = spn_def["product_layers"]["alt_pool_win"]
    sum_pooling = spn_def["product_layers"]["sum_pooling"]

    # Arguments ----------------------
    graph = nx.DiGraph()
    #       -------------------------  DAG
    #The leaf will represent the R Layer
    graph_leaf = Tensor(input_shape[0], input_shape[1], 1, 1, leaf_components)
    graph_leaf.convoluted = True
    graph.add_node(graph_leaf.id,type='R', regions_size=(graph_leaf.baseRegion_h,graph_leaf.baseRegion_w),label='R_{} scWin:({}.{}): sh({},{},{})'.format(graph_leaf.id,graph_leaf.baseRegion_h,graph_leaf.baseRegion_w,graph_leaf.height,graph_leaf.width,graph_leaf.channel))  
    #The unique leaf represents the representational layer - C(1.1)
    #There's one parent of the leaf which will represent a S Layer - same H and W, channel 1
    graph_first_s = Tensor(input_shape[0], input_shape[1], graph_leaf.baseRegion_h, graph_leaf.baseRegion_w, 1)   
    graph_first_s.tensor_type = '[]'
    graph.add_node(graph_first_s.id, type='S',regions_size=(graph_first_s.baseRegion_h,graph_first_s.baseRegion_w),label='S_{} scWin:({}.{}): sh({},{},{})'.format(graph_first_s.id, graph_first_s.baseRegion_h,graph_first_s.baseRegion_w,graph_first_s.height,graph_first_s.width,graph_first_s.channel))
    graph.add_edge(graph_leaf.id,graph_first_s.id,label='Concat&Conv')# this SumLayer node in NX
    graph_leaf.convoluted = True
    
    
    #applying polling on convoluted leaf    
    for polling in pool_windows:
        _graph_first_s = poll(_window = polling, _tensor = graph_first_s,_graph=graph) 
    
    
    for layer in range(0,(input_shape[1]+input_shape[0]-2 +1)): #associated to the DAG depth
        #list of region at this layer to group
        list_size_regions = []
        for row_step in range(1,input_shape[0]+1):
            for col_step in range(1,input_shape[1]+1):
                if (row_step +col_step -2) == layer:
                    list_size_regions.append((row_step,col_step))
                
        for size_region in list_size_regions:


            #get all tensors of the current layer
            dict_tensors = Tensor.get_tensors_dict()
            
            similar_tensors = [] #to tensor
            similar_tensors_ids = []
            for tensor_id in dict_tensors.keys(): 
                current_tensor = dict_tensors[tensor_id] 
                #region size that tensor groups
                current_region_size = (current_tensor.baseRegion_h, current_tensor.baseRegion_w)
                current_layer = current_tensor.at_layer
                current_type = current_tensor.tensor_type
                concat_flag = current_tensor.concatened
                #verifying if this tensor should be concatenated with others
                if size_region == current_region_size and layer == current_layer and current_type == '()' and concat_flag == False:
                    similar_tensors.append(current_tensor)
                    similar_tensors_ids.append(current_tensor.id)
                    current_tensor.concatened = True
            if len(similar_tensors) > 0:
                concat_flag = True
                if concat_flag:
                    #sum of all channels to compute output channel after concat
                    concat_channel = 0
                    for k in range(0,len(similar_tensors)):
                        concat_channel += similar_tensors[k].channel
                    new_concat_tensor = Tensor(input_shape[0], input_shape[1], similar_tensors[0].baseRegion_h, similar_tensors[0].baseRegion_w,concat_channel)
                    #new_concat_tensor = Tensor(input_shape[0], input_shape[1], similar_tensors[0].N, similar_tensors[0].baseRegion_h, similar_tensors[0].baseRegion_w,len(similar_tensors)*similar_tensors[0].channel)
                    #2 conv operation
                    new_conv_tensor = conv2d(similar_tensors_ids, new_concat_tensor, graph)#number of channels(depth) of weight 
                    #doing the polling in the new convoluted tensor to generate new () tensors
                    new_tensor = None
                    for polling in pool_windows:
                        #new_tensor = poll(_window = polling, _tensor = new_conv_tensor,_graph=graph, _spn=spn) 
                        new_tensor = poll(_window = polling, _tensor = new_conv_tensor,_graph=graph) 
                else:
                    #apply convolution to tensors at this layer and then all possible poolling layers
                    for tensor in similar_tensors:
                        new_conv_tensor = conv2d([tensor.id], tensor, graph)
                        new_poolled_tensor = None
                    for polling in pool_windows:
                        #new_tensor = poll(_window = polling, _tensor = new_conv_tensor,_graph=graph, _spn=spn) 
                        new_poolled_tensor = poll(_window = polling, _tensor = new_conv_tensor,_graph=graph) 
            else:
                pass
                #print('No tensors at this layer: ',layer)
    
    #[TODO: REMOVE]
    nx.drawing.nx_pydot.write_dot(graph, "nx.dot")
    
    nx.drawing.nx_pydot.write_dot(graph, "{}/nx.dot".format(save_dir))
    #raise Exception

    #print(Tensor.get_tensors_dict())
    
    print('total of tensors: ', len(Tensor.get_tensors_dict()))
    print('total of graph nodes: ', len(graph.nodes))
    
    dict_n_tensors = {}
    for row in range(1,input_shape[0]+1):
        for col in range(1,input_shape[1]+1):
            dict_n_tensors[(row,col)] = 0
    for node in graph.nodes:
        tensor = Tensor.getTensor(node)
        tuple_regions = (tensor.baseRegion_h,tensor.baseRegion_w)
        #print('node_id:{}  C{}.{}'.format(node,tensor.baseRegion_h,tensor.baseRegion_w))
        dict_n_tensors[tuple_regions] +=1

    for tensor_tuple in dict_n_tensors.keys():
        if dict_n_tensors[tensor_tuple] != 0:
            print('# of Tensors of type C{}.{}: {} '.format(tensor_tuple[0],tensor_tuple[1],dict_n_tensors[tensor_tuple]))
    
    # Create SPN
    spn = SumProductNetwork(input_shape=input_shape)
    dict_layers = {}

    #separating root id
    root_id = [n for n in graph.nodes if graph.out_degree(n) == 0]
    nodes_ids = [n for n in graph.nodes if graph.out_degree(n) != 0]
    #print(root_id)
    #print(nodes_ids, 'len: ', len(nodes_ids))
    for _id in nodes_ids:
        #print(_id,': ', list(graph.successors(_id)), ' type: ',graph.nodes[_id]['type'])

        if graph.nodes[_id]['type'] == 'R':
            # -------------------   Build leaf layer 
            leaf_layer = GaussianLeafLayer(num_leaf_components=leaf_components)
            spn.add_layer(leaf_layer)                    
            spn.set_leaf_layer(leaf_layer)

            dict_layers[_id] = leaf_layer
        elif graph.nodes[_id]['type'] == 'S':
            sum_layer = SumLayer(out_channels=first_sum_channel,
                         hard_inference=hard_inference,
                         share_parameters=share_parameters,
                         initializer=initializer)
            dict_layers[_id] = sum_layer
            spn.add_layer(sum_layer)                 
        elif graph.nodes[_id]['type'] == 'P':
            product_layer = ProductLayer(pooling_size=graph.nodes[_id]['window'],
                                             sum_pooling=sum_pooling)
            dict_layers[_id] = product_layer
            spn.add_layer(product_layer)

    #root
    sum_layer = SumLayer(out_channels=1,
                         hard_inference=hard_inference,
                         share_parameters=share_parameters,
                         initializer=initializer)
    dict_layers[root_id[0]] = sum_layer
    spn.add_layer(sum_layer)     
    spn.set_root_layer(sum_layer)
    
    

    for from_id, to_id in graph.edges:
        spn.add_forward_layer_edge(dict_layers[from_id], dict_layers[to_id])       
    
    #[TODO: REMOVE]
    spn.draw_conv_spn("spn.dot")
    #draw spn graph
    spn.draw_conv_spn('{}/spn.dot'.format(save_dir))
    #print('layers>>>>>\n', spn.layers)
    #print('Number>>>>>\n', len(spn.layers)
    raise Exception

    return spn

def single_branches(spn_definition):
    """
    Build an SPN based on a standard dictionary definition.

    Example
    -------
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
        ],
        "dense_layers": (8, 4)
    }
    """
    # Create SPN
    input_shape = spn_definition["input_shape"]
    dense_layers = spn_definition["dense_layers"]\
        if "dense_layers" in spn_definition else None
    spn = SumProductNetwork(input_shape=input_shape)

    # Build leaf layer
    leaf_layer = None
    if spn_definition["leaf"]["type"] == "gaussian":
        leaf_components = spn_definition["leaf"]["num_leaf_components"]
        leaf_layer = GaussianLeafLayer(num_leaf_components=leaf_components)
    else:
        raise NotImplementedError("Leaf type not implemented.")
    spn.add_layer(leaf_layer)
    spn.set_leaf_layer(leaf_layer)

    # Build branches
    def _build_branch(spn, input_shape,
                      first_sum_channel, channel_method,
                      pooling_windows, pooling_method):
        prev_layer = leaf_layer
        win_pool_size_opts = pooling_windows
        # We don't use the channel, but maintain it in here fixed to 1
        # just so this format is compatible with compute_output_shape
        # of a Product layer.
        curr_tensor_size = [input_shape[0], input_shape[0], 1]
        curr_sum_channel = first_sum_channel
        pool_choice_counter = 0
        pool_amt_not_working = 0
        while True:
            # Alternate among all given pooling windows
            win_pool_size = None
            if pooling_method == "alternate":
                win_pool_size = win_pool_size_opts[
                    pool_choice_counter % len(win_pool_size_opts)]
                pool_choice_counter += 1
            elif pooling_method == "turn":
                if pool_amt_not_working > 0:
                    pool_choice_counter += 1
                win_pool_size = win_pool_size_opts[
                    pool_choice_counter % len(win_pool_size_opts)]
            else:
                raise ValueError("Pooling layer method not valid.")

            pool_size = None
            # Check if can still apply current pooling layer
            if curr_tensor_size[0] >= win_pool_size[0] and\
                    curr_tensor_size[1] >= win_pool_size[1]:
                    pool_size = win_pool_size
            # If not, count as a fail
            else:
                pool_amt_not_working += 1
                # When all pooling windows fail, check if it is the end
                # (1, 1) output, or still one last pooling to do.
                if pool_amt_not_working < len(win_pool_size_opts):
                    continue
                else:
                    if curr_tensor_size[0] == 1 and\
                            curr_tensor_size[1] == 1:
                        break
                    else:
                        pool_size = (
                            curr_tensor_size[0], curr_tensor_size[1])

            sum_layer = SumLayer(out_channels=curr_sum_channel)
            spn.add_forward_layer_edge(prev_layer, sum_layer)
            # Increment channel using method
            if channel_method == "double":
                curr_sum_channel = 2 * curr_sum_channel
            elif channel_method == "constant":
                curr_sum_channel = curr_sum_channel
            else:
                raise ValueError("Incremental channel method not valid.")
            prev_layer = sum_layer

            product_layer = ProductLayer(pooling_size=pool_size)
            spn.add_forward_layer_edge(prev_layer, product_layer)
            prev_layer = product_layer

            win_pool_size = win_pool_size_opts[
                pool_choice_counter % len(win_pool_size_opts)]

            prod_layer_shape = product_layer.compute_output_shape(
                curr_tensor_size)
            curr_tensor_size = [
                prod_layer_shape[0], prod_layer_shape[1], 1]

            pool_amt_not_working = 0
        return prev_layer

    all_branches_output = []
    for branch in spn_definition["branches"]:
        first_sum_channel = branch["sum_layers"]["first_sum_channel"]
        channel_method = branch["sum_layers"]["channel_method"]
        pooling_windows = branch["product_layers"]["pooling_windows"]
        pooling_method = branch["product_layers"]["pooling_method"]
        branch_out = _build_branch(spn, input_shape, first_sum_channel,
                                   channel_method,
                                   pooling_windows, pooling_method)
        all_branches_output.append(branch_out)

    # Dense layers
    if dense_layers is not None:
        dense_amt, layers_amt = dense_layers
        dense_layer = SumLayer(out_channels=dense_amt)
        for branch_out in all_branches_output:
            spn.add_forward_layer_edge(branch_out, dense_layer)
        prev_dense_layer = dense_layer
        for i in range(layers_amt):
            dense_layer = SumLayer(out_channels=dense_amt)
            spn.add_forward_layer_edge(prev_dense_layer, dense_layer)
            prev_dense_layer = dense_layer
        # Build root layer
        root_layer = SumLayer(out_channels=1)
        spn.set_root_layer(root_layer)
        spn.add_forward_layer_edge(prev_dense_layer, root_layer)
    else:
        # Build root layer
        root_layer = SumLayer(out_channels=1)
        spn.set_root_layer(root_layer)
        for branch_out in all_branches_output:
            spn.add_forward_layer_edge(branch_out, root_layer)

    return spn
