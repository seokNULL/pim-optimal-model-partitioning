import os
import argparse
import json
import onnx
import psutil
import numpy
import time
from threading import Thread
from dataclasses import dataclass
from statistics import median

"""
This profiler tool could run a transformer model and print out the kernel time spent on each Node of the model.
Example of profiling of longformer model:
    python profiler.py --model longformer-base-4096_fp32.onnx --batch_size 1 --sequence_length 4096 --global_length 8 --samples 1000 --thread_num 8 --dummy_inputs longformer --use_pim
"""


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, help="onnx model path")

    parser.add_argument('-b', '--batch_size', required=False, type=int, default=1, help="batch size of input")

    parser.add_argument('-s',
                        '--sequence_length',
                        required=False,
                        type=int,
                        default=32,
                        help="sequence length of input")

    parser.add_argument('--past_sequence_length',
                        required=False,
                        type=int,
                        default=1,
                        help="past sequence length for gpt2")

    parser.add_argument('--global_length',
                        required=False,
                        type=int,
                        default=1,
                        help="number of global tokens for longformer")

    parser.add_argument(
        '--samples',
        required=False,
        type=int,
        default=1000,
        help="number of samples to test. Set it large enough to reduce the variance of performance result.")

    parser.add_argument(
        '--threshold',
        required=False,
        type=float,
        default=0,
        help=
        "Threshold of ratio of run time of a node among all nodes. Nodes that nodes with lower ratio will not be in detail results."
    )

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="number of threads to use")

    parser.add_argument('--input_ids_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for input ids, for bert")
    parser.add_argument('--segment_ids_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for segment ids, for bert")
    parser.add_argument('--input_mask_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for attention mask, for bert")

    parser.add_argument('--dummy_inputs',
                        required=False,
                        default='default',
                        choices=['bert', 'gpt2', 'longformer', 'default'],
                        help="Way to create dummy inputs. If your model is not aa")

    parser.add_argument('-g', '--use_pim', required=False, action='store_true', help="use PIM")
    parser.set_defaults(use_pim=False)

    parser.add_argument(
        '--basic_optimization',
        required=False,
        action='store_true',
        help="Enable only basic graph optimizations. By default, all optimizations are enabled in OnnxRuntime")
    parser.set_defaults(basic_optimization=False)

    parser.add_argument('--kernel_time_only',
                        required=False,
                        action='store_true',
                        help="Only include the kernel time and no fence time")
    parser.set_defaults(kernel_time_only=False)

    parser.add_argument('-v', '--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args(argv)
    return args


def create_bert_inputs(model, batch_size, sequence_length, samples, input_ids_name, segment_ids_name, input_mask_name):
    from bert_test_data import get_bert_inputs, generate_test_data
    input_ids, segment_ids, input_mask = get_bert_inputs(model, input_ids_name, segment_ids_name, input_mask_name)
    all_inputs = generate_test_data(batch_size,
                                    sequence_length,
                                    test_cases=samples,
                                    seed=123,
                                    verbose=False,
                                    input_ids=input_ids,
                                    segment_ids=segment_ids,
                                    input_mask=input_mask,
                                    random_mask_length=False)

    return all_inputs


def run_profile(device, onnx_model_path, use_pim, basic_optimization, thread_num, batch_size, sequence_length, all_inputs, align_map, partition_map=None):
    from pim_benchmark_helper import create_onnxruntime_session

    session = create_onnxruntime_session(device,
                                         onnx_model_path,
                                         use_pim,
                                         partition_map=partition_map,
                                         align_map=align_map,
                                         enable_all_optimization=not basic_optimization,
                                         num_threads=thread_num,
                                         enable_profiling=True)

    # for inputs in all_inputs:
    #     _ = session.run(None, inputs)

    for inputs in all_inputs:
        result = session.run(None, inputs)
        print("check result")
        print(result[0])
        # print(result[5])

    profile_file = session.end_profiling()
    return profile_file

def run_edge_profile(onnx_model_path, use_pim, basic_optimization, thread_num, batch_size, sequence_length, all_inputs, partition_map):
    from pim_benchmark_helper import create_onnxruntime_session
    
    align_map = {}

    session = create_onnxruntime_session(onnx_model_path,
                                         use_pim,
                                         partition_map=partition_map,
                                         align_map=align_map,
                                         enable_all_optimization=not basic_optimization,
                                         num_threads=thread_num,
                                         enable_profiling=True)

    for inputs in all_inputs:
        result = session.run(None, inputs)
        print(result[0])
        # print(result[5])

    profile_file = session.end_profiling()
    return profile_file


def load_profile_json(profile_file):
    print(f"loading profile output {profile_file} ...")

    with open(profile_file, "r") as f:
        sess_time = json.load(f)

    assert isinstance(sess_time, list)
    return sess_time

def generate_dcg(cpu_profile_file, pim_profile_file):

    @dataclass
    class Datafield:
        sink_ptr: tuple
        size: int
        sink_next_ptr: []
        src_ptr: tuple
        src_next_ptr: tuple
        color : str
        explored_edge_list : []
        explored_path : []

    @dataclass
    class Cost:
        node_cost: int
        edge_cost: int

    NUM_OF_NODES = 0 
    NUM_OF_DEVICES = 2
    prof_data = {}

    graph_idx_dict = {}

    # BUILD DCG
    pim_op_list = []
    cpu_prof = os.path.join(cpu_profile_file)
    with open(cpu_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')

                op_name = item['args']['op_name']
                ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()

                prof_data[node_name] = {}
                prof_data[node_name]['op_kind'] = op_name
                prof_data[node_name]['ep_type'] = ep_type
                prof_data[node_name]['graph_index'] = int(item['args']['graph_index'])
                # Output size / sizeof(float)
                prof_data[node_name]['elem_size'] = int(int(item['args']['output_size']) / 4)
                # if op_name == "ReduceMean":
                #     prof_data[node_name]['elem_size'] = int(int(item['args']['activation_size']) / 4)
                incoming_nodes = item['args']['input_nodes'].split()
                outgoing_nodes = item['args']['output_nodes'].split() 

                prof_data[node_name]['src_nodes'] = incoming_nodes
                prof_data[node_name]['sink_nodes'] = outgoing_nodes
                graph_idx_dict[node_name] = int(item['args']['graph_index'])

                NUM_OF_NODES += 1

    # print(NUM_OF_NODES)
    # print("PROFILE DATA")
    # for key, value in prof_data.items():
    #     print(key)
    #     print(value)
    #     print("\n")

    # Collect pim only
    pim_prof = os.path.join(pim_profile_file)
    with open(pim_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')
                ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
                if node_name in prof_data.keys() and ep_type == 'pim':
                    pim_op_list.append(node_name)

    # print("PIM OP LIST")
    # print(pim_op_list)

    empty_datafield = Datafield(sink_ptr=None, size=None, sink_next_ptr=None, src_ptr=None, src_next_ptr=None, color=None, explored_edge_list=None, explored_path=None)
    node_list = list(prof_data.keys())

    start_gen_dcg = time.time()

    DCG = {}
    dev_list = ["cpu", "pim"]
    for node_name in prof_data.keys():
        DCG[node_name] = [None] * NUM_OF_DEVICES
        for k in range(NUM_OF_DEVICES):
            if k == 0:
                DCG[node_name][k] = Datafield(None, None, None, None, None, 'white', [], None)
            else:
                DCG[node_name][k] = Datafield(None, None, None, None, None, None, [], None)

    edge_dict = {}
    source_list = []
    for cur_node, attr in prof_data.items():
        # print("Current node: ", cur_node)
        sink_nodes = attr['sink_nodes'][:]
        src_nodes = attr['src_nodes'][:]
        for dev in dev_list:
            dev_idx = dev_list.index(dev)
            # 1. CPU device
            if dev_idx == 0:
                pim_sink_nodes = [x for x in sink_nodes if x in pim_op_list]
                pim_src_nodes = [x for x in src_nodes if x in pim_op_list]
                if pim_sink_nodes:
                    DCG[cur_node][dev_idx].sink_ptr = (pim_sink_nodes[0], 1)
                    DCG[cur_node][dev_idx].size = attr['elem_size']
                    DCG[cur_node][dev_idx].color = 'white'
                    # DCG[cur_node][dev_idx].edge_list = []
                    # DCG[cur_node][dev_idx].ua_edge = str(attr['elem_size']) + str(dev_idx) + str(1)
                    # Add to edge_dict
                    edge = ((cur_node, dev_idx), (pim_sink_nodes[0], 1))
                    edge_attr = (dev_idx, 1, attr['elem_size'])
                    # print("\t", (pim_sink_nodes[0], 1))
                    # print("\t", edge_attr)
                    edge_dict[edge] = [edge_attr, 'white']
                if pim_src_nodes:
                    if (pim_src_nodes[0], 1) not in source_list:
                        DCG[cur_node][dev_idx].src_ptr = (pim_src_nodes[0], 1)
                        # Add to edge_dict
                        edge = ((pim_src_nodes[0], 1), (cur_node, dev_idx))
                        edge_attr = (1, dev_idx, attr['elem_size'])
                        # print("\t", (pim_src_nodes[0], 1))
                        # print("\t", edge_attr)
                        # edge_dict[edge] = [edge_attr, 'white']
                        source_list.append((pim_src_nodes[0], 1))
                    if len(pim_src_nodes) > 1:
                        DCG[cur_node][dev_idx].src_next_ptr = []
                        for i in range(1, len(pim_src_nodes)):
                            src_data_field = Datafield(None, None, None, None, None, 'white', None, None)
                            if (pim_src_nodes[i], 1) not in source_list:
                                source_list.append((pim_src_nodes[i], 1))
                                src_data_field.src_ptr = (pim_src_nodes[i], 1)
                                # Add to edge_dict
                                edge = ((pim_src_nodes[i], 1), (cur_node, dev_idx))
                                edge_attr = (1, dev_idx, attr['elem_size'])
                                # print("\t", (pim_src_nodes[i], 1))
                                # print("\t", edge_attr)
                                # edge_dict[edge] = [edge_attr, 'white']
                                
                                # DCG[cur_node][dev_idx].src_next_ptr.append(src_data_field)                            
                                DCG[cur_node][dev_idx].src_next_ptr = (pim_src_nodes[i], 1)                         
            # 2. PIM device
            else:
                if cur_node not in pim_op_list:
                    continue
                else:
                    DCG[cur_node][dev_idx].color = 'white'
                    # DCG[cur_node][dev_idx].edge_list = []  
                    if sink_nodes:
                        DCG[cur_node][dev_idx].sink_ptr = (sink_nodes[0], 0)
                        DCG[cur_node][dev_idx].size = attr['elem_size']
                        # DCG[cur_node][dev_idx].ua_edge = str(attr['elem_size']) + str(dev_idx) + str(0)
                        # Add to edge_dict
                        edge = ((cur_node, dev_idx), (sink_nodes[0], 0))
                        edge_attr = (dev_idx, 0, attr['elem_size'])
                        # print("\t", (sink_nodes[0], 0))
                        # print("\t", edge_attr)
                        edge_dict[edge] = [edge_attr, 'white']                              
                    if src_nodes:
                        if (src_nodes[0], 0) not in source_list:
                            source_list.append((src_nodes[0], 0))
                             # Add to edge_dict
                            edge = ((src_nodes[0], 0), (cur_node, dev_idx))
                            edge_attr = (0, dev_idx, attr['elem_size'])
                            # print("\t", (src_nodes[0], 0))
                            # print("\t", edge_attr)
                            # edge_dict[edge] = [edge_attr, 'white']
                            DCG[cur_node][dev_idx].src_ptr = (src_nodes[0], 0)
                        if len(src_nodes) > 1:
                            DCG[cur_node][dev_idx].src_next_ptr = []
                            for i in range(1, len(src_nodes)):
                                src_data_field = Datafield(None, None, None, None, None, 'white', None, None)
                                if (src_nodes[i], 0) not in source_list:
                                    source_list.append((src_nodes[i], 0))
                                    src_data_field.src_ptr = (src_nodes[i], 0)
                                    # Add to edge_dict
                                    edge = ((src_nodes[i], 0), (cur_node, dev_idx))
                                    edge_attr = (0, dev_idx, attr['elem_size'])
                                    # print("\t", (src_nodes[i], 0))
                                    # print("\t", edge_attr)
                                    # edge_dict[edge] = [edge_attr, 'white']
                                    
                                    # DCG[cur_node][dev_idx].src_next_ptr.append(src_data_field)  
                                    DCG[cur_node][dev_idx].src_next_ptr =  (src_nodes[i], 0)

    return DCG, edge_dict, graph_idx_dict, pim_op_list

def get_optimal_partition(cpu_profile_file, pim_profile_file, edge_profile_file):
    import os
    import json
    from statistics import median
    from dataclasses import dataclass

    @dataclass
    class Datafield:
        sink_ptr: tuple
        size: int
        sink_next_ptr: []
        src_ptr: tuple
        src_next_ptr: []
        color : str
        edge_list : []

    @dataclass
    class Cost:
        node_cost: int
        edge_cost: int

    NUM_OF_NODES = 0 
    NUM_OF_DEVICES = 2
    prof_data = {}

    # BUILD DCG
    pim_op_list = []
    cpu_prof = os.path.join(cpu_profile_file)
    with open(cpu_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')

                op_name = item['args']['op_name']
                ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()

                prof_data[node_name] = {}
                prof_data[node_name]['op_kind'] = op_name
                prof_data[node_name]['ep_type'] = ep_type
                prof_data[node_name]['graph_index'] = int(item['args']['graph_index'])
                # Output size / sizeof(float)
                prof_data[node_name]['elem_size'] = int(int(item['args']['output_size']) / 4)
                incoming_nodes = item['args']['input_nodes'].split()
                outgoing_nodes = item['args']['output_nodes'].split() 

                prof_data[node_name]['src_nodes'] = incoming_nodes
                prof_data[node_name]['sink_nodes'] = outgoing_nodes

                NUM_OF_NODES += 1

    # print("PROFILE DATA")
    # for key, value in prof_data.items():
    #     print(key)
    #     print(value)
    #     print("\n")

    # Collect pim only
    pim_prof = os.path.join(pim_profile_file)
    with open(pim_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')
                ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
                if node_name in prof_data.keys() and ep_type == 'pim':
                    pim_op_list.append(node_name)

    # print("PIM OP LIST")
    # print(pim_op_list)

    empty_datafield = Datafield(sink_ptr=None, size=None, sink_next_ptr=None, src_ptr=None, src_next_ptr=None, color=None, edge_list=None)
    node_list = list(prof_data.keys())

    DCG = {}
    dev_list = ["cpu", "pim"]
    for node_name in prof_data.keys():
        DCG[node_name] = [None] * NUM_OF_DEVICES
        for k in range(NUM_OF_DEVICES):
            if k == 0:
                DCG[node_name][k] = Datafield(None, None, None, None, None, 'white', [])
            else:
                DCG[node_name][k] = Datafield(None, None, None, None, None, None, None)

    edge_dict = {}
    source_list = []
    for cur_node, attr in prof_data.items():
        sink_nodes = attr['sink_nodes'][:]
        src_nodes = attr['src_nodes'][:]
        for dev in dev_list:
            dev_idx = dev_list.index(dev)
            # 1. CPU device
            if dev_idx == 0:
                pim_sink_nodes = [x for x in sink_nodes if x in pim_op_list]
                pim_src_nodes = [x for x in src_nodes if x in pim_op_list]
                if pim_sink_nodes:
                    DCG[cur_node][dev_idx].sink_ptr = (pim_sink_nodes[0], 1)
                    DCG[cur_node][dev_idx].size = attr['elem_size']
                    DCG[cur_node][dev_idx].color = 'white'
                    DCG[cur_node][dev_idx].edge_list = []
                    # Add to edge_dict
                    edge = ((cur_node, dev_idx), (pim_sink_nodes[0], 1))
                    edge_attr = (dev_idx, 1, attr['elem_size'])
                    edge_dict[edge] = [edge_attr, 'white']
                if pim_src_nodes:
                    if (pim_src_nodes[0], 1) not in source_list:
                        DCG[cur_node][dev_idx].src_ptr = (pim_src_nodes[0], 1)
                        # Add to edge_dict
                        edge = ((pim_src_nodes[0], 1), (cur_node, dev_idx))
                        edge_attr = (1, dev_idx, attr['elem_size'])
                        edge_dict[edge] = [edge_attr, 'white']
                        source_list.append((pim_src_nodes[0], 1))
                    if len(pim_src_nodes) > 1:
                        DCG[cur_node][dev_idx].src_next_ptr = []
                        for i in range(1, len(pim_src_nodes)):
                            src_data_field = Datafield(None, None, None, None, None, 'white', [])
                            if (pim_src_nodes[i], 1) not in source_list:
                                source_list.append((pim_src_nodes[i], 1))
                                src_data_field.src_ptr = (pim_src_nodes[i], 1)
                                # Add to edge_dict
                                edge = ((pim_src_nodes[i], 1), (cur_node, dev_idx))
                                edge_attr = (1, dev_idx, attr['elem_size'])
                                edge_dict[edge] = [edge_attr, 'white']
                                DCG[cur_node][dev_idx].src_next_ptr.append(src_data_field)                            
            # 2. PIM device
            else:
                if cur_node not in pim_op_list:
                    continue
                else:
                    DCG[cur_node][dev_idx].color = 'white'
                    DCG[cur_node][dev_idx].edge_list = []                
                    if sink_nodes:
                        DCG[cur_node][dev_idx].sink_ptr = (sink_nodes[0], 0)
                        DCG[cur_node][dev_idx].size = attr['elem_size']
                        # DCG[cur_node][dev_idx].color = 'white'
                        # DCG[cur_node][dev_idx].edge_list = []
                        # Add to edge_dict
                        edge = ((cur_node, dev_idx), (sink_nodes[0], 0))
                        edge_attr = (dev_idx, 0, attr['elem_size'])
                        edge_dict[edge] = [edge_attr, 'white']                              
                    if src_nodes:
                        if (src_nodes[0], 0) not in source_list:
                            source_list.append((src_nodes[0], 0))
                             # Add to edge_dict
                            edge = ((src_nodes[0], 0), (cur_node, dev_idx))
                            edge_attr = (0, dev_idx, attr['elem_size'])
                            edge_dict[edge] = [edge_attr, 'white']
                            DCG[cur_node][dev_idx].src_ptr = (src_nodes[0], 0)
                        if len(src_nodes) > 1:
                            DCG[cur_node][dev_idx].src_next_ptr = []
                            for i in range(1, len(src_nodes)):
                                src_data_field = Datafield(None, None, None, None, None, 'white', [])
                                if (src_nodes[i], 0) not in source_list:
                                    source_list.append((src_nodes[i], 0))
                                    src_data_field.src_ptr = (src_nodes[i], 0)
                                    # Add to edge_dict
                                    edge = ((src_nodes[i], 0), (cur_node, dev_idx))
                                    edge_attr = (0, dev_idx, attr['elem_size'])
                                    edge_dict[edge] = [edge_attr, 'white']
                                    DCG[cur_node][dev_idx].src_next_ptr.append(src_data_field)  

    # for key, value in DCG.items():
    #     print(key)
    #     for val in value:
    #         print(val)
    #     print("\n")

    # ALS

    ua_edge_list = {}
    ua_edge_prof = os.path.join(edge_profile_file)

    with open(ua_edge_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')
                op_name = item['args']['op_name']
                if "Memcpy" in op_name:
                    memcpy_time = float(item['dur'])
                    src_node = item['args']['input_nodes'].split()[0]
                    dst_node = item['args']['output_nodes'].split()[0]
                    dma_size = 0
                    if op_name == 'MemcpyFromHost':
                        dma_size = int(int(item['args']['activation_size'])/4)
                        ua_edge = str(dma_size) + '0' + '1'
                    else:
                        dma_size = int(int(item['args']['activation_size'])/2)
                        ua_edge = str(dma_size) + '1' + '0'
                    if ua_edge not in ua_edge_list.keys():
                        ua_edge_list[ua_edge] = [memcpy_time]
                    else:
                        ua_edge_list[ua_edge].append(memcpy_time)

    for key, value in ua_edge_list.items():
        ua_edge_list[key] = median(value)

    # print("ua_edge_list")
    # print(ua_edge_list)

    cpu_prof = cpu_profile_file
    pim_prof = pim_profile_file
    INF = 1000000000000000000000000

    schedule_graph = {}
    for node_name in prof_data.keys():
        schedule_graph[node_name] = [None] * NUM_OF_DEVICES
        for k in range(NUM_OF_DEVICES):
            schedule_graph[node_name][k] = Cost(None, None)

    json_file = [cpu_prof, pim_prof]
    for file in json_file:
        ep = file.split('_')[0]
        with open(file, 'r') as f:
            ep_json = json.load(f)
            for item in ep_json:
                if (item['name'].find('_kernel_time') != -1):
                    node_name = item['name'].replace('_kernel_time', '')
                    # print("\nnode_name: ", node_name)
                    ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
                    # print("dur: ", item['dur'])
                    if "Memcpy" not in node_name:
                        if ep == ep_type:
                            dev_idx = dev_list.index(ep)
                            schedule_graph[node_name][dev_idx].node_cost = float(item['dur'])
                            schedule_graph[node_name][dev_idx].edge_cost = 0
                        else:
                            dev_idx = dev_list.index(ep)
                            schedule_graph[node_name][dev_idx].node_cost = INF
                            schedule_graph[node_name][dev_idx].edge_cost = 0


    for node_name in prof_data.keys():
        # print("node_name: ", node_name)
        for k in range(NUM_OF_DEVICES):
            if DCG[node_name][k].src_ptr:
                src_dnn, src_dev = DCG[node_name][k].src_ptr
                dma_size = DCG[src_dnn][src_dev].size
                # print(dma_size)
                distinct_edge = str(dma_size) + str(DCG[node_name][k].src_ptr[1]) + str(k)
                schedule_graph[node_name][k].edge_cost += ua_edge_list[distinct_edge]
            if DCG[node_name][k].src_next_ptr:
                for item in DCG[node_name][k].src_next_ptr:
                    src_next_dnn, src_next_dev = item.src_ptr
                    dma_next_size = DCG[src_next_dnn][src_next_dev].size
                    distinct_next_edge = str(dma_next_size) + str(item.src_ptr[1]) + str(k)
                    schedule_graph[node_name][k].edge_cost += ua_edge_list[distinct_next_edge]

    # print(schedule_graph)
    start_op = time.time()

    f_cpu = [0 for i in range(NUM_OF_NODES)]
    f_pim = [0 for i in range(NUM_OF_NODES)]

    l_cpu = ['None' for i in range(NUM_OF_NODES)]
    l_pim = ['None' for i in range(NUM_OF_NODES)]

    f_cpu[0] = schedule_graph[node_list[0]][0].node_cost
    f_pim[0] = schedule_graph[node_list[0]][1].node_cost

    # print("First node: ", node_list[0])
    # print(f_cpu[0], f_pim[0])

    for j in range(1, NUM_OF_NODES):
        cur_node = node_list[j]
        # print(f_cpu[j-1], schedule_graph[node_list[j]][0].node_cost, "vs", f_pim[j-1], schedule_graph[node_list[j]][0].edge_cost, schedule_graph[node_list[j]][0].node_cost)
        if f_cpu[j-1] + schedule_graph[node_list[j]][0].node_cost <= f_pim[j-1] + schedule_graph[node_list[j]][0].edge_cost + schedule_graph[node_list[j]][0].node_cost:
            f_cpu[j] = f_cpu[j-1] + schedule_graph[node_list[j]][0].node_cost
            l_cpu[j] = 'cpu'
        else:
            f_cpu[j] = f_pim[j-1] + schedule_graph[node_list[j]][0].edge_cost + schedule_graph[node_list[j]][0].node_cost
            l_cpu[j] = 'pim'
        # print(f_pim[j-1], schedule_graph[node_list[j]][1].node_cost, f_cpu[j-1], schedule_graph[node_list[j]][1].edge_cost, schedule_graph[node_list[j]][1].node_cost)
        if f_pim[j-1] + schedule_graph[node_list[j]][1].node_cost <= f_cpu[j-1] + schedule_graph[node_list[j]][1].edge_cost + schedule_graph[node_list[j]][1].node_cost:
            f_pim[j] = f_pim[j-1] + schedule_graph[node_list[j]][1].node_cost
            l_pim[j] = 'pim'
        else:
            f_pim[j] = f_cpu[j-1] + schedule_graph[node_list[j]][1].edge_cost + schedule_graph[node_list[j]][1].node_cost
            l_pim[j] = 'cpu'

    if (f_cpu[NUM_OF_NODES-1] + 0 <= f_pim[NUM_OF_NODES-1] + 0):
        f_opt = f_cpu[NUM_OF_NODES-1] + 0
        l_opt = 'cpu'
    else:
        f_opt = f_pim[NUM_OF_NODES-1] + 0
        l_opt = 'pim'

    print("f_opt: ", f_opt)

    end_op = time.time()
    print("ALS time: ", end_op - start_op)

    dnn_partition = {'CPUExecutionProvider' : [], 'PIMExecutionProvider': []}
    set_ep = lambda x : 'CPUExecutionProvider' if x == 'cpu' else ('PIMExecutionProvider' if x == 'pim' else 'WRG')
    l_dev = l_opt
    dnn_partition[set_ep(l_dev)].append(prof_data[node_list[NUM_OF_NODES-1]]['graph_index'])
    # print("line: ", l_dev, "graph_idx: ", prof_data[node_list[NUM_OF_NODES-1]]['graph_index'])
    l_print = lambda dev, idx : l_cpu[idx] if dev == 'cpu' else l_pim[idx]

    check_list = []
    check_value = {}
    check_value[node_list[NUM_OF_NODES-1]] = (schedule_graph[node_list[NUM_OF_NODES-1]][dev_list.index(l_dev)].node_cost, schedule_graph[node_list[NUM_OF_NODES-1]][dev_list.index(l_dev)].edge_cost)
    for j in reversed(range(1, NUM_OF_NODES)):
        l_dev = l_print(l_dev, j)
        check_value[node_list[j-1]] = (schedule_graph[node_list[j-1]][dev_list.index(l_dev)].node_cost, schedule_graph[node_list[j-1]][dev_list.index(l_dev)].edge_cost)
        if set_ep(l_dev) == "PIMExecutionProvider":
            check_list.append(node_list[j-1])
        dnn_partition[set_ep(l_dev)].append(prof_data[node_list[j-1]]['graph_index'])

    for key, value in dnn_partition.items():
        value.sort(reverse=True)

    return dnn_partition

# def get_optimal_partition(DCG, graph_idx_dict, cpu_profile_file, pim_profile_file, edge_profile_file):

#     @dataclass
#     class Cost:
#         node_cost: int
#         edge_cost: int

#     dev_list = ["cpu", "pim"]
#     NUM_OF_DEVICES = 2
#     dcg_node_list = list(DCG.keys())
#     node_list = dcg_node_list
#     ua_edge_list = {}
#     ua_edge_prof = os.path.join(edge_profile_file)

#     with open(ua_edge_prof, 'r') as f:
#         data = json.load(f)
#         for item in data:
#             if (item['name'].find('_kernel_time') != -1):
#                 node_name = item['name'].replace('_kernel_time', '')
#                 op_name = item['args']['op_name']
#                 if "Memcpy" in op_name:
#                     memcpy_time = float(item['dur'])
#                     src_node = item['args']['input_nodes'].split()[0]
#                     dst_node = item['args']['output_nodes'].split()[0]
#                     dma_size = 0
#                     if op_name == 'MemcpyFromHost':
#                         dma_size = int(int(item['args']['activation_size'])/4)
#                         ua_edge = str(dma_size) + '0' + '1'
#                     else:
#                         dma_size = int(int(item['args']['activation_size'])/2)
#                         ua_edge = str(dma_size) + '1' + '0'
#                     if ua_edge not in ua_edge_list.keys():
#                         ua_edge_list[ua_edge] = [memcpy_time]
#                     else:
#                         ua_edge_list[ua_edge].append(memcpy_time)

#     for key, value in ua_edge_list.items():
#         ua_edge_list[key] = median(value)

#     print("ua_edge_list")
#     print(ua_edge_list)

#     cpu_prof = cpu_profile_file
#     pim_prof = pim_profile_file
#     INF = 1000000000000000000000000

#     schedule_graph = {}
#     for node_name in dcg_node_list:
#         schedule_graph[node_name] = [None] * NUM_OF_DEVICES
#         for k in range(NUM_OF_DEVICES):
#             schedule_graph[node_name][k] = Cost(None, None)

#     json_file = [cpu_prof, pim_prof]
#     for file in json_file:
#         ep = file.split('_')[0]
#         with open(file, 'r') as f:
#             ep_json = json.load(f)
#             for item in ep_json:
#                 if (item['name'].find('_kernel_time') != -1):
#                     node_name = item['name'].replace('_kernel_time', '')
#                     # print("\nnode_name: ", node_name)
#                     ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
#                     # print("dur: ", item['dur'])
#                     if "Memcpy" not in node_name:
#                         if ep == ep_type:
#                             dev_idx = dev_list.index(ep)
#                             schedule_graph[node_name][dev_idx].node_cost = float(item['dur'])
#                             schedule_graph[node_name][dev_idx].edge_cost = 0
#                         else:
#                             dev_idx = dev_list.index(ep)
#                             schedule_graph[node_name][dev_idx].node_cost = INF
#                             schedule_graph[node_name][dev_idx].edge_cost = 0


#     for node_name in dcg_node_list:
#         # print("node_name: ", node_name)
#         for k in range(NUM_OF_DEVICES):
#             if DCG[node_name][k].src_ptr:
#                 src_dnn, src_dev = DCG[node_name][k].src_ptr
#                 dma_size = DCG[src_dnn][src_dev].size
#                 print(dma_size)
#                 distinct_edge = str(dma_size) + str(DCG[node_name][k].src_ptr[1]) + str(k)
#                 schedule_graph[node_name][k].edge_cost += ua_edge_list[distinct_edge]
#             if DCG[node_name][k].src_next_ptr:
#                 for item in DCG[node_name][k].src_next_ptr:
#                     src_next_dnn, src_next_dev = item.src_ptr
#                     dma_next_size = DCG[src_next_dnn][src_next_dev].size
#                     distinct_next_edge = str(dma_next_size) + str(item.src_ptr[1]) + str(k)
#                     schedule_graph[node_name][k].edge_cost += ua_edge_list[distinct_next_edge]

#     # print(schedule_graph)

#     f_cpu = [0 for i in range(NUM_OF_NODES)]
#     f_pim = [0 for i in range(NUM_OF_NODES)]

#     l_cpu = ['None' for i in range(NUM_OF_NODES)]
#     l_pim = ['None' for i in range(NUM_OF_NODES)]

#     f_cpu[0] = schedule_graph[node_list[0]][0].node_cost
#     f_pim[0] = schedule_graph[node_list[0]][1].node_cost

#     # print("First node: ", node_list[0])
#     # print(f_cpu[0], f_pim[0])

#     for j in range(1, NUM_OF_NODES):
#         cur_node = node_list[j]
#         # print(f_cpu[j-1], schedule_graph[node_list[j]][0].node_cost, "vs", f_pim[j-1], schedule_graph[node_list[j]][0].edge_cost, schedule_graph[node_list[j]][0].node_cost)
#         if f_cpu[j-1] + schedule_graph[node_list[j]][0].node_cost <= f_pim[j-1] + schedule_graph[node_list[j]][0].edge_cost + schedule_graph[node_list[j]][0].node_cost:
#             f_cpu[j] = f_cpu[j-1] + schedule_graph[node_list[j]][0].node_cost
#             l_cpu[j] = 'cpu'
#         else:
#             f_cpu[j] = f_pim[j-1] + schedule_graph[node_list[j]][0].edge_cost + schedule_graph[node_list[j]][0].node_cost
#             l_cpu[j] = 'pim'
#         # print(f_pim[j-1], schedule_graph[node_list[j]][1].node_cost, f_cpu[j-1], schedule_graph[node_list[j]][1].edge_cost, schedule_graph[node_list[j]][1].node_cost)
#         if f_pim[j-1] + schedule_graph[node_list[j]][1].node_cost <= f_cpu[j-1] + schedule_graph[node_list[j]][1].edge_cost + schedule_graph[node_list[j]][1].node_cost:
#             f_pim[j] = f_pim[j-1] + schedule_graph[node_list[j]][1].node_cost
#             l_pim[j] = 'pim'
#         else:
#             f_pim[j] = f_cpu[j-1] + schedule_graph[node_list[j]][1].edge_cost + schedule_graph[node_list[j]][1].node_cost
#             l_pim[j] = 'cpu'

#     if (f_cpu[NUM_OF_NODES-1] + 0 <= f_pim[NUM_OF_NODES-1] + 0):
#         f_opt = f_cpu[NUM_OF_NODES-1] + 0
#         l_opt = 'cpu'
#     else:
#         f_opt = f_pim[NUM_OF_NODES-1] + 0
#         l_opt = 'pim'

#     print("f_opt: ", f_opt)

#     dnn_partition = {'CPUExecutionProvider' : [], 'PIMExecutionProvider': []}
#     set_ep = lambda x : 'CPUExecutionProvider' if x == 'cpu' else ('PIMExecutionProvider' if x == 'pim' else 'WRG')
#     l_dev = l_opt
#     # dnn_partition[set_ep(l_dev)].append(prof_data[node_list[NUM_OF_NODES-1]]['graph_index'])
#     dnn_partition[set_ep(l_dev)].append(graph_idx_dict[node_list[NUM_OF_NODES-1]])

#     # print("line: ", l_dev, "graph_idx: ", prof_data[node_list[NUM_OF_NODES-1]]['graph_index'])
#     l_print = lambda dev, idx : l_cpu[idx] if dev == 'cpu' else l_pim[idx]

#     check_list = []
#     check_value = {}
#     check_value[node_list[NUM_OF_NODES-1]] = (schedule_graph[node_list[NUM_OF_NODES-1]][dev_list.index(l_dev)].node_cost, schedule_graph[node_list[NUM_OF_NODES-1]][dev_list.index(l_dev)].edge_cost)
#     for j in reversed(range(1, NUM_OF_NODES)):
#         l_dev = l_print(l_dev, j)
#         check_value[node_list[j-1]] = (schedule_graph[node_list[j-1]][dev_list.index(l_dev)].node_cost, schedule_graph[node_list[j-1]][dev_list.index(l_dev)].edge_cost)
#         if set_ep(l_dev) == "PIMExecutionProvider":
#             check_list.append(node_list[j-1])
#         dnn_partition[set_ep(l_dev)].append(graph_idx_dict[node_list[j-1]])

#     for key, value in dnn_partition.items():
#         value.sort(reverse=True)

#     # print("Check value")

#     # for key in reversed(list(check_value.keys())):
#     #     print(check_value[key])

#     # print("PARTITION RESULT")
#     # print(check_list)
#     # print(dnn_partition)
#     return dnn_partition

def parse_profile_results(sess_time, kernel_time_only=False, threshold=0):
    node_time = {}
    node_provider = {}
    total = 0
    for item in sess_time:
        if item["cat"] == "Node" and "dur" in item and "args" in item and "op_name" in item["args"]:
            if "provider" in item["args"]:
                device = "CPU" if item["args"]["provider"] == "CPUExecutionProvider" else "PIM"
                if item["name"] not in node_provider:
                    node_provider[item["name"]] = device
                else:
                    assert node_provider[item["name"]] == device
            elif kernel_time_only:
                continue

            if item["name"] in node_time:
                node_time[item["name"]] += item["dur"]
            else:
                node_time[item["name"]] = item["dur"]
            total += item["dur"]

    results = []
    if (threshold > 0):
        results.append(f"Threshold of Percentage > {threshold:.2f}%")

    results.append(f"Duration\tPercentage\tProvider\tName")
    for k, v in sorted(node_time.items(), key=lambda x: x[1], reverse=True):
        provider = node_provider[k] if k in node_provider else ""
        ratio = v / total
        if ratio > threshold:
            results.append(f"{v}\t{ratio * 100.0:5.2f}\t{provider}\t{k}")

    return results

def parse_pim_results(pim_profile_records):
    graph_idx_list = []
    for item in pim_profile_records:
        if (item['name'].find('_kernel_time') != -1):
            node_name = item['name'].replace('_kernel_time', '')
            ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
            op_name = item['args']['op_name']

            if op_name not in ["MemcpyFromHost", "MemcpyToHost"]:
                if ep_type == "pim":
                    pim_executable_nodes.append(node_name)
                graph_idx_list.append(item['args']['graph_index'])
    # print(graph_idx_list)

    return pim_executable_nodes

def parse_cpu_results(cpu_profile_records):
    # # Prev. Roberta
    # pim_executable_nodes = ['Add_34', 'Add_42', 'MatMul_117', 'Mul_120', 'Add_1174', 'Gemm_1177', 'Gemm_1179']

    # # # Roberta
    # # pim_executable_nodes = ['Add_35', 'Sub_37', 'Add_42', 'Add_46', 'Add_50', 'Add_48', 'Add_52', 'MatMul_103', 'Add_105', 'Sub_107', 'Add_112', 'Add_116', 'Add_118', 'Add_125', 'MatMul_127', 'Add_129', 'Sub_131', 'Add_136', 'Add_140', 'Add_144', 'Add_142', 'Add_146', 'MatMul_197', 'Add_199', 'Sub_201', 'Add_206', 'Add_210', 'Add_212', 'Add_219', 'MatMul_221', 'Add_223', 'Sub_225', 'Add_230', 'Add_234', 'Add_238', 'Add_236', 'Add_240', 'MatMul_291', 'Add_293', 'Sub_295', 'Add_300', 'Add_304', 'Add_306', 'Add_313', 'MatMul_315', 'Add_317', 'Sub_319', 'Add_324', 'Add_328', 'Add_332', 'Add_330', 'Add_334', 'MatMul_385', 'Add_387', 'Sub_389', 'Add_394', 'Add_398', 'Add_400', 'Add_407', 'MatMul_409', 'Add_411', 'Sub_413', 'Add_418', 'Add_422', 'Add_426', 'Add_424', 'Add_428', 'MatMul_479', 'Add_481', 'Sub_483', 'Add_488', 'Add_492', 'Add_494', 'Add_501', 'MatMul_503', 'Add_505', 'Sub_507', 'Add_512', 'Add_516', 'Add_520', 'Add_518', 'Add_522', 'MatMul_573', 'Add_575', 'Sub_577', 'Add_582', 'Add_586', 'Add_588', 'Add_595', 'MatMul_597', 'Add_599', 'Sub_601', 'Add_606', 'Add_610', 'Add_614', 'Add_612', 'Add_616', 'MatMul_667', 'Add_669', 'Sub_671', 'Add_676', 'Add_680', 'Add_682', 'Add_689', 'MatMul_691', 'Add_693', 'Sub_695', 'Add_700', 'Add_704', 'Add_708', 'Add_706', 'Add_710', 'MatMul_761', 'Add_763', 'Sub_765', 'Add_770', 'Add_774', 'Add_776', 'Add_783', 'MatMul_785', 'Add_787', 'Sub_789', 'Add_794', 'Add_798', 'Add_802', 'Add_800', 'Add_804', 'MatMul_855', 'Add_857', 'Sub_859', 'Add_864', 'Add_868', 'Add_870', 'Add_877', 'MatMul_879', 'Add_881', 'Sub_883', 'Add_888', 'Add_892', 'Add_896', 'Add_894', 'Add_898', 'MatMul_949', 'Add_951', 'Sub_953', 'Add_958', 'Add_962', 'Add_964', 'Add_971', 'MatMul_973', 'Add_975', 'Sub_977', 'Add_982', 'Add_986', 'Add_990', 'Add_988', 'Add_992', 'MatMul_1043', 'Add_1045', 'Sub_1047', 'Add_1052', 'Add_1056', 'Add_1058', 'Add_1065', 'MatMul_1067', 'Add_1069', 'Sub_1071', 'Add_1076', 'Add_1080', 'Add_1084', 'Add_1082', 'Add_1086', 'MatMul_1137', 'Add_1139', 'Sub_1141', 'Add_1146', 'Add_1150', 'Add_1152', 'Add_1159', 'MatMul_1161', 'Add_1163', 'Sub_1165', 'Add_1170', 'Add_1174', 'Gemm_1177']
    # # Bert
    # pim_executable_nodes = ['Add_22', 'Sub_24', 'Add_29', 'Add_33', 'Add_37', 'Add_35', 'Add_51', 'MatMul_98', 'Add_100', 'Sub_102', 'Add_107', 'Add_111', 'Add_113', 'Mul_119', 'MatMul_122', 'Add_124', 'Sub_126', 'Add_131', 'Add_135', 'Add_139', 'Add_137', 'Add_153', 'MatMul_200', 'Add_202', 'Sub_204', 'Add_209', 'Add_213', 'Add_215', 'Mul_221', 'MatMul_224', 'Add_226', 'Sub_228', 'Add_233', 'Add_237', 'Add_241', 'Add_239', 'Add_255', 'MatMul_302', 'Add_304', 'Sub_306', 'Add_311', 'Add_315', 'Add_317', 'Mul_323', 'MatMul_326', 'Add_328', 'Sub_330', 'Add_335', 'Add_339', 'Add_343', 'Add_341', 'Add_357', 'MatMul_404', 'Add_406', 'Sub_408', 'Add_413', 'Add_417', 'Add_419', 'Mul_425', 'MatMul_428', 'Add_430', 'Sub_432', 'Add_437', 'Add_441', 'Gemm_444']
    # # # GPT-2
    # # pim_executable_nodes = ['Add_28', 'Sub_33', 'Add_38', 'Add_42', 'Gemm_55', 'Gemm_174', 'Add_179', 'Sub_181', 'Add_186', 'Add_190', 'Gemm_203', 'Mul_213']
    # # Inception
    ######################
    ## 1. Without Gemm
    ######################
    # 1) BERT
    pim_executable_nodes = ['Add_22', 'Sub_24', 'Add_29', 'Add_33', 'Add_37', 'Add_35', 'Add_51', 'MatMul_98', 'Add_100', 'Sub_102', 'Add_107', 'Add_111', 'Add_113']
    # 2) RoBERTa
    # pim_executable_nodes = ['Add_35', 'Sub_37', 'Add_42', 'Add_46', 'Add_50', 'Add_48', 'Add_52', 'MatMul_103', 'Add_105', 'Sub_107', 'Add_112', 'Add_116', 'Add_118']

    partition_map = {'CPUExecutionProvider' : [], 'PIMExecutionProvider' : []}
    for item in cpu_profile_records:
        if (item['name'].find('_kernel_time') != -1):
            node_name = item['name'].replace('_kernel_time', '')
            ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
            op_name = item['args']['op_name']


            if node_name in pim_executable_nodes:
                partition_map['PIMExecutionProvider'].append(int(item['args']['graph_index']))
            else:
                partition_map['CPUExecutionProvider'].append(int(item['args']['graph_index']))

    return partition_map

def run_bfs(DCG, edge_dict, graph_idx_dict, pim_op_list):
    dcg_node_list = list(DCG.keys())

    ua_edge_list = []
    for key, values in edge_dict.items():
        value = values[0]
        attr = str(value[2]) + str(value[0]) + str(value[1])
        if attr not in ua_edge_list:
            ua_edge_list.append(attr)

    import numpy as np
    explored_edge_list_max = len(ua_edge_list)

    dnn_partition = {'CPUExecutionProvider' : [], 'PIMExecutionProvider' : []}
    added_node_list = []

    ###########
    ## BFS
    ###########
    start_bfs = time.time()
    visit = list()
    queue = list()

    node_idx = 0

    start_nodes = DCG[dcg_node_list[node_idx]]
    for dev_idx, node_attr in enumerate(start_nodes):
        if node_attr.color != None:
            queue.append((node_idx, dev_idx))
            # print(DCG[dcg_node_list[node_idx]])

    while queue:
        ## (1) Dequeue queue's head
        current_node_idx, current_dev_idx = queue.pop(0)
        current_node_attr = DCG[dcg_node_list[current_node_idx]][current_dev_idx]
        if current_node_attr.color == None:
            continue
        else:
            # print("\nPopped node: ", dcg_node_list[current_node_idx], "\t", current_dev_idx)
            if (current_node_idx, current_dev_idx) not in visit:
                if current_node_attr.color != None:
                    visit.append((current_node_idx, current_dev_idx))

                    ## (2) Enqueue Successor
                    if current_node_idx < len(dcg_node_list) - 1:
                        next_node_idx = current_node_idx + 1
                        next_node_list = []
                        for next_dev_idx, next_node in enumerate(DCG[dcg_node_list[next_node_idx]]):
                            if next_node.color != None:
                                if next_node.color =='white':
                                    next_node.color = 'black'
                                    next_node_list.append((next_node_idx, next_dev_idx))
                        queue.extend(next_node_list)
                    current_node_name = dcg_node_list[current_node_idx]             

                    ## (3)
                    backward_node_list = []
                    if current_node_attr.src_ptr !=None:
                        backward_node_name, backward_dev_idx = DCG[dcg_node_list[current_node_idx]][current_dev_idx].src_ptr
                        backward_node_idx = dcg_node_list.index(backward_node_name)
                        backward_node_list.append((backward_node_idx, backward_dev_idx))                    
                    
                    if current_node_attr.src_next_ptr !=None:
                        backward_temp_node_attr = DCG[dcg_node_list[current_node_idx]][current_dev_idx].src_next_ptr
                        backward_node_name, backward_dev_idx = backward_temp_node_attr
                        backward_node_idx = dcg_node_list.index(backward_node_name)
                        backward_node_list.append((backward_node_idx, backward_dev_idx))                    
                    # print(backward_node_list)

                    ##Edge merging & Explored edge list update                 
                    if backward_node_list !=[]:
                        set_explored_edge_list = []
                        set_explored_path = []
                        for i in range(len(backward_node_list)):                        
                            incoming_node_idx, incoming_dev_idx = backward_node_list[i] 
                            incoming_node_name = dcg_node_list[incoming_node_idx]
                            dma_size = DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].size
                            incoming_node_ua_edge = str(dma_size) + str(incoming_dev_idx) + str(current_dev_idx)
                            ua_edge_list_idx = ua_edge_list.index(incoming_node_ua_edge)

                            before = np.array(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_edge_list).astype(bool)
                            current = np.array(np.eye(explored_edge_list_max)[ua_edge_list_idx]).astype(bool)
                            result = np.bitwise_or(before, current)

                            # if len(backward_node_list) >= 2:
                            set_explored_edge_list.append(result)
                            set_explored_path.append(str(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_path))                        
                            # print("Incoming edge number:", len(backward_node_list))
                            # print(set_explored_edge_list, set_explored_path)
                        if len(set_explored_edge_list) == 1: 
                            merged_edge_list = set_explored_edge_list[0]
                            merged_path  = set_explored_path[0]
                            DCG[current_node_name][current_dev_idx].explored_edge_list = merged_edge_list
                            
                            distance_merge_to_current = current_node_idx - incoming_node_idx
                            append_list = []
                            for j in range(distance_merge_to_current-1):    
                                append_node_name = dcg_node_list[incoming_node_idx+j+1]
                                append_list.append(append_node_name)
                                append_list.append('->')
                            if distance_merge_to_current == 1:
                                DCG[current_node_name][current_dev_idx].explored_path = str(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_path)+'->'+str(current_node_name)+':'+str(current_dev_idx)
                            else:
                                DCG[current_node_name][current_dev_idx].explored_path = str(DCG[dcg_node_list[incoming_node_idx]][incoming_dev_idx].explored_path)+'->' + str(append_list)+':0->'+str(current_node_name)+':'+str(current_dev_idx)

                        else:
                            # set_table = np.empty(shape=[len(ua_edge_list),len(set_explored_edge_list)])
                            
                            set_incoming = []
                            for i in range (0, len(set_explored_edge_list)):
                                set_index_vector = []
                                for j in range (0, len(ua_edge_list)):
                                    if set_explored_edge_list[i][j] == True:
                                        set_index_vector.append(j)
                                set_incoming.append(set_index_vector)
                            
                            ## Only implement with 2 set
                            set0 = set_incoming[0]
                            set1 = set_incoming[1]
                            if all(i in set1 for i in set0):
                                DCG[current_node_name][current_dev_idx].explored_edge_list = set_explored_edge_list[1]
                                incoming_node_idx = backward_node_list[1][0]
                                distance_merge_to_current = current_node_idx - incoming_node_idx
                                append_list = []
                                for j in range(0, distance_merge_to_current-1):    
                                    append_node_name = dcg_node_list[incoming_node_idx+j+1]
                                    append_list.append(append_node_name)
                                    append_list.append('->')
                                if append_list!=[]:
                                    # by jwlee
                                    tmp_append_node_name = append_list[0]
                                    # DCG[current_node_name][current_dev_idx].explored_path = str(set_explored_path[1])+'->' + str(append_list)+':0->'+str(current_node_name)+':'+str(current_dev_idx)
                                    DCG[current_node_name][current_dev_idx].explored_path = str(set_explored_path[1])+'->' + str(tmp_append_node_name)+':0->'+str(current_node_name)+':'+str(current_dev_idx)

                                    # print(append_list)
                                    # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                                else:
                                    DCG[current_node_name][current_dev_idx].explored_path = str(set_explored_path[1])+'->' + str(current_node_name)+':'+str(current_dev_idx)
                                    # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                            ##Case for not mergable set operation or Nth set... Need to modify
                            # else:                                

                    else :
                        if current_node_idx == 0:
                            DCG[current_node_name][current_dev_idx].explored_path = str('START')+'->'+ str(current_node_name)+':'+str(current_dev_idx)
                            # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                            DCG[current_node_name][current_dev_idx].explored_edge_list = np.zeros(explored_edge_list_max)
                        else :                        
                            DCG[current_node_name][current_dev_idx].explored_path = str(DCG[dcg_node_list[current_node_idx-1]][0].explored_path) +'->'+ str(current_node_name)+':'+str(current_dev_idx)
                            DCG[current_node_name][current_dev_idx].explored_edge_list = DCG[dcg_node_list[current_node_idx-1]][0].explored_edge_list
                            # print("node: ", current_node_name, "\t", "dev_idx: ", current_dev_idx)
                    
                    
                    ##Early termination 
                    end_point = np.array(DCG[current_node_name][current_dev_idx].explored_edge_list).astype(bool)
                    all_pass = np.array(np.ones(len(ua_edge_list)).astype(bool))
                    is_end = np.bitwise_and(end_point, all_pass)
                    if  all(is_end):
                        # print('All explored edge list is True:', current_node_name)
                        # print(DCG[current_node_name][current_dev_idx].explored_edge_list)
                        # print(DCG[current_node_name][current_dev_idx].explored_path)
                        tmp_str = DCG[current_node_name][current_dev_idx].explored_path.split('->')
                        # print(tmp_str)
                        for tmp in tmp_str:
                            # print(tmp)
                            if tmp != 'START':
                                # print(tmp)
                                ttmp = tmp.split(':')
                                tmp_node_name = ttmp[0]
                                tmp_dev_idx = ttmp[1]
                                # print(tmp_node_name, "\t", tmp_dev_idx)
                                if tmp_dev_idx == '0':
                                    dnn_partition['CPUExecutionProvider'].append(graph_idx_dict[tmp_node_name])
                                else:
                                    dnn_partition['PIMExecutionProvider'].append(graph_idx_dict[tmp_node_name])
                                    # print("\t", tmp_node_name)
                                added_node_list.append(tmp_node_name)
                        break
            
            # print(DCG[current_node_name][current_dev_idx].explored_edge_list)
            # print(DCG[current_node_name][current_dev_idx].explored_path)


    end_bfs = time.time()
    print("BFS time: ", end_bfs - start_bfs)

    # if set(pim_op_list) != set(added_node_list):
    #     not_included = set(pim_op_list) - set(added_node_list)
    #     for tmp_node in not_included:
    #         dnn_partition['CPUExecutionProvider'].append(graph_idx_dict[tmp_node])

    not_included = set(dcg_node_list) - set(added_node_list)
    for tmp_node in not_included:
        dnn_partition['CPUExecutionProvider'].append(graph_idx_dict[tmp_node])    

    return dnn_partition


def group_profile_results(sess_time, kernel_time_only=False, threshold=0):
    op_time = {}
    op_records = {}
    op_cpu_time = {}
    op_cpu_records = {}
    total = 0
    for item in sess_time:
        if item["cat"] == "Node" and "dur" in item and "args" in item and "op_name" in item["args"]:
            if kernel_time_only and "provider" not in item["args"]:
                continue

            op_name = item["args"]["op_name"]
            if op_name in op_time:
                op_time[op_name] += item["dur"]
                op_records[op_name] += 1
            else:
                op_time[op_name] = item["dur"]
                op_records[op_name] = 1

            total += item["dur"]

            is_cpu = "provider" in item["args"] and item["args"]["provider"] == "CPUExecutionProvider"
            if is_cpu:
                if op_name in op_cpu_time:
                    op_cpu_time[op_name] += item["dur"]
                    op_cpu_records[op_name] += 1
                else:
                    op_cpu_time[op_name] = item["dur"]
                    op_cpu_records[op_name] = 1

    results = [f"Duration\tPercentage\tCalls\tCpu_Duration\tCpu_Calls\tName"]
    for k, v in sorted(op_time.items(), key=lambda x: x[1], reverse=True):
        calls = op_records[k]
        cpu_time = op_cpu_time[k] if k in op_cpu_time else 0
        cpu_calls = op_cpu_records[k] if k in op_cpu_records else 0
        ratio = v / total
        if ratio > threshold:
            results.append(f"{v}\t{ratio * 100.0:5.2f}\t{calls}\t{cpu_time}\t{cpu_calls}\t{k}")
    return results


def get_dim_from_type_proto(dim):
    return getattr(dim, dim.WhichOneof('value')) if type(dim.WhichOneof('value')) == str else None


def get_shape_from_type_proto(type_proto):
    return [get_dim_from_type_proto(d) for d in type_proto.tensor_type.shape.dim]


def create_dummy_inputs(onnx_model_path, batch_size, sequence_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    dummy_inputs = {}
    for input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(input.type)
        symbol_dims = []
        for i, dim in enumerate(shape):
            if type(dim) == str:
                symbol_dims.append(i)

        # allowed symbolic dimensions: batch_size and sequence_length
        if len(symbol_dims) > 2:
            return None
        if len(symbol_dims) > 0:
            shape[symbol_dims[0]] = batch_size
        if len(symbol_dims) > 1:
            shape[symbol_dims[1]] = sequence_length

        elem_type = input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)
        data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs


def create_gpt2_inputs(onnx_model_path, batch_size, sequence_length, past_sequence_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    # The symbolic name shall be same as those used in Gpt2Helper.export_onnx(...) function.
    symbols = {
        'batch_size': batch_size,
        'seq_len': sequence_length,
        'past_seq_len': past_sequence_length,
        'total_seq_len': sequence_length + past_sequence_length
    }

    dummy_inputs = {}
    for input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(input.type)
        for i, dim in enumerate(shape):
            if type(dim) == str and dim not in symbols.keys():
                raise RuntimeError(f"symbol is not supported: {dim}")
            else:
                shape[i] = symbols[dim]

        elem_type = input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)
        data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs


def create_longformer_inputs(onnx_model_path, batch_size, sequence_length, global_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    symbols = {'batch_size': batch_size, 'sequence_length': sequence_length}

    dummy_inputs = {}
    for input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(input.type)
        for i, dim in enumerate(shape):
            if type(dim) == str and dim not in symbols.keys():
                raise RuntimeError(f"symbol is not supported: {dim}")
            else:
                shape[i] = symbols[dim]

        elem_type = input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)

        if "global" in input.name:
            data = numpy.zeros(shape, dtype=data_type)
            data[:, :global_length] = 1
        else:
            data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs

def infer_model(onnx_model_path, batch_size, sequence_length):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))

    org_file_name = onnx_model_path.split('/')[-1]
    org_file_path = onnx_model_path.replace(org_file_name, '')
    
    new_file_name = org_file_name.split('.')[0] + '-inferred' + '.onnx'
    new_file_path = org_file_path + new_file_name

    graph = onnx_model.model.graph

    for inp in graph.input:
        shape_proto = inp.type.tensor_type.shape.dim
        for dim_proto in shape_proto:
            if dim_proto.HasField('dim_param'):
                if dim_proto.dim_param == "batch_size":
                    dim_proto.ClearField('dim_param')
                    dim_proto.dim_value = batch_size
                elif dim_proto.dim_param == "max_seq_len":
                    dim_proto.ClearField('dim_param')
                    dim_proto.dim_value = sequence_length
                elif dim_proto.dim_param == "sentence_length":
                    dim_proto.ClearField('dim_param')
                    dim_proto.dim_value = sequence_length                
                else:
                    print("ERR")

    from onnx import helper, shape_inference
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    inferred_model = shape_inference.infer_shapes(onnx_model.model)
    # inferred_model = SymbolicShapeInference.infer_shapes(onnx_model.model)
    onnx.save(inferred_model, new_file_path)

    # print("New path: ", new_file_path)

    return new_file_path

def pim_helper(inferred_model):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(inferred_model))
    graph = onnx_model.model.graph
    align_map = {"mm_b" : [], "mm_y": []}

    for node in graph.node:
        if node.op_type == "MatMul":
            align_map["mm_b"].append(node.input[1])

    for node in graph.node:
        if "Gemm" in node.name:
            need_align = True
            for attr in node.attribute:
                if attr.name == "transB":
                    need_align = False
            if need_align:
                align_map["mm_b"].append(node.input[1])
                align_map["mm_y"].append(node.input[2])

    return align_map

def run_cpu(args, all_inputs):
    cpu_profile_file = run_profile("cpu", args.model, False, args.basic_optimization, args.thread_num, args.batch_size,
                               args.sequence_length, all_inputs, {})

    cpu_profile_records = load_profile_json(cpu_profile_file)

    return cpu_profile_file, cpu_profile_records

def run_pim(args, all_inputs):
    align_map = pim_helper(args.model)

    pim_profile_file = run_profile("pim", args.model, True, args.basic_optimization, args.thread_num, args.batch_size,
                               args.sequence_length, all_inputs, align_map)

    pim_profile_records = load_profile_json(pim_profile_file)

    return pim_profile_file, pim_profile_records

def run_edge(args, all_inputs, cpu_profile_records, cpu_profile_file, pim_profile_file):
    align_map = pim_helper(args.model)

    partition_map = parse_cpu_results(cpu_profile_records)

    # print(partition_map)

    edge_profile_file = run_profile("edge", args.model, True, args.basic_optimization, args.thread_num, args.batch_size,
                                        args.sequence_length, all_inputs, align_map, partition_map)

    edge_profile_records = load_profile_json(edge_profile_file)

    dnn_partition = get_optimal_partition(cpu_profile_file, pim_profile_file, edge_profile_file)

    return dnn_partition

def get_partition(args, all_inputs, cpu_profile_records, cpu_profile_file, pim_profile_file):
    align_map = pim_helper(args.model)
    DCG, edge_dict, graph_idx_dict, pim_op_list = generate_dcg(cpu_profile_file, pim_profile_file)
    partition_map = run_bfs(DCG, edge_dict, graph_idx_dict, pim_op_list)

    edge_profile_file = run_profile("edge", args.model, True, args.basic_optimization, args.thread_num, args.batch_size,
                                        args.sequence_length, all_inputs, align_map, partition_map)

    edge_profile_records = load_profile_json(edge_profile_file)

    dnn_partition = get_optimal_partition(cpu_profile_file, pim_profile_file, edge_profile_file)

    return dnn_partition

def run_opt(args, all_inputs, dnn_partition):
    align_map = pim_helper(args.model)

    opt_profile_file = run_profile("opt", args.model, True, args.basic_optimization, args.thread_num, args.batch_size,
                                        args.sequence_length, all_inputs, align_map, dnn_partition)

    opt_profile_records = load_profile_json(opt_profile_file)

    return opt_profile_file, opt_profile_records


def run(args):
    num_threads = args.thread_num if args.thread_num > 0 else psutil.cpu_count(logical=False)

    # Set OMP environment variable before importing onnxruntime. Needed for cpu only, and no impact for onnxruntime-gpu package.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)

    all_inputs = None
    if args.dummy_inputs == 'bert':
        all_inputs = create_bert_inputs(args.model, args.batch_size, args.sequence_length, args.samples,
                                        args.input_ids_name, args.segment_ids_name, args.input_mask_name)
    elif args.dummy_inputs == 'gpt2':
        all_inputs = create_gpt2_inputs(args.model, args.batch_size, args.sequence_length, args.past_sequence_length,
                                        args.samples)
    elif args.dummy_inputs == 'longformer':
        all_inputs = create_longformer_inputs(args.model, args.batch_size, args.sequence_length, args.global_length,
                                              args.samples)
    else:  # default
        all_inputs = create_dummy_inputs(args.model, args.batch_size, args.sequence_length, args.samples)

    # #####################
    # ####### CPU
    # #####################
    print("RUNNING CPU")

    cpu_profile_file, cpu_profile_records = run_cpu(args, all_inputs)

    lines = ["CPU PROFILE RESULT\n"]
    lines.append("=" * 64)
    lines += parse_profile_results(cpu_profile_records, args.kernel_time_only, args.threshold)

    lines.append("-" * 64)
    lines += group_profile_results(cpu_profile_records, args.kernel_time_only, args.threshold)

    # # #####################
    # # ####### PIM
    # # #####################
    # print("RUNNING PIM")

    # pim_profile_file, pim_profile_records = run_pim(args, all_inputs)
    # lines += ["PIM PROFILE RESULT\n"]
    # lines.append("=" * 64)
    # lines += parse_profile_results(pim_profile_records, args.kernel_time_only, args.threshold)

    # lines.append("-" * 64)
    # lines += group_profile_results(pim_profile_records, args.kernel_time_only, args.threshold)

    # # #####################
    # # ####### EDGE
    # # #####################
    # print("RUNNING EDGE")

    # dnn_partition = get_partition(args, all_inputs, cpu_profile_records, cpu_profile_file, pim_profile_file)
    # # dnn_partition = run_edge(args, all_inputs, cpu_profile_records, cpu_profile_file, pim_profile_file)
    # print("DNN PARTITION")
    # print(dnn_partition)
    
    # # #####################
    # # ####### OPT
    # # #####################

    # print("RUNNING OPT")
    # opt_profile_file, opt_profile_records = run_opt(args, all_inputs, dnn_partition)
    # lines += ["OPT PROFILE RESULT\n"]
    # lines.append("=" * 64)
    # lines += parse_profile_results(opt_profile_records, args.kernel_time_only, args.threshold)

    # lines.append("-" * 64)
    # lines += group_profile_results(opt_profile_records, args.kernel_time_only, args.threshold)

    return lines


if __name__ == '__main__':
    args = parse_arguments()
    print("Arguments", args)

    from pim_benchmark_helper import setup_logger
    setup_logger(args.verbose)

    results = run(args)
    print("Results:")
    print("-" * 64)
    for line in results:
        print(line)
    # print("SUCCESS")

