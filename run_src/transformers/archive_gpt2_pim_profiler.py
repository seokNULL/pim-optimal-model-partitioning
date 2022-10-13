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


def run_profile(device, onnx_model_path, use_pim, basic_optimization, thread_num, batch_size, sequence_length, all_inputs, align_map, partition_map=None, cpu_thread_map=None):
    from pim_benchmark_helper import create_onnxruntime_session

    session = create_onnxruntime_session(device,
                                         onnx_model_path,
                                         use_pim,
                                         partition_map=partition_map,
                                         align_map=align_map,
                                         cpu_thread_map=cpu_thread_map,
                                         enable_all_optimization=not basic_optimization,
                                         num_threads=thread_num,
                                         enable_profiling=True)

    # for inputs in all_inputs:
    #     _ = session.run(None, inputs)

    for inputs in all_inputs:
        result = session.run(None, inputs)
        print("check result")
        print(result[0])

    profile_file = session.end_profiling()
    return profile_file

def load_profile_json(profile_file):
    print(f"loading profile output {profile_file} ...")

    with open(profile_file, "r") as f:
        sess_time = json.load(f)

    assert isinstance(sess_time, list)
    return sess_time

def generate_dcg(cpu_serial_profile_file, cpu_parallel_profile_file, pim_profile_file):

    @dataclass
    class Datafield:
        sink_ptr: tuple
        size: int
        sink_next_ptr: []
        src_ptr: tuple
        src_next_ptr: []
        color : str
        profiled_edge_list : []
        profiled_path : []
        cost: float

    NUM_OF_NODES = 0 
    NUM_OF_DEVICES = 3
    INF = 1000000000000000000000000
    prof_data = {}

    graph_idx_dict = {}

    # BUILD DCG
    pim_op_list = []
    omp_op_list = []
    cpu_prof = os.path.join(cpu_serial_profile_file)
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

                if op_name in ["Add", "Mul", "Sub", "Div", "MatMul", "Gemm"]:
                    omp_op_list.append(node_name)

    print("NUM_OF_NODES")
    print(NUM_OF_NODES)

    print("OMP LIST")
    print(omp_op_list)

    pim_prof = os.path.join(pim_profile_file)
    with open(pim_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')
                ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
                if node_name in prof_data.keys() and ep_type == 'pim':
                    pim_op_list.append(node_name)

    # empty_datafield = Datafield(sink_ptr=None, size=None, sink_next_ptr=None, src_ptr=None, src_next_ptr=None, color=None, profiled_edge_list=None, profiled_path=None, cost=None)
    node_list = list(prof_data.keys())

    start_gen_dcg = time.time()

    DCG = {}
    # DEVICE NO. cpu_serial: 0, cpu_parallel: 1, pim: 2
    dev_list = ["cpu_serial", "cpu_parallel", "pim"]
    for node_name in prof_data.keys():
        DCG[node_name] = [None] * NUM_OF_DEVICES
        for k in range(NUM_OF_DEVICES):
            # print(k)
            # print(dev_list[k])
            if dev_list[k] == 'cpu_serial':
                DCG[node_name][k] = Datafield(None, None, None, None, None, 'white', None, None, INF)
            else:
                DCG[node_name][k] = Datafield(None, None, None, None, None, None, None, None, INF)


    for k in range(NUM_OF_DEVICES):
        prof_file = None
        if dev_list[k] == 'cpu_serial':
            prof_file = cpu_serial_profile_file
        elif dev_list[k] == 'cpu_parallel':
            prof_file = cpu_parallel_profile_file
        elif dev_list[k] == 'pim':
            prof_file = pim_profile_file
        else:
            IndexError
        print("prof_file: ", prof_file)
        node_prof = os.path.join(prof_file)
        with open(node_prof, 'r') as f:
            data = json.load(f)
            for item in data:
                if (item['name'].find('_kernel_time') != -1):
                    node_name = item['name'].replace('_kernel_time', '')
                    print("\nNode name: ", node_name)
                    print("Dev  name: ", dev_list[k])
                    ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
                    node_cost = item['dur']
                    print("Node cost: ", node_cost)
                    if "Memcpy" not in node_name:
                        if dev_list[k] == 'cpu_serial':
                            DCG[node_name][k].cost = node_cost
                        elif dev_list[k] == 'cpu_parallel':
                            if node_name in omp_op_list:
                                DCG[node_name][k].cost = node_cost
                        elif dev_list[k] == 'pim':
                            if node_name in pim_op_list:
                                DCG[node_name][k].cost = node_cost
                        else:
                            IndexError
    edge_dict = {}

    cpu_s_pim_source_list = []
    cpu_s_omp_source_list = []

    cpu_p_pim_source_list = []
    cpu_p_omp_source_list = []

    pim_pim_source_list = []
    pim_omp_source_list = []

    for cur_node, attr in prof_data.items():
        # print("Current node: ", cur_node)
        sink_nodes = attr['sink_nodes'][:]
        src_nodes = attr['src_nodes'][:]
        print("\nCurrent node: ", cur_node)
        print("src_nodes: ", src_nodes)
        print("sink_nodes: ", sink_nodes)
        for dev in dev_list:
            # 0. CPU Serial
            if dev == "cpu_serial":
                print("Dev: cpu_serial")
                # 0) To PIM.
                pim_sink_nodes = [x for x in sink_nodes if x in pim_op_list]
                pim_src_nodes = [x for x in src_nodes if x in pim_op_list]
                if pim_sink_nodes:
                    DCG[cur_node][dev_list.index("cpu_serial")].sink_ptr = (pim_sink_nodes[0], dev_list.index("pim"))
                    DCG[cur_node][dev_list.index("cpu_serial")].size = attr['elem_size']
                    DCG[cur_node][dev_list.index("cpu_serial")].color = 'white'
                    edge = ((cur_node, dev_list.index("cpu_serial")), (pim_sink_nodes[0], dev_list.index("pim")))
                    edge_attr = (dev_list.index("cpu_serial"), dev_list.index("pim"), attr['elem_size'])
                    edge_dict[edge] = [edge_attr, 'white']
                if pim_src_nodes:
                    print("pim_src_nodes: ", pim_src_nodes)
                    # print("cpu_s_pim_source_list: ", cpu_s_pim_source_list)
                    if (pim_src_nodes[0], dev_list.index("pim")) not in cpu_s_pim_source_list:
                        DCG[cur_node][dev_list.index("cpu_serial")].src_ptr = (pim_src_nodes[0], dev_list.index("pim"))
                        print(DCG[cur_node][dev_list.index("cpu_serial")].src_ptr)
                        edge = ((pim_src_nodes[0], dev_list.index("pim")), (cur_node, dev_list.index("cpu_serial")))
                        edge_attr = (dev_list.index("pim"), dev_list.index("cpu_serial"), attr['elem_size'])
                        cpu_s_pim_source_list.append((pim_src_nodes[0], dev_list.index("pim")))
                    if len(pim_src_nodes) > 1:
                        DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr = []
                        for i in range(1, len(pim_src_nodes)):
                            src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                            if (pim_src_nodes[i], dev_list.index("pim")) not in cpu_s_pim_source_list:
                                cpu_s_pim_source_list.append((pim_src_nodes[i], dev_list.index("pim")))
                                src_data_field.src_next_ptr = (pim_src_nodes[i], dev_list.index("pim"))
                                edge = ((pim_src_nodes[i], dev_list.index("pim")), (cur_node, dev_list.index("cpu_serial")))
                                edge_attr = (dev_list.index("pim"), dev_list.index("cpu_serial"), attr['elem_size'])                          
                                DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr.append((pim_src_nodes[i], dev_list.index("pim"))) 
                                print(DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr)
                        if DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr == []:
                            DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr = None
                # 1) To CPU Parallel:
                omp_sink_nodes = [x for x in sink_nodes if x in omp_op_list]
                omp_src_nodes = [x for x in src_nodes if x in omp_op_list]
                if omp_sink_nodes:
                    DCG[cur_node][dev_list.index("cpu_serial")].sink_ptr = (omp_sink_nodes[0], dev_list.index("cpu_parallel"))
                    DCG[cur_node][dev_list.index("cpu_serial")].size = attr['elem_size']
                    DCG[cur_node][dev_list.index("cpu_serial")].color = 'white'
                    edge = ((cur_node, dev_list.index("cpu_serial")), (omp_sink_nodes[0], dev_list.index("cpu_parallel")))
                    edge_attr = (dev_list.index("cpu_serial"), dev_list.index("cpu_parallel"), attr['elem_size'])
                    edge_dict[edge] = [edge_attr, 'white']
                if omp_src_nodes:
                    if (omp_src_nodes[0], dev_list.index("cpu_parallel")) not in cpu_s_omp_source_list:
                        if not DCG[cur_node][dev_list.index("cpu_serial")].src_ptr:
                            DCG[cur_node][dev_list.index("cpu_serial")].src_ptr = (omp_src_nodes[0], dev_list.index("cpu_parallel"))
                            edge = ((omp_src_nodes[0], dev_list.index("cpu_parallel")), (cur_node, dev_list.index("cpu_serial")))
                            edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("cpu_serial"), attr['elem_size'])
                            cpu_s_omp_source_list.append((omp_src_nodes[0], dev_list.index("cpu_parallel")))
                        else:
                            if DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr is None:
                                DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr = []
                            for i in range(0, len(omp_src_nodes)):
                                src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                                if (omp_src_nodes[i], dev_list.index("cpu_parallel")) not in cpu_s_omp_source_list:
                                    cpu_s_omp_source_list.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                                    src_data_field.src_next_ptr = (omp_src_nodes[i], dev_list.index("cpu_parallel"))
                                    edge = ((omp_src_nodes[i], dev_list.index("cpu_parallel")), (cur_node, dev_list.index("cpu_serial")))
                                    edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("cpu_serial"), attr['elem_size'])                          
                                    DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                            if DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr == []:
                                DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr = None                            
                    if len(omp_src_nodes) > 1:
                        if DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr is None:
                            DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr = []
                        for i in range(1, len(omp_src_nodes)):
                            src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                            if (omp_src_nodes[i], dev_list.index("cpu_parallel")) not in cpu_s_omp_source_list:
                                cpu_s_omp_source_list.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                                src_data_field.src_next_ptr = (omp_src_nodes[i], dev_list.index("cpu_parallel"))
                                edge = ((omp_src_nodes[i], dev_list.index("cpu_parallel")), (cur_node, dev_list.index("cpu_serial")))
                                edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("cpu_serial"), attr['elem_size'])                          
                                DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                        if DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr == []:
                            DCG[cur_node][dev_list.index("cpu_serial")].src_next_ptr = None
            elif dev == "cpu_parallel":
                print("Dev: cpu_parallel")
                if cur_node not in omp_op_list:
                    continue
                else:
                    DCG[cur_node][dev_list.index("cpu_parallel")].color = 'white'
                    # 0) To CPU Serial
                    if sink_nodes:
                        DCG[cur_node][dev_list.index("cpu_parallel")].sink_ptr = (sink_nodes[0], dev_list.index("cpu_serial"))
                        DCG[cur_node][dev_list.index("cpu_parallel")].size = attr['elem_size']
                        edge = ((cur_node, dev_list.index("cpu_parallel")), (sink_nodes[0], dev_list.index("cpu_serial")))
                        edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("cpu_serial"), attr['elem_size'])
                        edge_dict[edge] = [edge_attr, 'white']                              
                    if src_nodes:
                        if (src_nodes[0], dev_list.index("cpu_serial")) not in cpu_p_omp_source_list:
                            cpu_p_omp_source_list.append((src_nodes[0], dev_list.index("cpu_serial")))
                             # Add to edge_dict
                            edge = ((src_nodes[0], dev_list.index("cpu_serial")), (cur_node, dev_list.index("pim")))
                            edge_attr = (dev_list.index("cpu_serial"), dev_list.index("pim"), attr['elem_size'])
                            DCG[cur_node][dev_list.index("cpu_parallel")].src_ptr = (src_nodes[0], dev_list.index("cpu_serial"))
                        if len(src_nodes) > 1:
                            DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr = []
                            for i in range(1, len(src_nodes)):
                                src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                                if (src_nodes[i], dev_list.index("cpu_serial")) not in cpu_p_omp_source_list:
                                    cpu_p_omp_source_list.append((src_nodes[i], dev_list.index("cpu_serial")))
                                    src_data_field.src_next_ptr = (src_nodes[i], dev_list.index("cpu_serial"))
                                    edge = ((src_nodes[i], dev_list.index("cpu_serial")), (cur_node, dev_list.index("pim")))
                                    edge_attr = (dev_list.index("cpu_serial"), dev_list.index("pim"), attr['elem_size'])
                                    DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr.append((src_nodes[i], dev_list.index("cpu_serial")))
                            if DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr == []:
                                DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr = None
                    # 1) To PIM
                    pim_sink_nodes = [x for x in sink_nodes if x in pim_op_list]
                    pim_src_nodes = [x for x in src_nodes if x in pim_op_list]
                    if pim_sink_nodes:
                        DCG[cur_node][dev_list.index("cpu_parallel")].sink_ptr = (pim_sink_nodes[0], dev_list.index("pim"))
                        DCG[cur_node][dev_list.index("cpu_parallel")].size = attr['elem_size']
                        edge = ((cur_node, dev_list.index("cpu_parallel")), (pim_sink_nodes[0], dev_list.index("pim")))
                        edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("pim"), attr['elem_size'])
                        edge_dict[edge] = [edge_attr, 'white']                              
                    if pim_src_nodes:
                        if (pim_src_nodes[0], dev_list.index("pim")) not in cpu_p_pim_source_list:
                            # cpu_p_pim_source_list.append((pim_src_nodes[0], dev_list.index("pim")))
                            edge = ((pim_src_nodes[0], dev_list.index("pim")), (cur_node, dev_list.index("cpu_parallel")))
                            edge_attr = (dev_list.index("pim"), dev_list.index("cpu_parallel"), attr['elem_size'])
                            if not DCG[cur_node][dev_list.index("cpu_parallel")].src_ptr:
                                DCG[cur_node][dev_list.index("cpu_parallel")].src_ptr = (pim_src_nodes[0], dev_list.index("pim"))
                            else:
                                if DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr is None:
                                    DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr = []
                                for i in range(0, len(pim_src_nodes)):
                                    src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                                    if (pim_src_nodes[i], dev_list.index("pim")) not in cpu_p_pim_source_list:
                                        cpu_p_pim_source_list.append((pim_src_nodes[i], dev_list.index("pim")))
                                        src_data_field.src_next_ptr = (pim_src_nodes[i], dev_list.index("pim"))
                                        edge = ((pim_src_nodes[i], dev_list.index("pim")), (cur_node, dev_list.index("cpu_parallel")))
                                        edge_attr = (dev_list.index("pim"), dev_list.index("cpu_parallel"), attr['elem_size'])
                                        DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr.append((pim_src_nodes[i], dev_list.index("pim"))) 
                                if DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr == []:
                                    DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr = None   
                            cpu_p_pim_source_list.append((pim_src_nodes[0], dev_list.index("pim")))                                  
                        if len(pim_src_nodes) > 1:
                            if DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr is None:
                                DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr = []
                            for i in range(1, len(pim_src_nodes)):
                                src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                                if (pim_src_nodes[i], dev_list.index("pim")) not in cpu_p_pim_source_list:
                                    cpu_p_pim_source_list.append((pim_src_nodes[i], dev_list.index("pim")))
                                    src_data_field.src_next_ptr = (pim_src_nodes[i], dev_list.index("pim"))
                                    edge = ((pim_src_nodes[i], dev_list.index("pim")), (cur_node, dev_list.index("cpu_parallel")))
                                    edge_attr = (dev_list.index("pim"), dev_list.index("cpu_parallel"), attr['elem_size'])
                                    DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr.append((pim_src_nodes[i], dev_list.index("pim"))) 
                            if DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr == []:
                                DCG[cur_node][dev_list.index("cpu_parallel")].src_next_ptr = None               
            # 2. PIM device
            else:
                print("Dev: pim")
                if cur_node not in pim_op_list:
                    continue
                else:
                    DCG[cur_node][dev_list.index("pim")].color = 'white'
                    # 0) To CPU Serial
                    if sink_nodes:
                        DCG[cur_node][dev_list.index("pim")].sink_ptr = (sink_nodes[0], dev_list.index("cpu_serial"))
                        DCG[cur_node][dev_list.index("pim")].size = attr['elem_size']
                        edge = ((cur_node, dev_list.index("pim")), (sink_nodes[0], dev_list.index("cpu_serial")))
                        edge_attr = (dev_list.index("pim"), dev_list.index("cpu_serial"), attr['elem_size'])
                        edge_dict[edge] = [edge_attr, 'white']                              
                    if src_nodes:
                        if (src_nodes[0], dev_list.index("cpu_serial")) not in pim_pim_source_list:
                            pim_pim_source_list.append((src_nodes[0], dev_list.index("cpu_serial")))
                             # Add to edge_dict
                            edge = ((src_nodes[0], dev_list.index("cpu_serial")), (cur_node, dev_list.index("pim")))
                            edge_attr = (dev_list.index("cpu_serial"), dev_list.index("pim"), attr['elem_size'])
                            DCG[cur_node][dev_list.index("pim")].src_ptr = (src_nodes[0], dev_list.index("cpu_serial"))
                        if len(src_nodes) > 1:
                            if DCG[cur_node][dev_list.index("pim")].src_next_ptr is None:
                                DCG[cur_node][dev_list.index("pim")].src_next_ptr = []
                            for i in range(1, len(src_nodes)):
                                src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                                if (src_nodes[i], dev_list.index("cpu_serial")) not in pim_pim_source_list:
                                    pim_pim_source_list.append((src_nodes[i], dev_list.index("cpu_serial")))
                                    src_data_field.src_next_ptr = (src_nodes[i], dev_list.index("cpu_serial"))
                                    edge = ((src_nodes[i], dev_list.index("cpu_serial")), (cur_node, dev_list.index("pim")))
                                    edge_attr = (dev_list.index("cpu_serial"), dev_list.index("pim"), attr['elem_size'])
                                    DCG[cur_node][dev_list.index("pim")].src_next_ptr.append((src_nodes[i], dev_list.index("cpu_serial")))
                            if DCG[cur_node][dev_list.index("pim")].src_next_ptr == []:
                                DCG[cur_node][dev_list.index("pim")].src_next_ptr = None
                    # 1) To CPU Parallel
                    omp_sink_nodes = [x for x in sink_nodes if x in omp_op_list]
                    omp_src_nodes = [x for x in src_nodes if x in omp_op_list]
                    print("\n",cur_node)
                    if omp_sink_nodes:
                        DCG[cur_node][dev_list.index("pim")].sink_ptr = (omp_sink_nodes[0], dev_list.index("cpu_parallel"))
                        DCG[cur_node][dev_list.index("pim")].size = attr['elem_size']
                        edge = ((cur_node, dev_list.index("pim")), (omp_sink_nodes[0], dev_list.index("cpu_parallel")))
                        edge_attr = (dev_list.index("pim"), dev_list.index("cpu_parallel"), attr['elem_size'])
                        edge_dict[edge] = [edge_attr, 'white']                              
                    if omp_src_nodes:
                        if (omp_src_nodes[0], dev_list.index("cpu_parallel")) not in pim_omp_source_list:
                            # pim_omp_source_list.append((omp_src_nodes[0], dev_list.index("cpu_parallel")))
                             # Add to edge_dict
                            edge = ((omp_src_nodes[0], dev_list.index("cpu_parallel")), (cur_node, dev_list.index("pim")))
                            edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("pim"), attr['elem_size'])
                            if not DCG[cur_node][dev_list.index("pim")].src_ptr:
                                DCG[cur_node][dev_list.index("pim")].src_ptr = (omp_src_nodes[0], dev_list.index("cpu_parallel"))
                            else:
                                if DCG[cur_node][dev_list.index("pim")].src_next_ptr is None:
                                    DCG[cur_node][dev_list.index("pim")].src_next_ptr = []
                                for i in range(0, len(omp_src_nodes)):
                                    src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                                    if (omp_src_nodes[i], dev_list.index("cpu_parallel")) not in pim_omp_source_list:
                                        pim_omp_source_list.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                                        src_data_field.src_next_ptr = (omp_src_nodes[i], dev_list.index("cpu_parallel"))
                                        edge = ((omp_src_nodes[i], dev_list.index("cpu_parallel")), (cur_node, dev_list.index("pim")))
                                        edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("pim"), attr['elem_size'])
                                        DCG[cur_node][dev_list.index("pim")].src_next_ptr.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                                if DCG[cur_node][dev_list.index("pim")].src_next_ptr == []:
                                    DCG[cur_node][dev_list.index("pim")].src_next_ptr = None    
                            pim_omp_source_list.append((omp_src_nodes[0], dev_list.index("cpu_parallel")))                                 
                        if len(omp_src_nodes) > 1:
                            if DCG[cur_node][dev_list.index("pim")].src_next_ptr is None:
                                DCG[cur_node][dev_list.index("pim")].src_next_ptr = []
                            for i in range(1, len(omp_src_nodes)):
                                src_data_field = Datafield(None, None, None, None, None, 'white', None, None, INF)
                                if (omp_src_nodes[i], dev_list.index("cpu_parallel")) not in pim_omp_source_list:
                                    pim_omp_source_list.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                                    src_data_field.src_next_ptr = (omp_src_nodes[i], dev_list.index("cpu_parallel"))
                                    edge = ((omp_src_nodes[i], dev_list.index("cpu_parallel")), (cur_node, dev_list.index("pim")))
                                    edge_attr = (dev_list.index("cpu_parallel"), dev_list.index("pim"), attr['elem_size'])
                                    DCG[cur_node][dev_list.index("pim")].src_next_ptr.append((omp_src_nodes[i], dev_list.index("cpu_parallel")))
                            if DCG[cur_node][dev_list.index("pim")].src_next_ptr == []:
                                DCG[cur_node][dev_list.index("pim")].src_next_ptr = None                                 

    print("DA edges: ", edge_dict.keys())
    da_edge_cnt = 0
    cpu_serial_and_cpu_parallel_da_edges = []
    cpu_serial_and_pim_da_edges = []
    da_edges = []

    print("\n=========================")
    print("========== DCG ==========")
    print("=========================\n")

    for key, value in DCG.items():
        print("key: ", key)
        for val in value:
            print(val)
        print("\n")

    global_profiled_edge_cost = {}
    for edge, attrs in edge_dict.items():
        (src_node, src_device), (dst_node, dst_device) = edge
        # CPU_SERIAL and CPU_PARALLEL
        if (src_device == 0 and dst_device == 1) or (src_device == 1 and dst_device == 0):
            # print(attrs[0])
            if attrs[0] not in cpu_serial_and_cpu_parallel_da_edges:
                cpu_serial_and_cpu_parallel_da_edges.append(attrs[0])
        # CPU_SERIAL and PIM
        elif (src_device == 0 and dst_device == 2) or (src_device == 2 and dst_device == 0):
            # print(attrs[0])
            if attrs[0] not in cpu_serial_and_pim_da_edges:
                cpu_serial_and_pim_da_edges.append(attrs[0])
        elif (src_device == 1 and dst_device == 2) or (src_device == 2 and dst_device == 1):
            pass
            # print(edge)
            # print(attrs[0])
            # print(attrs[0])
        # print(src_node, "\t", src_device, "\t", dst_node, "\t", dst_device)
        if attrs[0] not in da_edges:
            da_edges.append(attrs[0])
            da_edge_cnt+=1
            (_src_device, _dst_device, _size) = attrs[0]
            da_edge = str(_size) + str(_src_device) + str(_dst_device)
            global_profiled_edge_cost[da_edge] = []
            if (_src_device == 0 and _dst_device == 1) or (_src_device == 1 and _dst_device == 0):
                global_profiled_edge_cost[da_edge] = [0]


    print("DA edge cnt: ", da_edge_cnt)
    print("cpu_serial_and_cpu_parallel_da_edges: ", cpu_serial_and_cpu_parallel_da_edges)
    print("cpu_serial_and_pim_da_edges: ", cpu_serial_and_pim_da_edges)
    print("da_edges: ", da_edges)

    print(len(cpu_serial_and_cpu_parallel_da_edges))
    print(len(cpu_serial_and_pim_da_edges))
    print(len(da_edges))

    remain_da_edges = list(set(da_edges) - set(cpu_serial_and_cpu_parallel_da_edges) - set(cpu_serial_and_pim_da_edges))
    print("remain_da_edges: ", remain_da_edges)
    # print("DCG keys: ", DCG.keys())

    global_profiled_edge_dict = dict.fromkeys(da_edges, 0)
    # global_profiled_edge_cost = dict.fromkeys(da_edges, [])
    for key, value in global_profiled_edge_dict.items():
        if key not in remain_da_edges:
            global_profiled_edge_dict[key] = 1

    print("global_profiled_edge_dict: ", global_profiled_edge_dict)
    edge_prof = os.path.join(pim_profile_file)
    with open(edge_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')
                ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
                op_name = item['args']['op_name']
                input_nodes =  item['args']['input_nodes']

                dur = item['dur']
                dma_size = 0
                if op_name == 'MemcpyFromHost':
                    dma_size = int(int(item['args']['activation_size'])/4)
                    da_edge = str(dma_size) + '0' + '2'
                    global_profiled_edge_cost[da_edge].append(dur)
                    print(op_name, "\t", da_edge, "\t", dur)
                elif op_name == "MemcpyToHost":
                    dma_size = int(int(item['args']['activation_size'])/2)
                    da_edge = str(dma_size) + '2' + '0'
                    global_profiled_edge_cost[da_edge].append(dur)
                    print(op_name, "\t", da_edge, "\t", dur)

    print("global_profiled_edge_cost")
    for key, value in global_profiled_edge_cost.items():
        print(key, "\t", value)

    return DCG, edge_dict, graph_idx_dict, pim_op_list, omp_op_list, global_profiled_edge_dict, global_profiled_edge_cost

def get_edge_cost(global_profiled_edge_cost, edge_profile_file):

    edge_prof = os.path.join(edge_profile_file)
    reduce_mean_size = 0
    with open(edge_prof, 'r') as f:
        data = json.load(f)
        for item in data:
            if (item['name'].find('_kernel_time') != -1):
                node_name = item['name'].replace('_kernel_time', '')
                print("\nCurrent node: ", node_name)
                ep_type = item['args']['provider'].replace('ExecutionProvider', '').lower()
                op_name = item['args']['op_name']
                input_nodes =  item['args']['input_nodes']
                incoming_nodes = item['args']['input_nodes'].split()
                print("input_nodes: ", input_nodes)
                if op_name == "ReduceMean":
                    reduce_mean_size = int(int(item['args']['activation_size']) / 4)
                dur = item['dur']
                dma_size = 0
                dma_flag = [s for s in incoming_nodes if "ReduceMean" in s]
                print(dma_flag)
                if op_name == 'MemcpyFromHost':

                    if dma_flag:
                        dma_size = reduce_mean_size
                    else:
                        dma_size = int(int(item['args']['activation_size']) / 4) 

                    da_edge = str(dma_size) + '1' + '2'
                    global_profiled_edge_cost[da_edge].append(dur)
                    print(op_name, "\t", da_edge, "\t", dur)
                elif op_name == "MemcpyToHost":

                    if dma_flag:
                        dma_size = reduce_mean_size
                    else:
                        dma_size = int(int(item['args']['activation_size']) / 2) 

                    da_edge = str(dma_size) + '2' + '1'
                    global_profiled_edge_cost[da_edge].append(dur)
                    print(op_name, "\t", da_edge, "\t", dur)

    da_edge_cost = dict.fromkeys(list(global_profiled_edge_cost.keys()), 0)
    for key, value in global_profiled_edge_cost.items():
        da_edge_cost[key] = median(value)


    return da_edge_cost

def get_optimal_partition(DCG, da_edge_cost, graph_idx_dict):
    ########################
    #### EDGE PROFILING
    ########################

    @dataclass
    class Cost:
        node_cost: int
        edge_cost: {}

    NUM_OF_DEVICES = 3
    NUM_OF_NODES = len(list(DCG.keys()))
    dev_list = ["cpu_s", "cpu_p", "pim"]

    schedule_graph = {}
    for node_name in list(DCG.keys()):
        print("Current node: ", node_name)
        schedule_graph[node_name] = [None] * NUM_OF_DEVICES
        for k in range(NUM_OF_DEVICES):
            schedule_graph[node_name][k] = Cost(0, {"cpu_s": 0, "cpu_p": 0, "pim": 0})
            schedule_graph[node_name][k].node_cost = DCG[node_name][k].cost
            print("Cost: ", DCG[node_name][k].cost)


    for node_name in DCG.keys():
        # print("node_name: ", node_name)
        for k in range(NUM_OF_DEVICES):
            if DCG[node_name][k].src_ptr:
                src_dnn, src_dev = DCG[node_name][k].src_ptr
                dma_size = DCG[src_dnn][src_dev].size
                distinct_edge = str(dma_size) + str(DCG[node_name][k].src_ptr[1]) + str(k)
                _edge_cost = da_edge_cost[distinct_edge]
                schedule_graph[node_name][k].edge_cost[dev_list[DCG[node_name][k].src_ptr[1]]] += _edge_cost
            if DCG[node_name][k].src_next_ptr:
                for item in DCG[node_name][k].src_next_ptr:
                    src_next_dnn, src_next_dev = item
                    dma_next_size = DCG[src_next_dnn][src_next_dev].size
                    distinct_next_edge = str(dma_next_size) + str(src_next_dev) + str(k)
                    _edge_cost = da_edge_cost[distinct_next_edge]
                    schedule_graph[node_name][k].edge_cost[dev_list[src_next_dev]] += _edge_cost

    print("DEBUG SCHEDULE GRAPH")
    for key, value in schedule_graph.items():
        print("key: ", key)
        for val in value:
            print("val: ", val)
    # DEVICE NO. cpu_serial: 0, cpu_parallel: 1, pim: 2
    node_list = list(DCG.keys())
    
    start_op = time.time()

    f_cpu_s = [0 for i in range(NUM_OF_NODES)]
    f_cpu_p = [0 for i in range(NUM_OF_NODES)]
    f_pim   = [0 for i in range(NUM_OF_NODES)]

    l_cpu_s = ['None' for i in range(NUM_OF_NODES)]
    l_cpu_p = ['None' for i in range(NUM_OF_NODES)]
    l_pim   = ['None' for i in range(NUM_OF_NODES)]

    f_cpu_s[0] = schedule_graph[node_list[0]][0].node_cost
    f_cpu_p[0] = schedule_graph[node_list[0]][1].node_cost
    f_pim[0]   = schedule_graph[node_list[0]][2].node_cost

    for j in range(1, NUM_OF_NODES):
        cur_node = node_list[j]
        print("\n===================================")
        print("Current node: ", cur_node)
        print("\n===================================")

        schedule_graph[cur_node][dev_list.index("cpu_s")].node_cost

        ### CPU_SERIAL
        cpu_s_cost = schedule_graph[cur_node][dev_list.index("cpu_s")].node_cost
        if f_cpu_s[j-1] <= f_cpu_p[j-1] + schedule_graph[cur_node][dev_list.index("cpu_s")].edge_cost["cpu_p"] and \
            f_cpu_s[j-1] <= f_pim[j-1] + schedule_graph[cur_node][dev_list.index("cpu_s")].edge_cost["pim"]:
            f_cpu_s[j] = f_cpu_s[j-1] + cpu_s_cost
            l_cpu_s[j] = "cpu_s"
        elif f_cpu_p[j-1] + schedule_graph[cur_node][dev_list.index("cpu_s")].edge_cost["cpu_p"] <= \
            f_cpu_s[j-1] <= f_pim[j-1] + schedule_graph[cur_node][dev_list.index("cpu_s")].edge_cost["pim"]:
            f_cpu_s[j] =  f_cpu_p[j-1] + schedule_graph[cur_node][dev_list.index("cpu_s")].edge_cost["cpu_p"] + cpu_s_cost
            l_cpu_s[j] = "cpu_p"
        else:
            f_cpu_s[j] = f_pim[j-1] + schedule_graph[cur_node][dev_list.index("cpu_s")].edge_cost["pim"] + cpu_s_cost
            l_cpu_s[j] = "pim"

        ### CPU_PARALLEL
        cpu_p_cost = schedule_graph[cur_node][dev_list.index("cpu_p")].node_cost
        if f_cpu_p[j-1] <= f_cpu_s[j-1] + schedule_graph[cur_node][dev_list.index("cpu_p")].edge_cost["cpu_s"] and \
            f_cpu_p[j-1] <= f_pim[j-1] + schedule_graph[cur_node][dev_list.index("cpu_p")].edge_cost["pim"]:
            f_cpu_p[j] = f_cpu_p[j-1] + cpu_p_cost
            l_cpu_p[j] = "cpu_p"
        elif f_cpu_s[j-1] + schedule_graph[cur_node][dev_list.index("cpu_p")].edge_cost["cpu_s"] <= \
            f_pim[j-1] + schedule_graph[cur_node][dev_list.index("cpu_p")].edge_cost["pim"]:
            f_cpu_p[j] = f_cpu_s[j-1] + schedule_graph[cur_node][dev_list.index("cpu_p")].edge_cost["cpu_s"] + cpu_p_cost
            l_cpu_p[j] = "cpu_s"
        else:
            f_cpu_p[j] = f_pim[j-1] + schedule_graph[cur_node][dev_list.index("cpu_p")].edge_cost["pim"] + cpu_p_cost
            l_cpu_p[j] = "pim"

        ### PIM
        pim_cost = schedule_graph[cur_node][dev_list.index("pim")].node_cost
        if f_pim[j-1] <= f_cpu_s[j-1] + schedule_graph[cur_node][dev_list.index("pim")].edge_cost["cpu_s"] and \
            f_pim[j-1] <= f_cpu_p[j-1] + schedule_graph[cur_node][dev_list.index("pim")].edge_cost["cpu_p"]:
            f_pim[j] = f_pim[j-1] + pim_cost
            l_pim[j] = "pim"
        elif f_cpu_s[j-1] + schedule_graph[cur_node][dev_list.index("pim")].edge_cost["cpu_s"] <= \
            f_cpu_p[j-1] + schedule_graph[cur_node][dev_list.index("pim")].edge_cost["cpu_p"]:
            f_pim[j] = f_cpu_s[j-1] + schedule_graph[cur_node][dev_list.index("pim")].edge_cost["cpu_s"] + pim_cost
            l_pim[j] = "cpu_s"
        else:
            f_pim[j] = f_cpu_p[j-1] + schedule_graph[cur_node][dev_list.index("pim")].edge_cost["cpu_p"] + pim_cost
            l_pim[j] = "cpu_p"

    values = [f_cpu_s[NUM_OF_NODES-1], f_cpu_p[NUM_OF_NODES-1] , f_pim[NUM_OF_NODES-1]]

    f_opt = min(values)
    l_opt = dev_list[values.index(f_opt)]

    print("f_opt: ", f_opt)

    end_op = time.time()
    print("ALS time: ", end_op - start_op)

    dnn_partition = {'CPUExecutionProvider' : [], 'PIMExecutionProvider': []}
    set_ep = lambda x : 'CPUExecutionProvider' if 'cpu' in x else ('PIMExecutionProvider' if x == 'pim' else 'WRG')
    l_dev = l_opt
    dnn_partition[set_ep(l_dev)].append(graph_idx_dict[node_list[NUM_OF_NODES-1]])
    l_print = lambda dev, idx : l_cpu_s[idx] if dev == 'cpu_s' else (l_cpu_p[idx] if dev == 'cpu_p' else l_pim[idx])

    cpu_serial_list = []

    for j in reversed(range(1, NUM_OF_NODES)):
        l_dev = l_print(l_dev, j)
        print("info: ", node_list[j-1], l_dev)
        if l_dev == "cpu_s":
            cpu_serial_list.append(graph_idx_dict[node_list[j-1]])

        dnn_partition[set_ep(l_dev)].append(graph_idx_dict[node_list[j-1]])

    for key, value in dnn_partition.items():
        value.sort(reverse=True)

    cpu_thread_map = {}
    cpu_thread_map["cpu_s"] = cpu_serial_list

    print("\n======================================")
    print("========== PARTITION RESULT ==========")
    print("======================================\n")

    for key, value in dnn_partition.items():
        for val in value:
            _node = list(graph_idx_dict.keys())[list(graph_idx_dict.values()).index(val)]
            if key == "CPUExecutionProvider":
                if val in cpu_serial_list:
                    print("Node: ", _node, "\t", "dev: ", "cpu_s")
                else:
                    print("Node: ", _node, "\t", "dev: ", "cpu_p")
            else:
                print("Node: ", _node, "\t", "dev: ", "pim")

    return dnn_partition, cpu_thread_map

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

def run_bfs(DCG, edge_dict, graph_idx_dict, pim_op_list, omp_op_list, global_profiled_edge_dict):
    dcg_node_list = list(DCG.keys())
    global_profiled_edge_list = []
    for key, value in global_profiled_edge_dict.items():
        global_profiled_edge_list.append(value)
    # print("global_profiled_edge_list: ", global_profiled_edge_list)
    da_edge_list = list(global_profiled_edge_dict.keys())
    print("da_edge_list: ", da_edge_list)
    for node_name in DCG.keys():
        for k in range(3):
            DCG[node_name][k].profiled_edge_list = [global_profiled_edge_list[:]]
            DCG[node_name][k].profiled_path = [[0 for i in range(len(list(DCG.keys())))]]

    ua_edge_list = []
    for key, values in edge_dict.items():
        value = values[0]
        attr = str(value[2]) + str(value[0]) + str(value[1])
        if attr not in ua_edge_list:
            ua_edge_list.append(attr)

    import numpy as np
    explored_edge_list_max = len(ua_edge_list)

    dnn_partition = {'cpu_s' : [], 'cpu_p' : [], 'pim' : []}
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
            print(DCG[dcg_node_list[node_idx]])

    while queue:
        # 1. Dequeue
        cur_node_idx, cur_dev_idx = queue.pop(0)
        cur_node_attrs = DCG[dcg_node_list[cur_node_idx]][cur_dev_idx]
        if cur_node_attrs.color == None:
            continue
        else:
            if (cur_node_idx, cur_dev_idx) not in visit:
                if cur_node_attrs.color != None:
                    print("\nCurrent node: ", dcg_node_list[cur_node_idx],"\tdevice: ", cur_dev_idx, "\n")
                    visit.append((cur_node_idx, cur_dev_idx))
                    # 2. Enqueue succ
                    if cur_node_idx < len(dcg_node_list) - 1:
                        succ_node_idx = cur_node_idx + 1
                        succ_node_list = []
                        for succ_dev_idx, succ_node in enumerate(DCG[dcg_node_list[succ_node_idx]]):
                            if succ_node.color != None:
                                if succ_node.color =='white':
                                    succ_node.color = 'black'
                                    succ_node_list.append((succ_node_idx, succ_dev_idx))
                        queue.extend(succ_node_list)
                    cur_node_name = dcg_node_list[cur_node_idx]
                    # print("\nCurrent node: ", cur_node_name)
                    # 3. Check edges from pred
                    pred_node_list = []
                    if cur_node_attrs.src_ptr != None:
                        # print("src_ptr")
                        pred_node_name, pred_dev_idx = DCG[dcg_node_list[cur_node_idx]][cur_dev_idx].src_ptr
                        # print("pred node: ", pred_node_name)
                        pred_node_idx = dcg_node_list.index(pred_node_name)
                        pred_node_list.append((pred_node_idx, pred_dev_idx))

                    if cur_node_attrs.src_next_ptr != None:
                        # print("src_next_ptr")
                        # len is one
                        for pred in cur_node_attrs.src_next_ptr:
                            (pred_node_name, pred_dev_idx) = pred
                            # print("pred node: ", pred_node_name)
                            pred_node_idx = dcg_node_list.index(pred_node_name)
                            pred_node_list.append((pred_node_idx, pred_dev_idx))
                    
                    if pred_node_list != []:
                        profiled_edge_list = []
                        profiled_path = []

                        for i in range(len(pred_node_list)):
                            pred_node_idx, pred_dev_idx = pred_node_list[i]
                            # print("Previous node: ", dcg_node_list[pred_node_idx], "\tdevice: ", pred_dev_idx)
                            pred_node_name = dcg_node_list[pred_node_idx]
                            dma_size = DCG[dcg_node_list[pred_node_idx]][pred_dev_idx].size
                            # da_edge = str(dma_size) + str(pred_dev_idx) + str(cur_dev_idx)
                            da_edge = (pred_dev_idx, cur_dev_idx, dma_size)
                            _da_edge = ((dcg_node_list[pred_node_idx], pred_dev_idx), (dcg_node_list[cur_node_idx], cur_dev_idx))
                            # print("da_edge: ", da_edge)
                            da_edge_idx = da_edge_list.index(da_edge)
                            # print("da_edge_idx: ", da_edge_idx)
                            new_profiled_edge_list = DCG[dcg_node_list[pred_node_idx]][pred_dev_idx].profiled_edge_list[:]
                            new_profiled_path_list = DCG[dcg_node_list[pred_node_idx]][pred_dev_idx].profiled_path[:]
                            # print("Prev edge list: ", new_profiled_edge_list)
                            for new_profiled_edge in new_profiled_edge_list:
                                if _da_edge in edge_dict.keys():
                                    if new_profiled_edge[da_edge_idx] == 0:
                                        # print("Found new edge")
                                        new_profiled_edge[da_edge_idx] = 1
                                        global_profiled_edge_list[da_edge_idx] = 1
                                else:
                                    continue
                            profiled_edge_list.extend(new_profiled_edge_list)
                            # print("New edge list : ", new_profiled_edge_list)
                            for new_profiled_path in new_profiled_path_list:
                                # new_profiled_path[pred_node_idx] = pred_dev_idx
                                new_profiled_path[cur_node_idx] = cur_dev_idx
                            profiled_path.extend(new_profiled_path_list)
                            # print("\n")

                        # print(profiled_edge_list)
                        if len(profiled_edge_list) == 1:
                            DCG[cur_node_name][cur_dev_idx].profiled_edge_list = [profiled_edge_list[0]]
                            DCG[cur_node_name][cur_dev_idx].profiled_path = [profiled_path[0]]
                        else:
                            set_list = []
                            for i in range(0, len(profiled_edge_list)):
                                set_index_vector = []
                                for j in range(0, len(global_profiled_edge_list)):
                                    if profiled_edge_list[i][j] == 1:
                                        set_index_vector.append(j)
                                set_list.append(set(set_index_vector))
                            # print("set_list: ", set_list)

                            cand = []
                            cand_set = set()
                            for i in range(0, len(set_list)):
                                tmp_cand = []
                                tmp_set = set()
                                for j in range(i+1, len(set_list)):
                                    if set_list[i].issuperset(set_list[j]):
                                        tmp_cand = [i]
                                        tmp_set = set_list[i]
                                    elif set_list[i].issubset(set_list[j]):
                                        tmp_cand = [j]
                                        tmp_set = set_list[j]
                                    else:
                                        tmp_cand = [i, j]
                                        tmp_set = set_list[i] & set_list[j]
                                if not cand:
                                    cand = tmp_cand
                                    cand_set = tmp_set
                                else:
                                    if cand_set.issuperset(tmp_set):
                                        continue
                                    elif cand_set.issubset(tmp_set):
                                        cand = tmp_cand
                                        cand_set = tmp_set                                        
                                    else:
                                        cand.extend(tmp_cand)
                                        cand_set.add(tmp_set)
                            print("cand_set: ", cand_set)

                            if len(cand) == 1:
                                DCG[cur_node_name][cur_dev_idx].profiled_edge_list = [profiled_edge_list[cand[0]][:]]
                                DCG[cur_node_name][cur_dev_idx].profiled_path = [profiled_path[cand[0]][:]]
                            else:
                                DCG[cur_node_name][cur_dev_idx].profiled_edge_list = []
                                DCG[cur_node_name][cur_dev_idx].profiled_path = []
                                for _cand in cand:
                                    DCG[cur_node_name][cur_dev_idx].profiled_edge_list.append(profiled_edge_list[_cand][:])
                                    DCG[cur_node_name][cur_dev_idx].profiled_path.append(profiled_path[_cand][:])

                    # pred is empty
                    else:
                        if cur_node_name not in pim_op_list and cur_node_name not in omp_op_list:
                            pred_node_idx = dcg_node_list.index(cur_node_name) - 1
                            print("pred_node_idx: ", pred_node_idx)
                            DCG[cur_node_name][cur_dev_idx].profiled_edge_list = DCG[dcg_node_list[pred_node_idx]][cur_dev_idx].profiled_edge_list[:]
                            DCG[cur_node_name][cur_dev_idx].profiled_path =      DCG[dcg_node_list[pred_node_idx]][cur_dev_idx].profiled_path[:]

                if len(DCG[cur_node_name][cur_dev_idx].profiled_edge_list) == 1:
                    res = all(ele == 1 for ele in DCG[cur_node_name][cur_dev_idx].profiled_edge_list[0])
                    if res:
                        print(DCG[cur_node_name][cur_dev_idx].profiled_path[0])
                        for dnn_idx, dev_idx in enumerate(DCG[cur_node_name][cur_dev_idx].profiled_path[0]):
                            if dev_idx == 2:
                                print("PIM : ", dcg_node_list[dnn_idx])
                                dnn_partition["pim"].append(graph_idx_dict[dcg_node_list[dnn_idx]])
                            elif dev_idx == 1:
                                print("CPU_P: ", dcg_node_list[dnn_idx])
                                dnn_partition["cpu_p"].append(graph_idx_dict[dcg_node_list[dnn_idx]])
                            else:
                                dnn_partition["cpu_s"].append(graph_idx_dict[dcg_node_list[dnn_idx]])
                        break
                    else:
                        continue
    # print(dnn_partition)
    end_bfs = time.time()
    print("BFS time: ", end_bfs - start_bfs)
    
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

    onnx.save(inferred_model, new_file_path)

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

def cpu_helper(inferred_model):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(inferred_model))
    graph = onnx_model.model.graph
    cpu_thread_map = {"elewise" : [], "matmul" : [], "gemm" : []}

    for node in graph.node:
        if node.op_type == "Add" or node.op_type == "Mul" or node.op_type == "Sub" or node.op_type == "Div":
            cpu_thread_map["elewise"].append(node.name)
        elif node.op_type == "MatMul":
            cpu_thread_map["matmul"].append(node.name)
        elif node.op_type == "Gemm":
            cpu_thread_map["gemm"].append(node.name)

    return cpu_thread_map

def run_cpu_serial(args, all_inputs):
    cpu_profile_file = run_profile("cpu_s", args.model, False, args.basic_optimization, 1, args.batch_size,
                               args.sequence_length, all_inputs, {}, {}, {})

    cpu_profile_records = load_profile_json(cpu_profile_file)

    return cpu_profile_file, cpu_profile_records

def run_cpu_parallel(args, all_inputs):
    cpu_profile_file = run_profile("cpu_p", args.model, False, args.basic_optimization, 4, args.batch_size,
                               args.sequence_length, all_inputs, {}, {}, {})

    cpu_profile_records = load_profile_json(cpu_profile_file)

    return cpu_profile_file, cpu_profile_records

def run_pim(args, all_inputs):
    align_map = pim_helper(args.model)

    pim_profile_file = run_profile("pim", args.model, True, args.basic_optimization, 1, args.batch_size,
                               args.sequence_length, all_inputs, align_map)

    pim_profile_records = load_profile_json(pim_profile_file)

    return pim_profile_file, pim_profile_records

def get_partition(args, all_inputs, cpu_serial_profile_file, cpu_parallel_profile_file, pim_profile_file):
    align_map = pim_helper(args.model)
    # cpu_thread_map = cpu_helper(args.model)

    DCG, edge_dict, graph_idx_dict, pim_op_list, omp_op_list, global_profiled_edge_dict, global_profiled_edge_cost = generate_dcg(cpu_serial_profile_file, cpu_parallel_profile_file, pim_profile_file)
    partition_map = run_bfs(DCG, edge_dict, graph_idx_dict, pim_op_list, omp_op_list, global_profiled_edge_dict)

    dnn_partition = {"CPUExecutionProvider" : [], "PIMExecutionProvider" : []}
    cpu_thread_map = {"cpu_s" : []}

    for key, value in partition_map.items():
        print(key, value)
        if key == "cpu_s" or key == "cpu_p":
            dnn_partition["CPUExecutionProvider"].extend(value)
            if key == "cpu_s":
                cpu_thread_map[key].extend(value)
        else:
            dnn_partition["PIMExecutionProvider"].extend(value)

    edge_profile_file = run_profile("edge", args.model, True, args.basic_optimization, args.thread_num, args.batch_size,
                                        args.sequence_length, all_inputs, align_map, dnn_partition, cpu_thread_map)

    da_edge_cost = get_edge_cost(global_profiled_edge_cost, edge_profile_file)

    opt_dnn_partition, cpu_thread_map = get_optimal_partition(DCG, da_edge_cost, graph_idx_dict)

    return opt_dnn_partition, cpu_thread_map

def run_opt(args, all_inputs, dnn_partition, cpu_thread_map):
    align_map = pim_helper(args.model)

    opt_profile_file = run_profile("opt", args.model, True, args.basic_optimization, args.thread_num, args.batch_size,
                                        args.sequence_length, all_inputs, align_map, dnn_partition, cpu_thread_map)

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
    # ####### CPU SERIAL
    # #####################
    print("RUNNING CPU SERIAL")

    cpu_serial_profile_file, cpu_serial_profile_records = run_cpu_serial(args, all_inputs)

    lines = ["CPU SEIRAL PROFILE RESULT\n"]
    lines.append("=" * 64)
    lines += parse_profile_results(cpu_serial_profile_records, args.kernel_time_only, args.threshold)

    lines.append("-" * 64)
    lines += group_profile_results(cpu_serial_profile_records, args.kernel_time_only, args.threshold)

    # #####################
    # ####### CPU PARALLEL
    # #####################
    print("RUNNING CPU")

    cpu_parallel_profile_file, cpu_parallel_profile_records = run_cpu_parallel(args, all_inputs)

    lines += ["CPU PARALLEL PROFILE RESULT\n"]
    lines.append("=" * 64)
    lines += parse_profile_results(cpu_parallel_profile_records, args.kernel_time_only, args.threshold)

    lines.append("-" * 64)
    lines += group_profile_results(cpu_parallel_profile_records, args.kernel_time_only, args.threshold)

    # #####################
    # ####### PIM
    # #####################
    print("RUNNING PIM")

    pim_profile_file, pim_profile_records = run_pim(args, all_inputs)
    lines += ["PIM PROFILE RESULT\n"]
    lines.append("=" * 64)
    lines += parse_profile_results(pim_profile_records, args.kernel_time_only, args.threshold)

    lines.append("-" * 64)
    lines += group_profile_results(pim_profile_records, args.kernel_time_only, args.threshold)


    # #####################
    # ####### EDGE
    # #####################
    print("RUNNING EDGE")

    dnn_partition, cpu_thread_map = get_partition(args, all_inputs, cpu_serial_profile_file, cpu_parallel_profile_file, pim_profile_file)
    
    import json
    with open("dnn_partition.json", "w") as partition_file:
        json.dump(dnn_partition, partition_file)
    with open("cpu_thread_map.json", "w") as cpu_thread_file:
        json.dump(cpu_thread_map, cpu_thread_file)
    # # #####################
    # # ####### OPT
    # # #####################

    # print("RUNNING OPT")
    # opt_profile_file, opt_profile_records = run_opt(args, all_inputs, dnn_partition, cpu_thread_map)
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

