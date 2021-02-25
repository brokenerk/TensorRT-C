import sys
import os
import uff
import numpy as np
import tensorrt as trt
import graphsurgeon as gs

output_name = ['NMS']
dims = [3,300,300]
layout = 7

# initialize
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)

'''
    Nombre:     fix_tf_ssd_model
    Funcion:    Parcha las operaciones de TensorFlow incompatibles
                de un modelo de CNN para ser cargado en TensorRT (SSD reentrenada)
    Entradas:   Path del archivo frozen graph de la red SSD reentrenada (.pb) TF1
    Salidas:    Modelo de la CNN parchada de tipo de archivo gs.DynamicGraph
'''
def fix_tf_ssd_model2(path):
    graph = gs.DynamicGraph(path)

    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape=[1, 3, 300, 300]
    )

    PriorBox = gs.create_plugin_node(
        name="GridAnchor",
        op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1], # Resolution 300
        #featureMapShapes=[29, 15, 8, 4, 2, 1], # Resolution 450
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=2,
        inputOrder=[0, 2, 1],
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        axis=1,
        ignoreBatch=0
    )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        axis=1,
        ignoreBatch=0
    )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "Cast": Input,
        "image_tensor": Input,
        "Concatenate": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    for node in graph.find_nodes_by_op('FusedBatchNormV3'):
        gs.update_node(node, op='FusedBatchNorm')

    graph.collapse_namespaces(namespace_plugin_map)
    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")

    # Create a constant Tensor and set it as input for GridAnchor_TRT
    data = np.array([1, 1], dtype=np.float32) 
    anchor_input = gs.create_node("AnchorInput", "Const", value=data)  
    graph.append(anchor_input)
    graph.find_nodes_by_op("GridAnchor_TRT")[0].input.insert(0, "AnchorInput")

    return graph

'''
    Nombre:     convert_to_tensorrt
    Funcion:    Carga un modelo CNN en TensorRT y se guarda el BIN 
                del motor de inferencia (archivo para usar CNNs en TensorRT)
    Entradas:   Path del archivo frozen graph de la red (.pb) TF1    
    Salidas:    Ninguno, pero guarda el archivo BIN de la CNN para TensorRT
'''
def convert_to_tensorrt(path_model):
    dynamic_graph = fix_tf_ssd_model2(path_model)
    uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), output_name, output_filename='ssd_reentrenada_model.uff')
    TRTbin = "TRT_ssd_reentrenada_model.bin"

    print("")
    for output in dynamic_graph.graph_outputs:
        print(output.name)
    print("")

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', dims)
        parser.register_output('NMS')
        parser.parse('ssd_reentrenada_model.uff', network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(TRTbin, 'wb') as f:
            f.write(buf)

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Forma de uso: python3 ConfigSSD2.py path_frozen_graph.pb")
        exit()

    #Parchar la SSD
    path_model = sys.argv[1]
    convert_to_tensorrt(path_model)