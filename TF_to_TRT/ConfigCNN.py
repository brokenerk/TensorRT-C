import sys
import uff
import numpy as np
import tensorrt as trt
import graphsurgeon as gs

# initialize
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)


'''
    Nombre:     fix_tf_ssd_model
    Funcion:    Parcha las operaciones de TensorFlow incompatibles
                de un modelo de CNN para ser cargado en TensorRT
    Entradas:   Path del archivo frozen graph de la red (.pb) TF1
    Salidas:    Modelo de la CNN parchada de tipo de archivo gs.DynamicGraph
'''
def fix_tf_model(path):
    graph = gs.DynamicGraph(path)

    # Change FusedBatchNormV3
    FusedBatchNormV3 = graph.find_nodes_by_op('FusedBatchNormV3')
    if(len(FusedBatchNormV3) > 0):
        for node in FusedBatchNormV3:
            gs.update_node(node, op='FusedBatchNorm')

    # Change AddV2
    AddV2 = graph.find_nodes_by_op('AddV2')
    if(len(AddV2) > 0):
        for node in AddV2:
            gs.update_node(node, op='Add')
    
    # Remove NoOp
    NoOp = graph.find_nodes_by_op('NoOp')
    if(len(NoOp) > 0):
        for node in NoOp:
            gs.update_node(node, op='Identity')
    
        all_noop_nodes = graph.find_nodes_by_op("Identity")
        graph.forward_inputs(all_noop_nodes)

    return graph

'''
    Nombre:     convert_to_tensorrt
    Funcion:    Carga un modelo CNN en TensorRT y se guarda el BIN 
                del motor de inferencia (archivo para usar CNNs en TensorRT)
    Entradas:   Path del archivo frozen graph de la red (.pb) TF1   
    Salidas:    Ninguno, pero guarda el archivo BIN de la CNN para TensorRT
'''
def convert_to_tensorrt(path_model):
    dynamic_graph = fix_tf_model(path_model)

    #Obtiene tamaño inputs
    ancho = dynamic_graph.graph_inputs[0].attr['shape'].shape.dim[1].size
    alto = dynamic_graph.graph_inputs[0].attr['shape'].shape.dim[2].size
    canales = dynamic_graph.graph_inputs[0].attr['shape'].shape.dim[3].size

    #Obtiene nombre outputs
    outputs_names = []
    for output in dynamic_graph.graph_outputs:
        outputs_names.append(output.name)

    uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), outputs_names, output_filename='cnn_model.uff')
    TRTbin = "TRT_cnn_model.bin"

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        #Carga input
        input_name = dynamic_graph.graph_inputs[0].name
        parser.register_input(input_name, [canales, alto, ancho])

        #Carga outputs
        for output in outputs_names:
            parser.register_output(output)

        parser.parse('cnn_model.uff', network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(TRTbin, 'wb') as f:
            f.write(buf)


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Forma de uso: python3 ConfigCNN.py path_frozen_graph.pb")
        exit()

    print("Este script requiere de TensorFlow 1")
    path_model = sys.argv[1]
    convert_to_tensorrt(path_model)