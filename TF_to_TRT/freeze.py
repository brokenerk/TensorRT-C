import sys
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import signature_constants
from keras import backend as K
import numpy as np

'''
    Nombre:     freeze_tf2_model (TensorFlow 2)
    Funcion:    Convierte modelos de TF2 en frozen graphs de TF1 y lo guarda
    Entradas:   Path del archivo saved_model.pb de TF2
    Salidas:    Ninguno, pero genera y guarda un Frozen graph de TF1
'''
def freeze_tf2_model(path_model):
    saved_model_loaded = tf.saved_model.load(path_model)
    graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
    layers = [op.name for op in frozen_func.graph.get_operations()]
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./",
                    name="frozen_graph.pb",
                    as_text=False)
    print("Modelo convertido a TF1 y guardado con nombre: frozen_graph.pb")



if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Forma de uso: python3 freeze.py path_saved_model")
        exit()

    print("Este script requiere de TensorFlow 2")
    path_model = sys.argv[1]
    freeze_tf2_model(path_model)