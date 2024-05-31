import torch
import timm

my_model = timm.create_model('resnet50', pretrained=True, pretrained_cfg = {'file': '/scratch/wpy2004/vtab_weights/in1000.pth.tar'})
example_input = torch.randn(1, 3, 64, 64)
# torch.onnx.export(my_model, example_input, "in1000.onnx")

# import onnx
# from onnx_tf.backend import prepare

# onnx_model = onnx.load("in1000.onnx")
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph("in1000tf")

# import tensorflow as tf_rep
# tf_model = tf.saved_model.load("in1000tf")

breakpoint()
print()