import torch
from models.progan import Generator
import os
import onnx
import onnxruntime as ort
import numpy as np

# https://pytorch.org/docs/stable/onnx.html
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

def export_as_onnx():

    resolution=64
    opset_version=12
    output_dirname="result/onnx"
    output_filename="progan_generator.onnx"
    check_exported_onnx=True
    verbose=False

    os.makedirs(output_dirname,exist_ok=True)
    g=Generator(resolution=resolution,start_resolution=resolution)


    dummy_input=torch.randn(5,512)
    g.eval()

    input_names = [ "input" ] 
    output_names = [ "output" ]

    os.makedirs('onnx',exist_ok=True)
    dst_path=os.path.join(output_dirname,output_filename)
    torch.onnx.export(g,dummy_input,dst_path,
        verbose=verbose,input_names=input_names,output_names=output_names,opset_version=opset_version,do_constant_folding=True,dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})


    if check_exported_onnx:
        
        # Load the ONNX model
        model = onnx.load(dst_path)

        # Check that the model is well formed
        onnx.checker.check_model(model)

        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))


        ort_session = ort.InferenceSession(dst_path)

        dummy_input=np.random.randn(2, 512).astype(np.float32)
        ort_outputs = ort_session.run(
            None,
            {"input": dummy_input},
        )

        torch_out = g(torch.from_numpy(dummy_input))
        np.testing.assert_allclose(torch_out.detach().numpy(),ort_outputs[0],rtol=1e-03,atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__=="__main__":
    export_as_onnx()