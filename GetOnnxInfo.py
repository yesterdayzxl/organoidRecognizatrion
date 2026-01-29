import onnx

def get_input_output_names(model_path):
    model = onnx.load(model_path)
    input_names = [input.name for input in model.graph.input]
    output_names = [output.name for output in model.graph.output]
    return input_names, output_names

if __name__ == '__main__':
    model_path = 'G:\\活细胞成像\\Project\\runs\\segment\\train26\\weights\\best.onnx'
    input_names, output_names = get_input_output_names(model_path)

    print("Input names:", input_names)
    print("Output names:", output_names)