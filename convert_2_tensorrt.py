import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


def build_engine(model_file, max_ws=512*1024*1024, fp16=False):
    print("building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # logging_manager = trt.LoggingManager()
    # debug_level = logging_manager.LEVEL.DEBUG
    # builder = trt.Builder(debug_level)
    builder = trt.Builder(TRT_LOGGER)
    # builder.fp16_enabled = True
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            #last_layer = network.get_layer(network.num_layers - 1)
            #network.mark_output(last_layer.get_output(0))
            engine = builder.build_engine(network, config=config)
            return engine
engine = build_engine("/home/hasan/Documents/exporting/mymodel.onnx")


with open('engine.trt', 'wb') as f:
    f.write(bytearray(engine.serialize()))
