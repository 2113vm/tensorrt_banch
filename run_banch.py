import pycuda.driver as cuda
from tqdm import tqdm
import pycuda.autoinit
import numpy as np
import tensorrt as trt
from time import time

import cv2
import torch
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import  ToTensorV2
from albumentations.augmentations.transforms import Normalize


BATCH_SIZE = 2


def postprocess(output_data):
    # get class names
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 0.1:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1


# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    # builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = BATCH_SIZE
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    
    config = builder.create_builder_config()
    # print(dir(config))
    config.max_workspace_size = 1 << 28
    serialized_engine = builder.build_serialized_network(network, config)
    with open("sample.engine", "wb") as f:
        f.write(serialized_engine)
        
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context


def main():

    from pathlib import Path
    from dataset import FlowerDataset
    from torch.utils.data import DataLoader

    dataset = FlowerDataset("./data/")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    ONNX_FILE_PATH = "./models/ResNet50.onnx"
    USE_TORCH = False
    
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH)
    # print(engine, dir(engine))

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
            

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    
    # preprocess input data

    times = []
    for num, batch in enumerate(tqdm(dataloader)):

        # let's imagine we have already had a tensor on GPU
        batch.to("cuda")

        start_time = time()

        if not USE_TORCH:
            # copy batch from cpu to gpu. device_input will be passed as tensorrt binding
            host_input = np.array(batch.cpu().numpy(), dtype=np.float32, order="C")
            cuda.memcpy_htod_async(device_input, host_input, stream)
        else:
            device_input = batch.cuda().contiguous().data_ptr()

        # run inference
        context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()

        output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, -1).to('cuda')

        times.append(time() - start_time)


    quantile_5 = np.quantile(times, 0.05)
    quantile_95 = np.quantile(times, 0.95)
    times = [t for t in times if quantile_5 <= t <= quantile_95]
    print(f"MEAN={np.mean(times)}, MEDIAN={np.median(times)}, STD={np.std(times)}")


if __name__ == "__main__":

    main()

