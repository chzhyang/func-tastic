import time
from parliament import Context
from flask import Request, jsonify, make_response
import datetime
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import utils
import imagenet_preprocessing
import imghdr
import numpy as np
import json

MODEL_NAME = 'resnet50'
MODEL_PATH = 'models/resnet50_fp32_pretrained_model.pb'
INPUTS = 'input'
OUTPUTS = 'predict'
RESNET_IMAGE_SIZE = 224

TEST_INPUT_DATA = 'data/ILSVRC2012_test_00000181.JPEG'

# """Run standard ImageNet preprocessing on the passed image file.
  # Args:
  #   file_name: string, path to file containing a JPEG image
  #   output_height: int, final height of image
  #   output_width: int, final width of image
  #   num_channels: int, depth of input image
  # Returns:
  #   Float array representing processed image with shape
  #     [output_height, output_width, num_channels]
  # Raises:
  #   ValueError: if image is not a JPEG.
  # """

class image_classifier:
  """Evaluate image classifier with optimized TensorFlow graph"""
  batch_size = 1
  model_name = MODEL_NAME
  input_graph = ""
  data_location = ""
  results_file_path = "" # need define+time
  # optimize options
  num_inter_threads = 1 
  num_intra_threads = 36 # physical cores
  data_num_inter_threads = 32
  data_num_intra_threads = 14
  num_cores = 28

  def __init__(self, 
               batch_size, 
               model_name, 
               input_graph, 
               data_location, 
               num_inter_threads=1, 
               num_intra_threads=36):
    self.batch_size = batch_size
    self.model_name = model_name
    self.input_graph = input_graph
    self.data_location = data_location
    self.num_inter_threads = num_inter_threads
    self.num_intra_threads = num_intra_threads 
    self.calibrate = False


  # Write out the file name, expected label, and top prediction
  def write_results_output(self, predictions, filenames, labels):
    top_predictions = np.argmax(predictions, 1)
    with open(self.results_file_path, "a") as fp:
      for filename, expected_label, top_prediction in zip(filenames, labels, top_predictions):
        fp.write("{},{},{}\n".format(filename, expected_label, top_prediction))

  def optimize_graph(self):
    print("Optimize graph...")
    data_config = tf.compat.v1.ConfigProto()
    data_config.intra_op_parallelism_threads = self.data_num_intra_threads
    data_config.inter_op_parallelism_threads = self.data_num_inter_threads
    data_config.use_per_session_threads = 1

    infer_config = tf.compat.v1.ConfigProto()
    infer_config.intra_op_parallelism_threads = self.num_intra_threads
    infer_config.inter_op_parallelism_threads = self.num_inter_threads
    infer_config.use_per_session_threads = 1

    return data_config, infer_config

  def data_preprocess(self, file_name, batch_size, output_height, output_width,
                      num_channels=3):
    if imghdr.what(self.data_location) != "jpeg":
          raise ValueError("At this time, only JPEG images are supported. "
                        "Please try another image.")
    image_buffer = tf.io.read_file(file_name)
    image_array = imagenet_preprocessing.preprocess_image(image_buffer,None,output_height,output_width,num_channels,False)
    input_shape = [batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
    image_array_tensor = tf.reshape(image_array, input_shape)

    return image_array_tensor

  def run(self):
    """run inference"""
    # time result log
    load_model_time = 0.0
    optimize_model_time = 0.0
    coldstart_inference_time = 0.0
    time_average = 0.0
    throughput = 0.0

    data_config, infer_config = self.optimize_graph()

    print("Data processing...")
    data_graph = tf.Graph()
    with data_graph.as_default():
      if (self.data_location):
        print("Inference with real data.")
        images = self.data_preprocess(self.data_location, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
      else:
        print("Inference with dummy data.")
        input_shape = [self.batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
        images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')

    print("Load model...")
    load_model_start = time.time()
    infer_graph = tf.Graph()
    with infer_graph.as_default():
      graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v1.gfile.FastGFile(self.input_graph, 'rb') as input_file:
        input_graph_content = input_file.read()
        graph_def.ParseFromString(input_graph_content)
      # output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
        output_graph = graph_def
      tf.import_graph_def(output_graph, name='')

    load_model_end = time.time()
    load_model_time = load_model_end - load_model_start
    print("## Load model time: %10.3f ms" % (load_model_time * 1000))
    print("\tOptimize model time: %.3f ms" % (optimize_model_time * 1000))

    # Define input and output Tensors for detection_graph
    input_tensor = infer_graph.get_tensor_by_name('input:0')
    output_tensor = infer_graph.get_tensor_by_name('predict:0')

    data_sess = tf.compat.v1.Session(graph=data_graph,  config=data_config)
    infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)
    
    total_time = 0
    iteration = 0
    warm_up_iteration = 10
    while iteration < 50:
      iteration += 1
      start_time = time.time()
      image_np = data_sess.run(images)
      predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
      time_consume = time.time() - start_time
      print('Iteration %d: %.6f sec' % (iteration, time_consume))
      total_time = time_consume

      if iteration == 1:
        coldstart_inference_time = time_consume
      if iteration > warm_up_iteration:
        total_time += time_consume
      
    time_average = total_time / (iteration - warm_up_iteration)
    print('Average time: %.6f sec' % (time_average))

    return predictions, total_time.microseconds/1000, load_model_time, optimize_model_time, coldstart_inference_time, time_average, throughput

def request_handler(req: Request) -> str:
    if req.method == "GET":
      iterations = 3
      load_model_time = 0.0
      optimize_model_time = 0.0
      coldstart_inference_time = 0.0
      time_average = 0.0
      throughput = 0.0
      data = {}
      for i in range(iterations):
        graph = image_classifier(1,MODEL_NAME,MODEL_PATH,TEST_INPUT_DATA,1,36)
        predictions, latency, a, b, c, d, e = graph.run()

        load_model_time += a
        optimize_model_time += b
        coldstart_inference_time += c
        time_average += d
        throughput += e

        predictions_lables = utils.get_top_predictions(predictions, False, 5)
        data = {
          "top_predictions" : predictions_lables, 
          "inference_latency(ms)" : latency
        }
        # headers = { "content-type": "application/json" }
      return jsonify(data)
    elif req.method == "POST":
        print("request form: ", req.form)
        # print("request url: ", req.form.get('url'))
        # input_url = req.form.get('url')
        #  todo: download url to data
        # input_data = ""
        # graph = image_classifier_optimized_graph(1,MODEL_NAME,MODEL_PATH,input_data,1,36)
        # predictions, inference_latency = graph.run()
        # predictions_lables = utils.get_top_predictions(predictions, False, 5)
        # return {'top_predictions': predictions_lables, 'inference_latency': inference_latency}
        return "{}", 200
def main(context: Context):
    """ 
    Function template
    The context parameter contains the Flask request object and any
    CloudEvent received with the request.
    """
    # Add your business logic here
    print("Received request")
    print(tf.__version__)
    if 'request' in context.keys():
        return request_handler(context.request)
    else:
        print("Empty request", flush=True)
        return "{}", 400