# Python Standard Stuff
import os
import importlib
from argparse import ArgumentParser
# PyTorch
import torch
# numpy
import numpy as np
# ROS
import roslib
import rospy
from sensor_msgs.msg import Image
# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError
# PIL
from PIL import Image as PILImage
from torchvision.transforms import ToTensor



SUBSCRIBER_TOPIC='/realsense_zr300/rgb/image_raw/'
PUBLISHER_TOPIC= 'output/scalar_image_raw'



class NetworkInferencer:

  def __init__(self, net, use_cuda):
    # Save net
    self.net = net
    self.use_cuda = use_cuda
    # Create ROS interfaces
    self.val_pub = rospy.Publisher(PUBLISHER_TOPIC, Image, queue_size=2)
    self.subscriber = rospy.Subscriber(SUBSCRIBER_TOPIC, 
                                       Image, 
                                       self.callback, 
                                       queue_size=2)
    # Create Cv Converter
    self.bridge = CvBridge()
    # Listen for messages.
    rospy.spin()

  def do_inference(self, input):
    if use_cuda:
      input = input.cuda()

    with torch.no_grad():
      output = self.net(input)
      output = output[0,0,:,:]

    if use_cuda:
      output = output.cpu()

    return output

  def get_numpy_img(self, ros_img):
    shape = (ros_img.height, ros_img.width, 3)
    np_img = np.fromstring(ros_img.data, dtype=(np.uint8, 3)).reshape(shape)
    np_img = np.transpose(np_img, (2, 1, 0))
    np_img = np_img.astype(np.float32)/255.0

    return np_img


  def callback(self, ros_img):
    # Convert image to numpy
    np_img = self.get_numpy_img(ros_img)
    # Convert to Tensor
    tensor_img = torch.Tensor(np_img)

    # Do the inference
    tensor_out = self.do_inference(tensor_img.unsqueeze(0))

    # Do output conversion
    np_out = tensor_out.numpy()
    np_out = np_out.transpose()
    # Convert to ROS image
    ros_out = Image(encoding="32FC1")
    ros_out.height = np_out.shape[0]
    ros_out.width = np_out.shape[1]
    contig = np.ascontiguousarray(np_out)
    ros_out.data = contig.tostring()
    ros_out.step = contig.strides[0]
    # Publish
    self.val_pub.publish(ros_out)
    






def load_network(args, use_cuda):
  # Import network.
  assert os.path.exists(args.model + ".py"), "Error: model definition not found"
  model_file = importlib.import_module(args.model)
  # Create network.
  net = model_file.Net(encoder=None, 
                       softmax_classes=args.force_n_classes, 
                       likelihood_loss=args.likelihood_loss)
  if use_cuda:
    net = torch.nn.DataParallel(net).cuda()
  # Load saved weights
  pretrained = torch.load(args.weights)['state_dict']
  net.load_state_dict(pretrained, strict=True)
  print('Successfully loaded network weights')

  return net



if __name__ == '__main__':
  # Read command line arguments
  parser = ArgumentParser()
  parser.add_argument('--model',type=str)
  parser.add_argument('--weights', type=str)
  parser.add_argument('--force-cuda', action='store_true', default=False)
  parser.add_argument('--force-n-classes', type=int, default=0)
  parser.add_argument('--likelihood-loss', action='store_true', default=False)
  args = parser.parse_args()

  # Enforce CUDA
  use_cuda = torch.cuda.is_available()
  if args.force_cuda:
    assert use_cuda, 'CUDA not available on system'

  # Get network.
  net = load_network(args, use_cuda)

  # Start ROS.
  rospy.init_node('inferencer')

  # Start inferencer.
  NetworkInferencer(net, use_cuda)

