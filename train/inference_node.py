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



# SUBSCRIBER_TOPIC='/realsense_zr300/rgb/image_raw'
SUBSCRIBER_TOPIC='/realsense_d435/color/image_raw'
PROPERTY_TOPIC= 'inferencer/scalar_image_raw'
STD_TOPIC= 'inferencer/std_image_raw'



class NetworkInferencer:



  def __init__(self, net, use_cuda, args):
    # Save net
    self.net = net
    self.use_cuda = use_cuda
    self.args = args
    # Create ROS interfaces
    self.property_pub = rospy.Publisher(PROPERTY_TOPIC, Image, queue_size=2)
    if self.args.likelihood_loss:
      self.std_pub = rospy.Publisher(STD_TOPIC, Image, queue_size=2)
    self.subscriber = rospy.Subscriber(SUBSCRIBER_TOPIC, 
                                       Image, 
                                       self.callback, 
                                       queue_size=2)
    # Listen for messages.
    rospy.spin()



  def do_inference(self, input):

    with torch.no_grad():
      if use_cuda:
        input = input.cuda()

      # Do reshape, transpose and such.
      net_in = input.view(self.shape[0], self.shape[1], self.shape[2])  # Reshape to image dims
      net_in.transpose_(0,2)  # Move RGB to channel dimension
      net_in.transpose_(1,2)  # Swap height and width
      net_in.div_(255.0)  # Create from 255 white to 1.0 white

      output = self.net(net_in.unsqueeze(0))  # Unsqueeze to create batch dimension

    return output[0,:,:,:].cpu()



  def get_numpy_img(self, ros_img):
    self.shape = (ros_img.height, ros_img.width, 3)
    np_img = np.fromstring(ros_img.data, dtype=(np.uint8, 3))

    return np_img



  def publish_output(self, np_out, header):
    property_out = Image(encoding="32FC1")
    property_out.header = header
    property_out.height = self.shape[0]
    property_out.width = self.shape[1]
    if self.args.likelihood_loss:
      contig = np.ascontiguousarray(np_out[0])
      contig_std = np.ascontiguousarray(np_out[1])
    else:
      contig = np.ascontiguousarray(np_out)
    property_out.step = contig.strides[0]
    if self.args.likelihood_loss:
      std_out = property_out
      std_out.data = contig_std.tostring()
      self.std_pub.publish(std_out)
    property_out.data = contig.tostring()
    # Publish
    self.property_pub.publish(property_out)



  def callback(self, ros_img):
    # Convert image to numpy
    np_img = self.get_numpy_img(ros_img)
    # Convert to Tensor
    tensor_img = torch.Tensor(np_img)
    # Do the inference
    tensor_out = self.do_inference(tensor_img)
    # Do output conversion
    np_out = tensor_out.numpy()
    # Convert to ROS image
    self.publish_output(np_out, ros_img.header)



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
  if use_cuda:
    print('Using CUDA')

  # Get network.
  net = load_network(args, use_cuda)

  # Start ROS.
  rospy.init_node('inferencer')

  # Start inferencer.
  NetworkInferencer(net, use_cuda, args)

