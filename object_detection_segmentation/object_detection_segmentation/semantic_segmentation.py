import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import tensorflow as tf
import numpy as np


class SemanticSegmentation(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image,'/camera/rgb/image_raw',self.image_callback,10)
        self.model = self.load_model('best_model.keras')
        self.model_input_size = (256, 256) 

        self.get_logger().info('Semantic Segmentation Node initialized.')

    def load_model(self, model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            self.get_logger().info(f'Model loaded from {model_path}')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

    def preprocess_image(self, image):
        image_resized = cv2.resize(image, self.model_input_size)
        image_normalized = image_resized / 255.0
        return np.expand_dims(image_normalized, axis=0)

    def postprocess_segmentation(self, segmentation_map, original_image_shape):
        segmentation_map_resized = cv2.resize(segmentation_map, (original_image_shape[1], original_image_shape[0]))
        segmentation_map_colored = cv2.applyColorMap((segmentation_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return segmentation_map_colored

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_tensor = self.preprocess_image(cv_image)
            segmentation_map = self.model.predict(input_tensor)[0]
            segmentation_map = np.argmax(segmentation_map, axis=-1)
            segmentation_map_colored = self.postprocess_segmentation(segmentation_map, cv_image.shape)
            overlay = cv2.addWeighted(cv_image, 0.6, segmentation_map_colored, 0.4, 0)
            cv2.imshow('Semantic Segmentation', overlay)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
