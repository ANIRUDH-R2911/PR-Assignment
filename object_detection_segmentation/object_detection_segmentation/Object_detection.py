import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np


class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image,'/camera/rgb/image_raw',self.image_callback,10)
        self.model = self.load_model('best.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info('Object Detection Node initialized.')

    def load_model(self, model_path):
        try:
            model = torch.load(model_path, map_location=self.device)
            self.get_logger().info(f'Model loaded from {model_path}')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

    def preprocess_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640, 640))  
        image_normalized = image_resized / 255.0  
        image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return image_tensor

    def draw_detections(self, image, predictions):
        for detection in predictions:
            x1, y1, x2, y2, conf, label = detection  
            if conf > 0.3:  
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Object Detection', image)
        cv2.waitKey(1)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_tensor = self.preprocess_image(cv_image)
            with torch.no_grad():
                output = self.model(input_tensor)[0] 
            predictions = self.postprocess_output(output)
            self.draw_detections(cv_image, predictions)
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    def postprocess_output(self, output):
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        predictions = []
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.3:  
                x1, y1, x2, y2 = box
                predictions.append([x1, y1, x2, y2, score, label])
        return predictions


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
