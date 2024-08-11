from src.yolo.model import YOLOv1
from src.yolo.transform import *
from PIL import Image
from torch.nn.functional import softmax
# Model, loss function, and optimizer

class Inference:
    def __init__(self, model_path="dump/checkpoint/checkpoint_14.pt"):
        self.transform = ComposeTransforms([
            Resize((448, 448)),
            # RandomHorizontalFlip(),
            # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLOv1(num_classes=3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.objectness_threshold = 0.5


    def __call__(self, image_filename=""):
        self.load_image(image_filename)
        self.result = self.inference_yolo()
        return self


    def load_image(self, image_filename):
        self.image = Image.open(image_filename).convert("RGB")
        if self.transform:
            self.image, _ = self.transform(self.image, []) 
    
    def inference_yolo(self):
        self.image = self.image.to(self.device).unsqueeze(0)
        return self.model(self.image)
    

    def parse_yolo_result(self):
        self.bounding_boxes = []
        self.objectness_probability = []
        self.class_probabilities = []

        self.batch = self.result.size(0)
        self.S = self.result.size(1)
        for b in range(self.batch):
            for i in range(self.S):
                for j in range(self.S):
                    pred = self.result[b, i, j]
                    if pred[4] > self.objectness_threshold: 
                        self.bounding_boxes.append(self.yolo_xyxy(pred[:4], i, j))
                        self.objectness_probability.append(pred[4])
                        self.class_probabilities.append(softmax(pred[10:]))
                    
                    if pred[9] > self.objectness_threshold:
                        self.bounding_boxes.append(self.yolo_xyxy(pred[5:9], i, j))
                        self.objectness_probability.append(pred[9])
                        self.class_probabilities.append(softmax(pred[10:]))
        return self.bounding_boxes, self.objectness_probability, self.class_probabilities

    def yolo_xyxy(self, yolo_bbox, grid_x, grid_y):
        _, _, img_width, img_height = self.image.shape

        relative_x_center, relative_y_center, width, height = yolo_bbox
        x_center = (relative_x_center + grid_x) / self.S
        y_center = (relative_y_center + grid_y) / self.S

        x_center = int(x_center * img_width)
        y_center = int(y_center * img_height)
        width = int(width * img_width)
        height = int(height * img_height)
        return x_center, y_center, width, height


if __name__ == "__main__":
    image_path = "/home/user/YOLO/data/test_zip/test/apple_78.jpg"

    inference = Inference()
    inference = inference(image_path)

    bbox, conf, class_prob = inference.parse_yolo_result()
    print((bbox))
    print((conf))
    print((class_prob))
