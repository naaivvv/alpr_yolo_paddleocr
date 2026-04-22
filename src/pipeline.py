from ultralytics import YOLO
import cv2
import logging
from src.utils import crop_image
from src.recognizer import PlateRecognizer

class ALPRPipeline:
    def __init__(self, yolo_model_path: str):
        """
        Initialize the full ALPR Pipeline (YOLO26 Detection + PaddleOCR Recognition).
        """
        logging.info(f"Loading YOLO model from {yolo_model_path}...")
        self.detector = YOLO(yolo_model_path)
        self.names = self.detector.names
        logging.info(f"YOLO classes loaded: {self.names}")
        
        self.recognizer = PlateRecognizer(lang='en')

    def process_image(self, image_path: str):
        """
        Process a single snapshot:
        1. YOLO detects cars and plates.
        2. If a plate is found, it is cropped.
        3. PaddleOCR reads the text from the cropped plate.
        """
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            return None, []

        # Run YOLO inference
        # imgsz=640 is standard, conf=0.25 filters out weak detections
        results = self.detector(image, imgsz=640, conf=0.25, verbose=False)[0]
        
        final_results = []
        
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_name = self.names[cls_id]
            
            det_info = {
                'box': coords.tolist(),
                'class_name': cls_name,
                'class_id': cls_id,
                'conf': conf,
                'text': "",
                'text_conf': 0.0
            }
            
            # Perform OCR only on the 'plate' class
            if cls_name.lower() == 'plate':
                plate_crop = crop_image(image, coords)
                text, text_conf = self.recognizer.recognize(plate_crop)
                det_info['text'] = text
                det_info['text_conf'] = text_conf
                
            final_results.append(det_info)
            
        return image, final_results
