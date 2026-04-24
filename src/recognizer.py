from paddleocr import PaddleOCR
import numpy as np
import logging


class PlateRecognizer:
    def __init__(self, lang='en'):
        """
        Initialize PaddleOCR 2.x for CPU inference on Raspberry Pi.

        Using paddlepaddle==2.6.2 + paddleocr==2.9.1 (pinned).
        PaddlePaddle 3.x introduced a Windows oneDNN C++ bug that crashes
        all CPU inference, so we stay on the stable 2.x line for both
        Windows development and Raspberry Pi deployment.
        """
        logging.info("Initializing PaddleOCR 2.x (use_gpu=False)...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,   # rotate text-line classification
            lang=lang,
            use_gpu=False,        # CPU inference — correct for Pi and Win dev
            show_log=False,       # suppress verbose paddle logs
        )
        logging.info("PaddleOCR initialized successfully.")

    def recognize(self, image_crop: np.ndarray):
        """
        Perform OCR on a cropped plate image using PaddleOCR 2.x API.

        PaddleOCR 2.x `.ocr()` returns:
            [  # outer list = one entry per image
                [  # inner list = one entry per detected text line
                    [box_coords, (text, confidence)]
                ]
            ]

        Returns:
            (text: str, avg_confidence: float)
        """
        if image_crop is None or image_crop.size == 0:
            return "", 0.0

        result = self.ocr.ocr(image_crop, cls=True)

        # Handle cases where nothing is detected
        if not result or not result[0]:
            return "", 0.0

        candidates = []
        # result[0] → lines for our single image
        for line in result[0]:
            box, (text, conf) = line
            
            # box is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            # Calculate the approximate height of the text bounding box
            h1 = abs(box[3][1] - box[0][1])
            h2 = abs(box[2][1] - box[1][1])
            height = (h1 + h2) / 2.0
            
            # Clean text: keep only alphanumeric characters (removes spaces, dashes)
            clean_text = "".join(c for c in str(text).upper() if c.isalnum())
            
            if clean_text:
                candidates.append({
                    "text": clean_text,
                    "conf": float(conf),
                    "height": height,
                    "y_center": (box[0][1] + box[2][1]) / 2.0,
                    "x_center": (box[0][0] + box[2][0]) / 2.0
                })

        if not candidates:
            return "", 0.0

        # Find the maximum text height to identify the main plate characters
        max_height = max(c["height"] for c in candidates)
        
        # Filter out smaller texts (e.g., "REGION IV-A", "MATATAG", dealership names)
        # We consider any text >= 40% of the max height as main plate text
        # (Lowered to 40% to ensure single isolated letters like 'Z' aren't accidentally dropped)
        main_texts = [c for c in candidates if c["height"] >= 0.4 * max_height]
        
        # Group into lines based on y_center to handle multi-line plates robustly
        main_texts.sort(key=lambda c: c["y_center"])
        lines = []
        current_line = []
        for c in main_texts:
            if not current_line:
                current_line.append(c)
            else:
                avg_y = sum(x["y_center"] for x in current_line) / len(current_line)
                if abs(c["y_center"] - avg_y) < max_height * 0.4:
                    current_line.append(c)
                else:
                    lines.append(current_line)
                    current_line = [c]
        if current_line:
            lines.append(current_line)
            
        # Sort each line left-to-right
        final_texts = []
        for line in lines:
            line.sort(key=lambda c: c["x_center"])
            final_texts.extend(line)

        final_text = "".join(c["text"] for c in final_texts)
        
        # Fix for old Philippine plates where the Rizal monument is detected as '1' or 'I'
        # e.g., 'ZKD1538' -> 'ZKD538'
        import re
        final_text = re.sub(r'^([A-Z]{3})[1I]([0-9]{3,4})$', r'\1\2', final_text)

        avg_conf = sum(c["conf"] for c in final_texts) / len(final_texts) if final_texts else 0.0

        return final_text, avg_conf
