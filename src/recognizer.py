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

        texts = []
        confs = []

        # result[0] → lines for our single image
        for line in result[0]:
            box, (text, conf) = line
            clean = str(text).strip().upper()
            if clean:
                texts.append(clean)
                confs.append(float(conf))

        final_text = " ".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return final_text, avg_conf
