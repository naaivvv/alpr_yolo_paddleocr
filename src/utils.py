import cv2
import numpy as np

def crop_image(image: np.ndarray, box: list) -> np.ndarray:
    """Crop the bounding box from the image safely."""
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    
    # Ensure coordinates are within image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    return image[y1:y2, x1:x2]

def draw_results(image: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes and OCR text on the image."""
    drawn_img = image.copy()
    
    for res in results:
        x1, y1, x2, y2 = map(int, res['box'])
        
        # Base label (e.g., 'car 0.95' or 'plate 0.88')
        label = f"{res['class_name']} {res['conf']:.2f}"
        
        # Color coding: Green for plates, Blue for vehicles
        color = (0, 255, 0) if res['class_name'].lower() == 'plate' else (255, 0, 0)
        
        # Append OCR text if it's a plate and text was found
        if res.get('text'):
            label += f" | {res['text']} ({res['text_conf']:.2f})"
        
        # Draw bounding box
        cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background for better readability
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(drawn_img, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(drawn_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return drawn_img
