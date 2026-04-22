import argparse
import cv2
import logging
import os
from src.pipeline import ALPRPipeline
from src.utils import draw_results

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="ALPR Snapshot Inference (Raspberry Pi)")
    parser.add_argument('--image', type=str, required=True, help='Path to input image snapshot')
    parser.add_argument('--model', type=str, default='models/yolo26_custom.pt', help='Path to trained YOLO model')
    parser.add_argument('--output', type=str, default='output/result.jpg', help='Path to save annotated image')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        logging.error(f"Input image not found: {args.image}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize ALPR Pipeline
    pipeline = ALPRPipeline(yolo_model_path=args.model)
    
    # Process Snapshot
    logging.info(f"Processing snapshot: {args.image}")
    original_img, results = pipeline.process_image(args.image)
    
    if original_img is not None:
        plates_found = 0
        for res in results:
            if res['class_name'].lower() == 'plate':
                plates_found += 1
                logging.info(f"⭐ FOUND PLATE: '{res['text']}' (OCR Conf: {res['text_conf']:.2f}, Det Conf: {res['conf']:.2f})")
            else:
                logging.info(f"Detected {res['class_name']} (Det Conf: {res['conf']:.2f})")
                
        if plates_found == 0:
            logging.warning("No license plates detected in the image.")

        # Draw annotations and save
        output_img = draw_results(original_img, results)
        cv2.imwrite(args.output, output_img)
        logging.info(f"Annotated snapshot saved to {args.output}")

if __name__ == "__main__":
    main()
