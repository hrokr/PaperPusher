"""
Image preprocessing module for OCR quality improvement.
Based on techniques from the Medium article about OCR preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing to improve OCR accuracy."""

    def __init__(self):
        self.temp_dir = Path("/tmp/ocr_preprocessing")
        self.temp_dir.mkdir(exist_ok=True)

    def preprocess_image(self, image_path: str, output_path: str) -> bool:
        """
        Apply preprocessing techniques to improve OCR quality.

        Args:
            image_path: Path to input image
            output_path: Path for processed image

        Returns:
            bool: Success status
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return False

            # Apply preprocessing pipeline
            processed_img = self._preprocessing_pipeline(img)

            # Save processed image
            success = cv2.imwrite(output_path, processed_img)
            if success:
                logger.info(f"Preprocessed image saved to: {output_path}")
                return True
            else:
                logger.error(f"Failed to save processed image: {output_path}")
                return False

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return False

    def _preprocessing_pipeline(self, img: np.ndarray) -> np.ndarray:
        """Apply the complete preprocessing pipeline."""

        # 1. Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 2. Noise reduction
        denoised = self._denoise_image(gray)

        # 3. Contrast enhancement
        enhanced = self._enhance_contrast(denoised)

        # 4. Skew correction
        corrected = self._correct_skew(enhanced)

        # 5. Binarization (convert to black and white)
        binary = self._binarize_image(corrected)

        # 6. Morphological operations to clean up
        cleaned = self._morphological_cleanup(binary)

        return cleaned

    def _denoise_image(self, img: np.ndarray) -> np.ndarray:
        """Remove noise from the image."""
        # Non-local means denoising works well for scanned documents
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _correct_skew(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct skew in the image."""
        try:
            # Find edges
            edges = cv2.Canny(img, 50, 150, apertureSize=3)

            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:20]:  # Use first 20 lines
                    angle = theta * 180 / np.pi
                    # Convert to angle from horizontal
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)

                if angles:
                    # Use median angle for robustness
                    skew_angle = np.median(angles)

                    # Only correct if angle is significant
                    if abs(skew_angle) > 0.5:
                        return self._rotate_image(img, -skew_angle)

            return img

        except Exception as e:
            logger.warning(f"Skew correction failed: {str(e)}")
            return img

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        height, width = img.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))

        # Adjust translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform rotation
        return cv2.warpAffine(
            img,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _binarize_image(self, img: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white)."""
        # Otsu's thresholding automatically finds optimal threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _morphological_cleanup(self, img: np.ndarray) -> np.ndarray:
        """Clean up binary image using morphological operations."""
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # Remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    def preprocess_pdf_pages(self, pdf_path: str, output_dir: str) -> list:
        """
        Extract pages from PDF and preprocess them.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save preprocessed images

        Returns:
            list: Paths to preprocessed images
        """
        try:
            import fitz  # PyMuPDF - add to requirements if using this

            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

            doc = fitz.open(pdf_path)
            preprocessed_pages = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Convert to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # Save temporary image
                temp_img_path = self.temp_dir / f"page_{page_num}.png"
                with open(temp_img_path, "wb") as f:
                    f.write(img_data)

                # Preprocess
                processed_path = output_dir / f"processed_page_{page_num}.png"
                if self.preprocess_image(str(temp_img_path), str(processed_path)):
                    preprocessed_pages.append(str(processed_path))

                # Clean up temp file
                temp_img_path.unlink(missing_ok=True)

            doc.close()
            return preprocessed_pages

        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
            return []
        except Exception as e:
            logger.error(f"Error preprocessing PDF pages: {str(e)}")
            return []


def main():
    """Example usage of the preprocessor."""
    preprocessor = ImagePreprocessor()

    # Example: preprocess a single image
    # success = preprocessor.preprocess_image("input.jpg", "output.jpg")

    # Example: preprocess PDF pages
    # pages = preprocessor.preprocess_pdf_pages("document.pdf", "output_dir")

    pass


if __name__ == "__main__":
    main()
