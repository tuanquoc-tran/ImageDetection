"""
Main AOI Connector Inspection System
Orchestrates the complete inspection workflow
"""

import cv2
import numpy as np
import time
import os
from typing import Optional
from pathlib import Path

from config import InspectionConfig, DefectTypes, InspectionResult
from inspectors import InspectorFactory
from image_utils import VisualizationHelper, ImagePreprocessor


class ConnectorInspectionSystem:
    """Main inspection system class"""
    
    def __init__(self, config: InspectionConfig = None):
        """
        Initialize the inspection system
        
        Args:
            config: InspectionConfig object (uses default if None)
        """
        self.config = config if config else InspectionConfig()
        self.inspectors = InspectorFactory.create_all_inspectors(self.config)
        
        # Load reference images if configured
        InspectorFactory.load_reference_images(self.inspectors, self.config)
        
        # Create debug output directory if needed
        if self.config.SAVE_DEBUG_IMAGES:
            os.makedirs(self.config.DEBUG_IMAGE_PATH, exist_ok=True)
        
        print("AOI Connector Inspection System initialized")
        print(f"Target processing time: {self.config.TARGET_PROCESSING_TIME_MS} ms")
    
    def inspect(self, image: np.ndarray, image_id: str = "unknown") -> InspectionResult:
        """
        Perform complete inspection on an image
        
        Args:
            image: Input image (BGR format)
            image_id: Identifier for the image (for logging/debugging)
            
        Returns:
            InspectionResult object
        """
        start_time = time.time()
        result = InspectionResult()
        
        # Validate input
        if image is None or image.size == 0:
            result.add_defect("INVALID_IMAGE", "Input image is invalid")
            result.finalize()
            return result
        
        try:
            # Optional: Enhance image quality
            # enhanced_image = ImagePreprocessor.enhance_contrast(image)
            # For speed, we'll skip enhancement unless needed
            enhanced_image = image
            
            # Step 1: Check connector position (if enabled)
            if self.config.ENABLE_POSITION_CHECK:
                is_ok, offset, details = self.inspectors['connector_position'].inspect(
                    enhanced_image
                )
                result.details['position'] = details
                
                if not is_ok:
                    result.add_defect(
                        DefectTypes.POSITION_OFFSET,
                        f"Offset: ({offset[0]}, {offset[1]}) pixels"
                    )
            
            # Step 2: Check wire presence (if enabled)
            if self.config.ENABLE_WIRE_PRESENCE_CHECK:
                is_ok, missing_wires, details = self.inspectors['wire_presence'].inspect(
                    enhanced_image
                )
                result.details['wire_presence'] = details
                
                if not is_ok:
                    result.add_defect(
                        DefectTypes.MISSING_WIRE,
                        f"Missing wires at positions: {missing_wires}"
                    )
            
            # Step 3: Check wire order/color (if enabled and no missing wires)
            if self.config.ENABLE_WIRE_ORDER_CHECK:
                is_ok, incorrect_positions, details = self.inspectors['wire_order'].inspect(
                    enhanced_image
                )
                result.details['wire_order'] = details
                
                if not is_ok:
                    result.add_defect(
                        DefectTypes.WRONG_WIRE_ORDER,
                        f"Wrong color at positions: {incorrect_positions}"
                    )
            
            # Step 4: Check insertion depth (if enabled and no missing wires)
            if self.config.ENABLE_INSERTION_DEPTH_CHECK:
                is_ok, poorly_inserted, details = self.inspectors['wire_insertion'].inspect(
                    enhanced_image
                )
                result.details['insertion_depth'] = details
                
                if not is_ok:
                    result.add_defect(
                        DefectTypes.INSUFFICIENT_INSERTION,
                        f"Poor insertion at positions: {poorly_inserted}"
                    )
            
            # Finalize result
            result.finalize()
            
            # Calculate processing time
            end_time = time.time()
            result.processing_time_ms = (end_time - start_time) * 1000
            
            # Performance warning
            if result.processing_time_ms > self.config.TARGET_PROCESSING_TIME_MS:
                print(f"Warning: Processing time {result.processing_time_ms:.2f} ms exceeds target")
            
            # Save debug images if enabled
            if self.config.SAVE_DEBUG_IMAGES:
                self._save_debug_output(image, result, image_id)
            
            # Show visualization if enabled
            if self.config.SHOW_VISUALIZATION:
                self._show_results(image, result)
            
        except Exception as e:
            print(f"Error during inspection: {e}")
            result.add_defect("PROCESSING_ERROR", str(e))
            result.finalize()
        
        return result
    
    def inspect_from_file(self, image_path: str) -> InspectionResult:
        """
        Load image from file and perform inspection
        
        Args:
            image_path: Path to image file
            
        Returns:
            InspectionResult object
        """
        image = cv2.imread(image_path)
        if image is None:
            result = InspectionResult()
            result.add_defect("FILE_ERROR", f"Could not load image: {image_path}")
            result.finalize()
            return result
        
        image_id = Path(image_path).stem
        return self.inspect(image, image_id)
    
    def inspect_batch(self, image_paths: list) -> list:
        """
        Inspect multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of InspectionResult objects
        """
        results = []
        
        for i, path in enumerate(image_paths):
            print(f"\nInspecting image {i+1}/{len(image_paths)}: {path}")
            result = self.inspect_from_file(path)
            results.append(result)
            print(result)
        
        # Print summary
        ok_count = sum(1 for r in results if r.status == "OK")
        ng_count = len(results) - ok_count
        avg_time = np.mean([r.processing_time_ms for r in results])
        
        print(f"\n{'='*50}")
        print(f"Batch Inspection Summary:")
        print(f"Total: {len(results)} | OK: {ok_count} | NG: {ng_count}")
        print(f"Average processing time: {avg_time:.2f} ms")
        print(f"{'='*50}")
        
        return results
    
    def _save_debug_output(self, image: np.ndarray, result: InspectionResult, 
                          image_id: str):
        """
        Save debug visualization images
        
        Args:
            image: Original image
            result: Inspection result
            image_id: Image identifier
        """
        try:
            # Create result overlay
            overlay = VisualizationHelper.create_result_overlay(
                image, result, self.config
            )
            
            # Save overlay
            output_path = os.path.join(
                self.config.DEBUG_IMAGE_PATH,
                f"{image_id}_result.jpg"
            )
            cv2.imwrite(output_path, overlay)
            
        except Exception as e:
            print(f"Warning: Could not save debug output: {e}")
    
    def _show_results(self, image: np.ndarray, result: InspectionResult):
        """
        Display inspection results in a window
        
        Args:
            image: Original image
            result: Inspection result
        """
        try:
            overlay = VisualizationHelper.create_result_overlay(
                image, result, self.config
            )
            
            cv2.imshow("Inspection Result", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Warning: Could not display results: {e}")
    
    def calibrate_from_golden_sample(self, golden_image_path: str):
        """
        Calibrate system using a golden sample image
        
        Args:
            golden_image_path: Path to golden sample image
        """
        print(f"Calibrating from golden sample: {golden_image_path}")
        
        golden_image = cv2.imread(golden_image_path)
        if golden_image is None:
            print("Error: Could not load golden sample image")
            return
        
        # Set as reference template for position inspector
        from image_utils import ROIExtractor
        ref_roi = ROIExtractor.extract_roi(golden_image, self.config.CONNECTOR_ROI)
        self.inspectors['connector_position'].set_reference_template(ref_roi)
        
        print("Calibration complete")


def main():
    """Main entry point for standalone execution"""
    
    print("="*60)
    print("AOI Connector Inspection System")
    print("="*60)
    
    # Initialize system
    config = InspectionConfig()
    inspector = ConnectorInspectionSystem(config)
    
    # Example: Inspect a single image
    print("\nUsage examples:")
    print("1. Inspect single image:")
    print("   result = inspector.inspect_from_file('path/to/image.jpg')")
    print("")
    print("2. Inspect from camera/array:")
    print("   image = cv2.imread('image.jpg')")
    print("   result = inspector.inspect(image)")
    print("")
    print("3. Batch inspection:")
    print("   results = inspector.inspect_batch(['img1.jpg', 'img2.jpg'])")
    print("")
    print("4. Calibrate with golden sample:")
    print("   inspector.calibrate_from_golden_sample('golden.jpg')")
    
    # Check for sample images in Samples directory
    samples_dir = Path("Samples")
    if samples_dir.exists():
        sample_images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
        
        if sample_images:
            print(f"\nFound {len(sample_images)} sample images")
            print("Inspecting samples...")
            results = inspector.inspect_batch([str(p) for p in sample_images[:5]])  # First 5
        else:
            print("\nNo sample images found in Samples directory")
            print("Place test images in the Samples folder to run inspection")
    else:
        print("\nSamples directory not found")
        print("Create a 'Samples' folder and add test images to begin inspection")
    
    print("\nSystem ready for integration")


if __name__ == "__main__":
    main()
