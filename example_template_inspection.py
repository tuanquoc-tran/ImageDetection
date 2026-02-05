"""
Example: Using Sample Teaching Templates in Inspection
Demonstrates integration of taught templates with inspection system
"""

import cv2
import json
import numpy as np
from pathlib import Path
from template_matcher import TemplateMatcher, AdaptiveTemplateMatcher
from roi_manager import ROI, ROIMode


class TemplateBasedInspector:
    """Inspector that uses templates from Sample Teaching"""
    
    def __init__(self, config_file: str):
        """
        Initialize inspector with product configuration
        
        Args:
            config_file: Path to JSON configuration from Sample Teaching
        """
        self.config = self.load_config(config_file)
        self.matcher = AdaptiveTemplateMatcher(match_threshold=0.75)
        self.load_templates()
        
    def load_config(self, config_file: str) -> dict:
        """Load product configuration"""
        with open(config_file, 'r') as f:
            return json.load(f)
            
    def load_templates(self):
        """Load all templates from configuration"""
        templates = self.config.get('templates', [])
        
        for template_data in templates:
            name = template_data['name']
            
            # Handle both old dict format and new ROI format
            if 'filename' in template_data.get('metadata', {}):
                filepath = template_data['metadata']['filename']
            else:
                filepath = template_data.get('filename', f"templates/{name}.png")
            
            if Path(filepath).exists():
                # Convert to ROI object for consistency
                roi = ROI.from_dict(template_data)
                
                self.matcher.load_template_from_file(
                    name=name,
                    filepath=filepath,
                    metadata=template_data
                )
                print(f"Loaded template: {name}")
            else:
                print(f"Warning: Template file not found: {filepath}")
                
    def inspect_with_template(self, image: np.ndarray, template_name: str) -> dict:
        """
        Inspect image using specific template
        
        Args:
            image: Input image
            template_name: Name of template to use
            
        Returns:
            Inspection result dictionary
        """
        # Get template metadata
        template_info = self.matcher.get_template_info(template_name)
        if not template_info:
            return {
                'status': 'error',
                'message': f'Template {template_name} not found'
            }
            
        metadata = template_info['metadata']
        
        # Extract ROI - handle both formats
        if isinstance(metadata, dict):
            roi_data = metadata.get('roi')
            if not roi_data:
                roi_data = {
                    'x': metadata.get('x', 0),
                    'y': metadata.get('y', 0),
                    'width': metadata.get('width', 100),
                    'height': metadata.get('height', 100)
                }
        else:
            roi_data = None
        
        # Perform template matching
        result = self.matcher.match_template_multiscale(
            image=image,
            template_name=template_name,
            roi=roi_data
        )
        
        return {
            'status': 'OK' if result.matched else 'NG',
            'matched': result.matched,
            'confidence': result.confidence,
            'location': result.location,
            'similarity': result.similarity_score,
            'defect_type': result.defect_type,
            'template': template_name
        }
        
    def inspect_all_templates(self, image: np.ndarray) -> dict:
        """
        Inspect image against all loaded templates
        
        Args:
            image: Input image
            
        Returns:
            Overall inspection result
        """
        results = []
        all_passed = True
        
        for template_name in self.matcher.list_templates():
            result = self.inspect_with_template(image, template_name)
            results.append(result)
            
            if not result.get('matched', False):
                all_passed = False
                
        return {
            'status': 'OK' if all_passed else 'NG',
            'overall_pass': all_passed,
            'template_results': results,
            'templates_checked': len(results)
        }
        
    def compare_roi_to_reference(self, image: np.ndarray, template_name: str) -> dict:
        """
        Compare ROI in image to reference template
        
        Args:
            image: Input image
            template_name: Reference template name
            
        Returns:
            Comparison result
        """
        template_info = self.matcher.get_template_info(template_name)
        if not template_info:
            return {'status': 'error', 'message': 'Template not found'}
            
        metadata = template_info['metadata']
        
        # Extract ROI - handle both formats
        if isinstance(metadata, dict):
            roi_data = metadata.get('roi')
            if not roi_data:
                roi_data = {
                    'x': metadata.get('x', 0),
                    'y': metadata.get('y', 0),
                    'width': metadata.get('width', 100),
                    'height': metadata.get('height', 100)
                }
        else:
            return {'status': 'error', 'message': 'No ROI defined for template'}
        
        # Extract and compare ROI
        result = self.matcher.compare_roi_to_template(
            image=image,
            roi=roi_data,
            template_name=template_name
        )
        
        return {
            'status': 'OK' if result.matched else 'NG',
            'matched': result.matched,
            'confidence': result.confidence,
            'similarity': result.similarity_score,
            'roi': roi_data,
            'defect_type': result.defect_type
        }
        
    def visualize_result(self, image: np.ndarray, inspection_result: dict) -> np.ndarray:
        """
        Visualize inspection result on image
        
        Args:
            image: Input image
            inspection_result: Result from inspect_with_template
            
        Returns:
            Image with visualization overlay
        """
        vis_image = image.copy()
        
        # Draw match location
        if inspection_result.get('location'):
            x, y = inspection_result['location']
            
            # Get template size
            template_name = inspection_result['template']
            template_info = self.matcher.get_template_info(template_name)
            if template_info:
                h, w = template_info['size']
                
                # Draw rectangle
                color = (0, 255, 0) if inspection_result['matched'] else (0, 0, 255)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence
                text = f"{inspection_result['confidence']:.2f}"
                cv2.putText(vis_image, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                           
        # Draw status
        status = inspection_result['status']
        status_color = (0, 255, 0) if status == 'OK' else (0, 0, 255)
        cv2.putText(vis_image, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                   
        return vis_image


def example_single_template_inspection():
    """Example: Inspect image with single template"""
    print("=== Single Template Inspection Example ===\n")
    
    # Initialize inspector
    inspector = TemplateBasedInspector('product_config.json')
    
    # Load inspection image
    image = cv2.imread('Samples/test_image.jpg')
    if image is None:
        print("Error: Could not load test image")
        return
        
    # Inspect with specific template
    result = inspector.inspect_with_template(image, 'connector_reference')
    
    print(f"Status: {result['status']}")
    print(f"Matched: {result['matched']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Location: {result['location']}")
    
    # Visualize
    vis_image = inspector.visualize_result(image, result)
    cv2.imwrite('inspection_result.jpg', vis_image)
    print("\nResult saved to inspection_result.jpg")


def example_multi_template_inspection():
    """Example: Inspect image with multiple templates"""
    print("\n=== Multi-Template Inspection Example ===\n")
    
    inspector = TemplateBasedInspector('product_config.json')
    
    # Load inspection image
    image = cv2.imread('Samples/test_image.jpg')
    if image is None:
        print("Error: Could not load test image")
        return
        
    # Inspect with all templates
    result = inspector.inspect_all_templates(image)
    
    print(f"Overall Status: {result['status']}")
    print(f"Templates Checked: {result['templates_checked']}")
    print(f"All Passed: {result['overall_pass']}\n")
    
    # Print individual results
    for i, template_result in enumerate(result['template_results'], 1):
        print(f"Template {i}: {template_result['template']}")
        print(f"  Status: {template_result['status']}")
        print(f"  Confidence: {template_result.get('confidence', 0):.3f}")
        print()


def example_roi_comparison():
    """Example: Compare specific ROI to reference"""
    print("\n=== ROI Comparison Example ===\n")
    
    inspector = TemplateBasedInspector('product_config.json')
    
    # Load inspection image
    image = cv2.imread('Samples/test_image.jpg')
    if image is None:
        print("Error: Could not load test image")
        return
        
    # Compare ROI
    result = inspector.compare_roi_to_reference(image, 'connector_reference')
    
    print(f"Status: {result['status']}")
    print(f"Matched: {result['matched']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"ROI: {result['roi']}")
    if result.get('defect_type'):
        print(f"Defect: {result['defect_type']}")


def example_batch_inspection():
    """Example: Batch process multiple images"""
    print("\n=== Batch Inspection Example ===\n")
    
    inspector = TemplateBasedInspector('product_config.json')
    
    # Get all images in Samples directory
    samples_dir = Path('Samples')
    if not samples_dir.exists():
        print("Samples directory not found")
        return
        
    image_files = list(samples_dir.glob('*.jpg')) + list(samples_dir.glob('*.png'))
    
    results_summary = {'OK': 0, 'NG': 0}
    
    for image_file in image_files:
        print(f"Processing: {image_file.name}")
        
        image = cv2.imread(str(image_file))
        if image is None:
            continue
            
        result = inspector.inspect_all_templates(image)
        
        status = result['status']
        results_summary[status] += 1
        
        print(f"  Result: {status}")
        
    print(f"\n=== Summary ===")
    print(f"Total: {len(image_files)}")
    print(f"OK: {results_summary['OK']}")
    print(f"NG: {results_summary['NG']}")
    print(f"Yield: {results_summary['OK']/len(image_files)*100:.1f}%")


def example_custom_matcher():
    """Example: Custom template matching parameters"""
    print("\n=== Custom Matcher Example ===\n")
    
    # Create custom matcher with different parameters
    matcher = TemplateMatcher(
        match_threshold=0.85,  # Higher threshold
        method=cv2.TM_CCOEFF_NORMED
    )
    
    # Load templates manually
    matcher.load_template_from_file(
        name='strict_reference',
        filepath='templates/reference.png',
        metadata={'threshold': 0.85, 'product': 'high_precision'}
    )
    
    # Load inspection image
    image = cv2.imread('Samples/test_image.jpg')
    if image is None:
        print("Error: Could not load test image")
        return
        
    # Match with custom parameters
    result = matcher.match_template(image, 'strict_reference')
    
    print(f"Matched: {result.matched}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Using threshold: 0.85 (strict)")


if __name__ == '__main__':
    """
    Run examples
    
    Prerequisites:
    1. Run vision_gui.py and create templates using Sample Teaching
    2. Save configuration as 'product_config.json'
    3. Place test images in 'Samples/' directory
    """
    
    print("Template-Based Inspection Examples")
    print("=" * 50)
    
    # Check if config exists
    if not Path('product_config.json').exists():
        print("\nError: product_config.json not found!")
        print("Please run vision_gui.py and create templates first.")
        print("Then save the configuration as 'product_config.json'")
        exit(1)
    
    try:
        # Run examples
        example_single_template_inspection()
        example_multi_template_inspection()
        example_roi_comparison()
        example_batch_inspection()
        example_custom_matcher()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()
