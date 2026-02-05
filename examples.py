"""
Example usage and integration guide for the AOI Connector Inspection System
Demonstrates various use cases and integration patterns
"""

import cv2
import numpy as np
from main import ConnectorInspectionSystem
from config import InspectionConfig


def example_1_basic_inspection():
    """Example 1: Basic single image inspection"""
    print("\n" + "="*60)
    print("Example 1: Basic Single Image Inspection")
    print("="*60)
    
    # Initialize inspection system with default configuration
    inspector = ConnectorInspectionSystem()
    
    # Inspect an image from file
    result = inspector.inspect_from_file("Samples/connector_test.jpg")
    
    # Print results
    print(result)
    
    # Access specific result fields
    print(f"\nStatus: {result.status}")
    print(f"Defect Type: {result.defect_type}")
    print(f"Processing Time: {result.processing_time_ms:.2f} ms")
    
    if result.defects:
        print(f"Defects Found: {', '.join(result.defects)}")


def example_2_custom_configuration():
    """Example 2: Custom configuration for specific application"""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    config = InspectionConfig()
    
    # Adjust detection parameters
    config.POSITION_MAX_OFFSET_X = 20
    config.POSITION_MAX_OFFSET_Y = 20
    config.WIRE_PRESENCE_MIN_AREA = 600
    config.INSERTION_MIN_DEPTH = 250
    
    # Adjust wire configuration for different connector
    config.NUM_WIRES = 6
    config.WIRE_SLOT_SPACING = 150
    
    # Enable debug output
    config.SAVE_DEBUG_IMAGES = True
    config.DEBUG_IMAGE_PATH = "./debug_output/"
    
    # Initialize with custom config
    inspector = ConnectorInspectionSystem(config)
    
    # Inspect image
    result = inspector.inspect_from_file("Samples/connector_test.jpg")
    print(result)


def example_3_batch_processing():
    """Example 3: Batch processing multiple images"""
    print("\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    inspector = ConnectorInspectionSystem()
    
    # List of images to inspect
    image_paths = [
        "Samples/connector_001.jpg",
        "Samples/connector_002.jpg",
        "Samples/connector_003.jpg",
    ]
    
    # Batch inspection
    results = inspector.inspect_batch(image_paths)
    
    # Analyze results
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result.status}")
        if result.status == "NG":
            print(f"  Defects: {', '.join(result.defects)}")


def example_4_camera_integration():
    """Example 4: Integration with camera/frame grabber"""
    print("\n" + "="*60)
    print("Example 4: Camera Integration (Simulated)")
    print("="*60)
    
    inspector = ConnectorInspectionSystem()
    
    # Simulated camera frame capture
    # In real application, replace with actual camera API:
    # frame = camera.grab_frame()
    
    # For this example, load from file
    frame = cv2.imread("Samples/connector_test.jpg")
    
    if frame is not None:
        # Inspect the frame
        result = inspector.inspect(frame, image_id="camera_frame_001")
        
        print(f"Status: {result.status}")
        print(f"Processing Time: {result.processing_time_ms:.2f} ms")
        
        # In production, you might trigger actions based on result
        if result.status == "NG":
            print("Action: Reject part")
            # trigger_reject_mechanism()
        else:
            print("Action: Accept part")
            # trigger_accept_mechanism()


def example_5_golden_sample_calibration():
    """Example 5: Calibration with golden sample"""
    print("\n" + "="*60)
    print("Example 5: Golden Sample Calibration")
    print("="*60)
    
    inspector = ConnectorInspectionSystem()
    
    # Calibrate using a known good sample
    golden_sample_path = "Samples/golden_sample.jpg"
    inspector.calibrate_from_golden_sample(golden_sample_path)
    
    # Now inspect test images
    result = inspector.inspect_from_file("Samples/connector_test.jpg")
    print(result)


def example_6_selective_inspection():
    """Example 6: Selective inspection (only specific checks)"""
    print("\n" + "="*60)
    print("Example 6: Selective Inspection")
    print("="*60)
    
    # Configure to only check specific defects
    config = InspectionConfig()
    config.ENABLE_POSITION_CHECK = False  # Skip position check
    config.ENABLE_WIRE_ORDER_CHECK = False  # Skip color check
    config.ENABLE_WIRE_PRESENCE_CHECK = True  # Check presence
    config.ENABLE_INSERTION_DEPTH_CHECK = True  # Check insertion
    
    inspector = ConnectorInspectionSystem(config)
    
    result = inspector.inspect_from_file("Samples/connector_test.jpg")
    print(result)
    print("\nNote: Only wire presence and insertion depth were checked")


def example_7_detailed_analysis():
    """Example 7: Accessing detailed inspection data"""
    print("\n" + "="*60)
    print("Example 7: Detailed Analysis")
    print("="*60)
    
    inspector = ConnectorInspectionSystem()
    result = inspector.inspect_from_file("Samples/connector_test.jpg")
    
    # Access detailed information
    print(f"Status: {result.status}\n")
    
    if 'position' in result.details:
        pos_details = result.details['position']
        print(f"Position Check:")
        print(f"  Offset X: {pos_details['offset_x']} pixels")
        print(f"  Offset Y: {pos_details['offset_y']} pixels")
        print(f"  Match Score: {pos_details['match_score']:.3f}\n")
    
    if 'wire_presence' in result.details:
        presence_details = result.details['wire_presence']
        print(f"Wire Presence Check:")
        for i, present in enumerate(presence_details['wire_presence']):
            status = "OK" if present else "MISSING"
            print(f"  Wire {i+1}: {status}")
        print()
    
    if 'insertion_depth' in result.details:
        insertion_details = result.details['insertion_depth']
        print(f"Insertion Depth Check:")
        for i, depth in enumerate(insertion_details['insertion_depths']):
            print(f"  Wire {i+1}: {depth:.1f} pixels")


def example_8_plc_integration():
    """Example 8: Integration with PLC/industrial control system"""
    print("\n" + "="*60)
    print("Example 8: PLC Integration Pattern")
    print("="*60)
    
    inspector = ConnectorInspectionSystem()
    
    # Simulated PLC trigger
    print("Waiting for PLC trigger signal...")
    # In real application: while plc.wait_for_trigger():
    
    # Capture image
    # image = camera.grab_frame()
    image = cv2.imread("Samples/connector_test.jpg")
    
    # Inspect
    result = inspector.inspect(image, image_id="plc_trigger_001")
    
    # Send result to PLC
    print(f"\nSending to PLC:")
    print(f"  Result: {1 if result.status == 'OK' else 0}")
    print(f"  Defect Code: {result.defect_type}")
    
    # In real application:
    # plc.write_register("inspection_result", 1 if result.status == "OK" else 0)
    # plc.write_register("defect_code", encode_defect(result.defect_type))


def example_9_performance_optimization():
    """Example 9: Performance optimization tips"""
    print("\n" + "="*60)
    print("Example 9: Performance Optimization")
    print("="*60)
    
    # Configuration for maximum speed
    config = InspectionConfig()
    
    # Reduce ROI sizes if possible
    config.WIRE_SLOT_WIDTH = 150  # Smaller ROI = faster processing
    config.WIRE_SLOT_HEIGHT = 300
    
    # Adjust processing parameters
    config.GAUSSIAN_BLUR_KERNEL = (3, 3)  # Smaller kernel = faster
    
    # Disable non-critical checks if needed
    # config.ENABLE_WIRE_ORDER_CHECK = False
    
    # Disable debug output for production
    config.SAVE_DEBUG_IMAGES = False
    config.SHOW_VISUALIZATION = False
    
    inspector = ConnectorInspectionSystem(config)
    
    # Test performance
    import time
    num_iterations = 10
    
    print(f"Running {num_iterations} inspections for performance test...")
    
    image = cv2.imread("Samples/connector_test.jpg")
    times = []
    
    for i in range(num_iterations):
        result = inspector.inspect(image, f"perf_test_{i}")
        times.append(result.processing_time_ms)
    
    print(f"\nPerformance Results:")
    print(f"  Average: {np.mean(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")
    print(f"  Max: {np.max(times):.2f} ms")
    print(f"  Std Dev: {np.std(times):.2f} ms")


def example_10_error_handling():
    """Example 10: Proper error handling"""
    print("\n" + "="*60)
    print("Example 10: Error Handling")
    print("="*60)
    
    inspector = ConnectorInspectionSystem()
    
    try:
        # Attempt to inspect non-existent file
        result = inspector.inspect_from_file("non_existent_file.jpg")
        
        if result.status == "NG":
            print(f"Inspection failed: {result.defect_type}")
            if result.defects:
                print(f"Errors: {result.defects}")
        
    except Exception as e:
        print(f"Error during inspection: {e}")
    
    # Handling invalid image data
    try:
        invalid_image = np.zeros((0, 0), dtype=np.uint8)
        result = inspector.inspect(invalid_image)
        print(f"Result with invalid image: {result.status}")
    except Exception as e:
        print(f"Caught exception with invalid image: {e}")


def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("# AOI Connector Inspection System - Usage Examples")
    print("#"*60)
    
    examples = [
        ("Basic Inspection", example_1_basic_inspection),
        ("Custom Configuration", example_2_custom_configuration),
        ("Batch Processing", example_3_batch_processing),
        ("Camera Integration", example_4_camera_integration),
        ("Golden Sample Calibration", example_5_golden_sample_calibration),
        ("Selective Inspection", example_6_selective_inspection),
        ("Detailed Analysis", example_7_detailed_analysis),
        ("PLC Integration", example_8_plc_integration),
        ("Performance Optimization", example_9_performance_optimization),
        ("Error Handling", example_10_error_handling),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nTo run examples, modify this file or call functions directly")
    print("Example: python examples.py")
    
    # Uncomment to run specific examples:
    # example_1_basic_inspection()
    # example_9_performance_optimization()


if __name__ == "__main__":
    main()
