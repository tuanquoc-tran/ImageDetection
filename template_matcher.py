"""
Template Matching Module for Industrial Vision Inspection
Handles template-based defect detection and comparison
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of template matching operation"""
    matched: bool
    confidence: float
    location: Optional[Tuple[int, int]] = None  # (x, y) of best match
    similarity_score: float = 0.0
    defect_type: Optional[str] = None
    

class TemplateMatcher:
    """Template matching engine for defect detection"""
    
    def __init__(self, match_threshold: float = 0.8, method=cv2.TM_CCOEFF_NORMED):
        """
        Initialize template matcher
        
        Args:
            match_threshold: Minimum confidence for a valid match (0-1)
            method: OpenCV template matching method
        """
        self.match_threshold = match_threshold
        self.method = method
        self.templates = {}
        
    def add_template(self, name: str, template_image: np.ndarray, metadata: Dict = None):
        """
        Add a template to the matcher
        
        Args:
            name: Template identifier
            template_image: Template image (BGR or grayscale)
            metadata: Optional metadata dictionary
        """
        self.templates[name] = {
            'image': template_image,
            'metadata': metadata or {},
            'size': template_image.shape[:2]
        }
        
    def load_template_from_file(self, name: str, filepath: str, metadata: Dict = None):
        """
        Load template from image file
        
        Args:
            name: Template identifier
            filepath: Path to template image
            metadata: Optional metadata dictionary
        """
        template_image = cv2.imread(filepath)
        if template_image is not None:
            self.add_template(name, template_image, metadata)
            return True
        return False
        
    def match_template(self, image: np.ndarray, template_name: str, 
                      roi: Optional[Dict] = None) -> MatchResult:
        """
        Match a template against an image
        
        Args:
            image: Input image
            template_name: Name of template to match
            roi: Optional ROI dictionary {'x', 'y', 'width', 'height'}
            
        Returns:
            MatchResult object
        """
        if template_name not in self.templates:
            return MatchResult(matched=False, confidence=0.0, 
                             defect_type="template_not_found")
            
        template_data = self.templates[template_name]
        template = template_data['image']
        
        # Extract ROI if specified
        if roi:
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            search_image = image[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            search_image = image
            offset_x, offset_y = 0, 0
            
        # Validate image size
        if (search_image.shape[0] < template.shape[0] or 
            search_image.shape[1] < template.shape[1]):
            return MatchResult(matched=False, confidence=0.0,
                             defect_type="image_too_small")
            
        # Perform template matching
        try:
            result = cv2.matchTemplate(search_image, template, self.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Get confidence based on method
            if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                confidence = 1.0 - min_val
                match_loc = min_loc
            else:
                confidence = max_val
                match_loc = max_loc
                
            # Adjust location for ROI offset
            absolute_loc = (match_loc[0] + offset_x, match_loc[1] + offset_y)
            
            # Determine if match is valid
            matched = confidence >= self.match_threshold
            
            return MatchResult(
                matched=matched,
                confidence=confidence,
                location=absolute_loc,
                similarity_score=confidence,
                defect_type=None if matched else "template_mismatch"
            )
            
        except cv2.error as e:
            return MatchResult(matched=False, confidence=0.0,
                             defect_type=f"cv_error: {str(e)}")
            
    def multi_match(self, image: np.ndarray, template_name: str,
                   threshold: float = None) -> List[Tuple[int, int, float]]:
        """
        Find multiple matches of a template in an image
        
        Args:
            image: Input image
            template_name: Name of template to match
            threshold: Match threshold (uses default if None)
            
        Returns:
            List of (x, y, confidence) tuples for all matches above threshold
        """
        if template_name not in self.templates:
            return []
            
        threshold = threshold or self.match_threshold
        template = self.templates[template_name]['image']
        
        # Perform template matching
        result = cv2.matchTemplate(image, template, self.method)
        
        # Find all matches above threshold
        locations = np.where(result >= threshold)
        matches = []
        
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], result[pt[1], pt[0]]))
            
        # Sort by confidence
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
        
    def compare_roi_to_template(self, image: np.ndarray, roi: Dict,
                                template_name: str) -> MatchResult:
        """
        Extract ROI from image and compare to template
        
        Args:
            image: Input image
            roi: ROI dictionary {'x', 'y', 'width', 'height'}
            template_name: Name of template to compare against
            
        Returns:
            MatchResult object
        """
        if template_name not in self.templates:
            return MatchResult(matched=False, confidence=0.0,
                             defect_type="template_not_found")
            
        # Extract ROI
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        roi_image = image[y:y+h, x:x+w]
        
        template = self.templates[template_name]['image']
        
        # Resize if dimensions don't match
        if roi_image.shape[:2] != template.shape[:2]:
            roi_image = cv2.resize(roi_image, (template.shape[1], template.shape[0]))
            
        # Calculate similarity using multiple metrics
        similarity_scores = []
        
        # Structural Similarity (SSIM)
        try:
            gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Normalize images
            gray_roi = cv2.normalize(gray_roi, None, 0, 255, cv2.NORM_MINMAX)
            gray_template = cv2.normalize(gray_template, None, 0, 255, cv2.NORM_MINMAX)
            
            # Calculate mean squared error
            mse = np.mean((gray_roi.astype(float) - gray_template.astype(float)) ** 2)
            max_pixel_value = 255.0
            if mse == 0:
                similarity = 1.0
            else:
                psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
                similarity = min(1.0, psnr / 50.0)  # Normalize to 0-1
                
            similarity_scores.append(similarity)
            
        except Exception:
            pass
            
        # Histogram comparison
        try:
            hist_roi = cv2.calcHist([roi_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist_template = cv2.calcHist([template], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            hist_roi = cv2.normalize(hist_roi, hist_roi).flatten()
            hist_template = cv2.normalize(hist_template, hist_template).flatten()
            
            hist_similarity = cv2.compareHist(hist_roi, hist_template, cv2.HISTCMP_CORREL)
            similarity_scores.append(max(0, hist_similarity))
            
        except Exception:
            pass
            
        # Average similarity scores
        if similarity_scores:
            avg_similarity = np.mean(similarity_scores)
        else:
            avg_similarity = 0.0
            
        matched = avg_similarity >= self.match_threshold
        
        return MatchResult(
            matched=matched,
            confidence=avg_similarity,
            location=(x, y),
            similarity_score=avg_similarity,
            defect_type=None if matched else "roi_mismatch"
        )
        
    def get_template_info(self, template_name: str) -> Optional[Dict]:
        """Get information about a template"""
        return self.templates.get(template_name)
        
    def list_templates(self) -> List[str]:
        """List all loaded template names"""
        return list(self.templates.keys())
        
    def remove_template(self, template_name: str) -> bool:
        """Remove a template from the matcher"""
        if template_name in self.templates:
            del self.templates[template_name]
            return True
        return False
        
    def clear_templates(self):
        """Remove all templates"""
        self.templates.clear()


class AdaptiveTemplateMatcher(TemplateMatcher):
    """
    Enhanced template matcher with adaptive threshold and scale invariance
    """
    
    def __init__(self, match_threshold: float = 0.8, 
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 scale_steps: int = 5):
        """
        Initialize adaptive matcher
        
        Args:
            match_threshold: Minimum confidence for match
            scale_range: (min_scale, max_scale) for multi-scale matching
            scale_steps: Number of scales to test
        """
        super().__init__(match_threshold)
        self.scale_range = scale_range
        self.scale_steps = scale_steps
        
    def match_template_multiscale(self, image: np.ndarray, template_name: str,
                                  roi: Optional[Dict] = None) -> MatchResult:
        """
        Match template at multiple scales
        
        Args:
            image: Input image
            template_name: Template to match
            roi: Optional ROI
            
        Returns:
            Best MatchResult across all scales
        """
        if template_name not in self.templates:
            return MatchResult(matched=False, confidence=0.0,
                             defect_type="template_not_found")
            
        template_data = self.templates[template_name]
        template = template_data['image']
        
        # Extract ROI if specified
        if roi:
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            search_image = image[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            search_image = image
            offset_x, offset_y = 0, 0
            
        best_result = MatchResult(matched=False, confidence=0.0)
        
        # Test multiple scales
        scales = np.linspace(self.scale_range[0], self.scale_range[1], self.scale_steps)
        
        for scale in scales:
            # Resize template
            scaled_w = int(template.shape[1] * scale)
            scaled_h = int(template.shape[0] * scale)
            
            if scaled_w <= 0 or scaled_h <= 0:
                continue
                
            if (scaled_w > search_image.shape[1] or 
                scaled_h > search_image.shape[0]):
                continue
                
            scaled_template = cv2.resize(template, (scaled_w, scaled_h))
            
            # Match at this scale
            try:
                result = cv2.matchTemplate(search_image, scaled_template, self.method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    confidence = 1.0 - min_val
                    match_loc = min_loc
                else:
                    confidence = max_val
                    match_loc = max_loc
                    
                # Update best result if this is better
                if confidence > best_result.confidence:
                    absolute_loc = (match_loc[0] + offset_x, match_loc[1] + offset_y)
                    best_result = MatchResult(
                        matched=confidence >= self.match_threshold,
                        confidence=confidence,
                        location=absolute_loc,
                        similarity_score=confidence,
                        defect_type=None if confidence >= self.match_threshold else "template_mismatch"
                    )
                    
            except cv2.error:
                continue
                
        return best_result
