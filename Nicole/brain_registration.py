import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
import cv2
import matplotlib.pyplot as plt
import h5py
import json
from scipy.spatial.distance import cdist
from skimage import measure
from typing import Tuple, Optional
from pipeline_utils import normalize_arr  # Assuming this is a utility function for normalization

class BrainHemisphereRegistration:
    def __init__(self, sample_mask: np.ndarray, reference_mask: np.ndarray):
        """Initialize registration with sample and reference (binary) masks."""
        self.sample_mask = sample_mask
        self.reference_mask = reference_mask

        # Extract boundary points
        self.sample_boundary = self._extract_boundary_points(self.sample_mask)
        self.reference_boundary = self._extract_boundary_points(self.reference_mask)
        
        # Subsample for efficiency while maintaining shape characteristics
        self.sample_points = self._subsample_points(self.sample_boundary, max_points=500)
        self.reference_points = self._subsample_points(self.reference_boundary, max_points=500)
        
        print(f"Sample boundary points: {len(self.sample_points)}")
        print(f"Reference boundary points: {len(self.reference_points)}")
    
    def _extract_boundary_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary points from binary mask."""
        # Use skimage for more robust contour detection
        contours = measure.find_contours(mask, 0.5)
        
        # Take the longest contour (main boundary)
        if len(contours) == 0:
            raise ValueError("No contours found in mask")
        
        main_contour = max(contours, key=len)
        
        # Convert to (x, y) format
        boundary_points = np.column_stack([main_contour[:, 1], main_contour[:, 0]])
        
        return boundary_points
    
    def _subsample_points(self, points: np.ndarray, max_points: int = 500) -> np.ndarray:
        """Subsample points to maintain shape characteristics."""
        if len(points) <= max_points:
            return points
        
        # Calculate curvature to identify important points
        curvature = self._calculate_curvature(points)
        
        # Combine uniform sampling with curvature-based sampling
        n_uniform = max_points // 2
        n_curvature = max_points - n_uniform
        
        # Uniform sampling
        uniform_indices = np.linspace(0, len(points)-1, n_uniform, dtype=int)
        uniform_points = points[uniform_indices]
        
        # High-curvature sampling
        curvature_indices = np.argsort(curvature)[-n_curvature:]
        curvature_points = points[curvature_indices]
        
        # Combine and remove duplicates
        all_points = np.vstack([uniform_points, curvature_points])
        unique_points = np.unique(all_points, axis=0)
        
        return unique_points
    
    def _calculate_curvature(self, points: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate discrete curvature at each point."""
        n = len(points)
        curvature = np.zeros(n)
        
        for i in range(n):
            # Get neighboring points
            prev_idx = (i - window_size) % n
            next_idx = (i + window_size) % n
            
            # Calculate vectors
            v1 = points[i] - points[prev_idx]
            v2 = points[next_idx] - points[i]
            
            # Calculate angle change (curvature approximation)
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                curvature[i] = np.abs(np.arccos(cos_angle))
        
        return curvature
    
    def _affine_transform_points(self, points: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply affine transformation to points."""
        # Extract transformation parameters
        # params = [a, b, c, d, tx, ty] where:
        # x' = a*x + b*y + tx
        # y' = c*x + d*y + ty
        a, b, c, d, tx, ty = params
        
        # Create transformation matrix
        transform_matrix = np.array([[a, b, tx],
                                   [c, d, ty],
                                   [0, 0, 1]])
        
        # Convert points to homogeneous coordinates
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        
        # Apply transformation
        transformed = (transform_matrix @ homogeneous_points.T).T
        
        return transformed[:, :2]
    
    def _cost_function(self, params: np.ndarray) -> float:
        """Cost function for optimization; combines multiple distance metrics for robust registration."""
        # Transform sample points
        transformed_sample = self._affine_transform_points(self.sample_points, params)
        
        # Calculate bidirectional distance
        # Distance from transformed sample to reference
        dist_matrix_1 = cdist(transformed_sample, self.reference_points)
        min_distances_1 = np.min(dist_matrix_1, axis=1)
        
        # Distance from reference to transformed sample
        dist_matrix_2 = cdist(self.reference_points, transformed_sample)
        min_distances_2 = np.min(dist_matrix_2, axis=1)
        
        # Combine distances
        cost = np.mean(min_distances_1) + np.mean(min_distances_2)
        
        # Add regularization to prevent extreme transformations
        det = params[0] * params[3] - params[1] * params[2]  # determinant
        if det <= 0.01:  # Prevent degenerate transformations
            cost += 1000
        
        # Penalize extreme scaling
        scale_penalty = np.abs(det - 1.0) * 10
        cost += scale_penalty
        
        return cost
    
    def register(self, max_iterations: int = 1000,n_restarts: int = 5, verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Perform registration using multiple optimization restarts.

        Returns:
            best_transform_matrix: 3x3 transformation matrix
            results: Dictionary with optimization results
        """
        best_cost = float('inf')
        best_params = None
        all_results = []
        
        for restart in range(n_restarts):
            if verbose:
                print(f"Optimization restart {restart + 1}/{n_restarts}")
            
            # Initialize with different starting points
            if restart == 0:
                # Start with identity + small perturbation
                initial_params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                initial_params += np.random.normal(0, 0.1, 6)
            else:
                # Random initialization around reasonable values
                scale_range = (0.5, 2.0)
                rotation_range = (-np.pi/4, np.pi/4)
                
                scale = np.random.uniform(*scale_range)
                rotation = np.random.uniform(*rotation_range)
                
                # Create random affine transformation
                cos_r, sin_r = np.cos(rotation), np.sin(rotation)
                initial_params = np.array([
                    scale * cos_r, -scale * sin_r,
                    scale * sin_r, scale * cos_r,
                    np.random.uniform(-50, 50),  # translation
                    np.random.uniform(-50, 50)
                ])
            
            # Optimization with bounds to prevent extreme transformations
            bounds = [
                (-3, 3),   # a: scaling/rotation component
                (-3, 3),   # b: shear/rotation component  
                (-3, 3),   # c: shear/rotation component
                (-3, 3),   # d: scaling/rotation component
                (-200, 200),  # tx: translation x
                (-200, 200)   # ty: translation y
            ]
            
            try:
                result = minimize(
                    self._cost_function,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iterations, 'disp': False}
                )
                
                all_results.append(result)
                
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_params = result.x
                    
                if verbose:
                    print(f"  Cost: {result.fun:.4f}, Success: {result.success}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Optimization failed: {e}")
                continue
        
        if best_params is None:
            raise RuntimeError("All optimization attempts failed")
        
        # Convert best parameters to transformation matrix
        a, b, c, d, tx, ty = best_params
        transform_matrix = np.array([[a, b, tx],
                                   [c, d, ty],
                                   [0, 0, 1]])
        
        results = {
            'best_cost': best_cost,
            'best_params': best_params,
            'all_results': all_results,
            'n_successful': sum(1 for r in all_results if r.success),
            'transform_matrix': transform_matrix
        }
        
        if verbose:
            print(f"\nBest cost: {best_cost:.4f}")
            print(f"Successful optimizations: {results['n_successful']}/{n_restarts}")
            print(f"Transformation matrix:\n{transform_matrix}")
            
            # Calculate transformation properties
            det = a * d - b * c
            scale = np.sqrt(det)
            print(f"Determinant: {det:.4f}")
            print(f"Approximate scale: {scale:.4f}")
        
        return transform_matrix, results
    
    def apply_transform(self, mask: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Apply transformation to a mask."""
        # Convert to cv2 format (2x3 matrix)
        cv2_transform = transform_matrix[:2, :]
        
        # Apply transformation
        transformed_mask = cv2.warpAffine(
            (normalize_arr(mask)*255).astype(np.uint8),
            cv2_transform,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_LINEAR
        )
        
        return transformed_mask
    
    def visualize_registration(self, transform_matrix: np.ndarray, figsize: Tuple[int, int] = (15, 5)):
        """Visualize the registration results."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original masks
        axes[0].imshow(self.reference_mask, alpha=0.5, cmap='Blues', label='Reference')
        axes[0].imshow(self.sample_mask, alpha=0.5, cmap='Reds', label='Sample')
        axes[0].set_title('Before Registration')
        axes[0].legend()
        axes[0].axis('off')
        
        # Transform sample mask
        transformed_sample = self.apply_transform(self.sample_mask, transform_matrix)
        
        # After registration
        axes[1].imshow(self.reference_mask, alpha=0.5, cmap='Blues', label='Reference')
        axes[1].imshow(transformed_sample, alpha=0.5, cmap='Reds', label='Transformed Sample')
        axes[1].set_title('After Registration')
        axes[1].legend()
        axes[1].axis('off')
        
        # Boundary comparison
        # Extract parameters correctly from 3x3 matrix
        # Matrix format: [[a, b, tx], [c, d, ty], [0, 0, 1]]
        a, b, tx = transform_matrix[0, :]
        c, d, ty = transform_matrix[1, :]
        params = np.array([a, b, c, d, tx, ty])
        
        transformed_sample_points = self._affine_transform_points(self.sample_points, params)
        
        axes[2].scatter(self.reference_points[:, 0], self.reference_points[:, 1], 
                    color='b', alpha=0.7, s=1, label='Reference boundary')
        axes[2].scatter(transformed_sample_points[:, 0], transformed_sample_points[:, 1], 
                    color='r', alpha=0.7, s=1, label='Transformed sample boundary')
        axes[2].set_title('Boundary Alignment')
        axes[2].legend()
        axes[2].axis('equal')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def _get_hemisphere_masks(frame_npy, coords_dict, verbose):
    # determine the orientation of the frame based on the order of the three midline coordinates (y-coordinates)
    if coords_dict['bregma'][1] < coords_dict['crosspoint_medianline_frontalpole'][1]:
        orientation = 'up'
    else:   
        orientation = 'down'

    # create the hemisphere masks using predefined midline from coordinates file
    x_intercept = coords_dict['midline_equation']['x_intercept']
    angle_rad = np.radians(coords_dict['midline_equation']['angle_degrees'])

    # Convert to slope-intercept form for processing
    if abs(np.cos(angle_rad)) > 1e-6:
        m = np.tan(angle_rad)
    else:
        # line is vertical 
        m = 1e6 if coords_dict['midline_equation']['angle_degrees'] > 0 else -1e6
    
    # Line passes through (x_intercept, height/2)
    y = frame_npy.shape[0] / 2  # Use actual sample frame height
    b = y - m * x_intercept
    
    if verbose:
        print(f"Using predefined midline: x_intercept={x_intercept:.1f}, angle={coords_dict['midline_equation']['angle_degrees']:.1f}°")
        print(f"Converted to equation: y = {m:.4f}*x + {b:.2f}")

    # Create hemisphere masks for the sample
    height, width = frame_npy.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    if orientation == 'down':
        right_mask = (yy < (m * xx + b))
    else:   
        right_mask = (yy > (m * xx + b))
    left_mask = ~right_mask

    return right_mask, left_mask

def analyze_affine_transformation(transform_matrix: np.ndarray, verbose: bool = True) -> dict:
    """
    Analyze and quantify the components of an affine transformation.
    
    Args:
        transform_matrix: 3x3 transformation matrix
        verbose: Whether to print analysis
        
    Returns:
        Dictionary with transformation analysis
    """
    # Extract parameters from matrix
    a, b, tx = transform_matrix[0, :]
    c, d, ty = transform_matrix[1, :]
    
    # Calculate transformation components
    analysis = {}
    
    # 1. Translation
    translation_magnitude = np.sqrt(tx**2 + ty**2)
    translation_angle = np.arctan2(ty, tx) * 180 / np.pi
    analysis['translation'] = {
        'magnitude': translation_magnitude,
        'angle_degrees': translation_angle,
        'tx': tx,
        'ty': ty
    }
    
    # 2. Determinant (overall scaling + orientation)
    det = a * d - b * c
    analysis['determinant'] = det
    
    # 3. Decompose into rotation, scale, and shear using SVD
    # Create the linear part (no translation)
    linear_matrix = np.array([[a, b], [c, d]])
    
    # SVD decomposition: A = U * S * V^T
    U, S, Vt = np.linalg.svd(linear_matrix)
    
    # Extract uniform scaling
    uniform_scale = np.sqrt(det)
    analysis['uniform_scale'] = uniform_scale
    
    # Extract individual axis scaling
    scale_x = S[0]
    scale_y = S[1]
    analysis['scaling'] = {
        'uniform_scale': uniform_scale,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'scale_ratio': scale_x / scale_y if scale_y != 0 else float('inf')
    }
    
    # Extract rotation
    # For rotation, we need to be careful about reflections
    if det > 0:
        # No reflection
        rotation_matrix = U @ Vt
        rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
    else:
        # Reflection present
        rotation_angle = np.arctan2(-U[1, 0], U[0, 0]) * 180 / np.pi
    
    analysis['rotation'] = {
        'angle_degrees': rotation_angle,
        'angle_radians': rotation_angle * np.pi / 180
    }
    # 4. Shear analysis
    # Method 1: Direct shear calculation
    if abs(a) > 1e-10:  # Avoid division by zero
        shear_x = b / a  # Shear in x direction
    else:
        shear_x = 0
        
    if abs(d) > 1e-10:  # Avoid division by zero
        shear_y = c / d  # Shear in y direction
    else:
        shear_y = 0
    
    # Alternative shear measure: how much the transformation deviates from similarity
    # (pure rotation + uniform scaling)
    similarity_matrix = uniform_scale * np.array([
        [np.cos(rotation_angle * np.pi / 180), -np.sin(rotation_angle * np.pi / 180)],
        [np.sin(rotation_angle * np.pi / 180), np.cos(rotation_angle * np.pi / 180)]
    ])
    
    shear_deviation = np.linalg.norm(linear_matrix - similarity_matrix, 'fro')
    
    analysis['shear'] = {
        'shear_x': shear_x,
        'shear_y': shear_y,
        'shear_magnitude': np.sqrt(shear_x**2 + shear_y**2),
        'deviation_from_similarity': shear_deviation
    }
    
    # 5. Overall transformation type classification
    transform_type = []
    
    if translation_magnitude > 0.1:
        transform_type.append(f"Translation ({translation_magnitude:.1f} pixels)")
    
    if abs(uniform_scale - 1.0) > 0.05:
        if uniform_scale > 1.0:
            transform_type.append(f"Scaling up ({uniform_scale:.2f}x)")
        else:
            transform_type.append(f"Scaling down ({uniform_scale:.2f}x)")
    
    if abs(rotation_angle) > 1.0:
        transform_type.append(f"Rotation ({rotation_angle:.1f}°)")
    
    if analysis['shear']['shear_magnitude'] > 0.05:
        transform_type.append(f"Shear ({analysis['shear']['shear_magnitude']:.2f})")
    
    if abs(scale_x / scale_y - 1.0) > 0.05:
        transform_type.append(f"Anisotropic scaling ({scale_x:.2f}x, {scale_y:.2f}x)")
    
    if det < 0:
        transform_type.append("Reflection")
    
    analysis['transformation_type'] = transform_type if transform_type else ["Identity (no change)"]
    # 6. Transformation "strength" metrics
    analysis['strength'] = {
        'translation_strength': translation_magnitude / 100,  # Normalized by 100 pixels
        'rotation_strength': abs(rotation_angle) / 45,  # Normalized by 45 degrees
        'scale_strength': abs(uniform_scale - 1.0),  # Deviation from no scaling
        'shear_strength': analysis['shear']['shear_magnitude'],
        'overall_strength': np.sqrt(
            (translation_magnitude / 100)**2 + 
            (rotation_angle / 45)**2 + 
            (uniform_scale - 1.0)**2 + 
            analysis['shear']['shear_magnitude']**2
        )
    }
    if verbose:
        print("=== AFFINE TRANSFORMATION ANALYSIS ===")
        print(f"\nTransformation Matrix:")
        print(f"[[{a:8.4f}, {b:8.4f}, {tx:8.2f}]")
        print(f" [{c:8.4f}, {d:8.4f}, {ty:8.2f}]")
        print(f" [0.0000, 0.0000, 1.00]]")
        
        print(f"\nDeterminant: {det:.4f}")
        
        print(f"\n--- TRANSFORMATION COMPONENTS ---")
        print(f"Translation: {translation_magnitude:.2f} pixels at {translation_angle:.1f}°")
        print(f"  • X translation: {tx:.2f} pixels")
        print(f"  • Y translation: {ty:.2f} pixels")
        print(f"\nScaling:")
        print(f"  • Uniform scale: {uniform_scale:.3f}x")
        print(f"  • X-axis scale: {scale_x:.3f}x")
        print(f"  • Y-axis scale: {scale_y:.3f}x")
        print(f"  • Scale ratio: {analysis['scaling']['scale_ratio']:.3f}")
        
        print(f"\nRotation: {rotation_angle:.2f}°")
        
        print(f"\nShear:")
        print(f"  • X shear: {shear_x:.4f}")
        print(f"  • Y shear: {shear_y:.4f}")  
        print(f"  • Shear magnitude: {analysis['shear']['shear_magnitude']:.4f}")
        print(f"  • Deviation from similarity: {shear_deviation:.4f}")
        
        print(f"\n--- TRANSFORMATION TYPE ---")
        for t in analysis['transformation_type']:
            print(f"  • {t}")
        
        print(f"\n--- STRENGTH ANALYSIS ---")
        strength = analysis['strength']
        print(f"Translation strength: {strength['translation_strength']:.3f}")
        print(f"Rotation strength: {strength['rotation_strength']:.3f}")
        print(f"Scale strength: {strength['scale_strength']:.3f}")
        print(f"Shear strength: {strength['shear_strength']:.3f}")
        print(f"Overall strength: {strength['overall_strength']:.3f}")
        
        print("\n" + "="*50)
    
    return analysis, results

def create_symmetric_hemisphere_transform(transform_matrix: np.ndarray,
                                        midline_equation: dict,
                                        image_shape: tuple) -> np.ndarray:
    """
    Create transformation for symmetric hemisphere using the actual brain midline.
    This reflects the transformation across the brain's midline (which may not be vertical).
    
    Args:
        transform_matrix: Original transformation matrix for one hemisphere
        midline_equation: Dictionary with 'x_intercept' and 'angle_degrees' keys
        image_shape: (height, width) of the image for proper coordinate handling
    
    Returns:
        Transformation matrix for opposite hemisphere
    """
    # Extract midline parameters
    x_intercept = midline_equation['x_intercept']
    angle_rad = np.radians(midline_equation['angle_degrees'])
    
    # Convert to slope-intercept form: y = mx + b
    if abs(np.cos(angle_rad)) > 1e-6:
        m = np.tan(angle_rad)
    else:
        # Vertical line case
        m = 1e6 if midline_equation['angle_degrees'] > 0 else -1e6
    
    # Line passes through (x_intercept, height/2)
    y_point = image_shape[0] / 2
    b = y_point - m * x_intercept
    
    # Create reflection matrix across the midline y = mx + b
    # For a line ax + by + c = 0, the reflection matrix is:
    # R = I - 2 * (nn^T) / (n^T * n)
    # where n = [a, b] is the normal vector
    
    # Convert y = mx + b to ax + by + c = 0 form
    # mx - y + b = 0, so a = m, b = -1, c = b
    a_coeff = m
    b_coeff = -1
    
    # Normal vector to the line
    normal = np.array([a_coeff, b_coeff])
    normal_norm_sq = np.dot(normal, normal)
    
    # Create 2D reflection matrix
    reflection_2d = np.eye(2) - 2 * np.outer(normal, normal) / normal_norm_sq
    
    # Extract the linear part of the original transformation
    linear_part = transform_matrix[:2, :2]
    translation_part = transform_matrix[:2, 2]
    
    # Apply reflection to the linear transformation
    # For symmetric hemispheres, we want: R * A * R^(-1)
    # Since R is its own inverse for reflections: R * A * R
    reflected_linear = reflection_2d @ linear_part @ reflection_2d
    
    # For translation, we need to reflect the translation vector
    # and then adjust for the midline position
    reflected_translation = reflection_2d @ translation_part
    
    # Create the final transformation matrix
    mirrored_transform = np.array([
        [reflected_linear[0, 0], reflected_linear[0, 1], reflected_translation[0]],
        [reflected_linear[1, 0], reflected_linear[1, 1], reflected_translation[1]],
        [0, 0, 1]
    ])
    
    return mirrored_transform



# Example usage and comparison
if __name__ == "__main__":
    # load reference data
    reference_right_mask_npy = np.load(r"D:\allen_reference_atlas\640_540_binary_boundary_mask_nose_down_right_hemisphere.npy")
    reference_left_mask_npy = np.load(r"D:\allen_reference_atlas\640_540_binary_boundary_mask_nose_down_left_hemisphere.npy")
    
    # load sample hemisphere masks and full frame 
    sample_full_frame_npy = np.load(r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\Nicole\TEMP_OUTPUT\7203_test_blue_frame.npy")
    sample_right_mask_npy = np.load(r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\Nicole\TEMP_OUTPUT\7203_test_hemisphere_mask_right_hemisphere.npy")
    sample_left_mask_npy = np.load(r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\Nicole\TEMP_OUTPUT\7203_test_hemisphere_mask_left_hemisphere.npy")
    # load sample midline 
    sample_midline_dict = json.load(open(r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\Nicole\TEMP_OUTPUT\7203_test_midline.json"))['midline_equation']
    
 
  
    # compute the right hemisphere registration 
    r_registration = BrainHemisphereRegistration(sample_right_mask_npy, reference_right_mask_npy)
    right_transform_matrix, results = r_registration.register(max_iterations=2000,n_restarts=5,verbose=True)

    # Apply transformation to new data
    transformed_sample_mask_r = r_registration.apply_transform(sample_right_mask_npy, right_transform_matrix)
    transformed_sample_right_frame = r_registration.apply_transform(sample_full_frame_npy, right_transform_matrix) * reference_right_mask_npy
    
    #registration.visualize_registration(right_transform_matrix)
    l_registration = BrainHemisphereRegistration(sample_left_mask_npy, reference_left_mask_npy)
    left_transform_matrix, results = l_registration.register(max_iterations=2000,n_restarts=5,verbose=True)

    # Apply transformation to new data
    transformed_sample_mask_l = l_registration.apply_transform(sample_left_mask_npy, left_transform_matrix)
    transformed_sample_left_frame = l_registration.apply_transform(sample_full_frame_npy, left_transform_matrix) * reference_left_mask_npy

    plt.figure(figsize=(8, 4))
    plt.subplot(3,2,1)
    plt.axis('off')
    plt.imshow(reference_right_mask_npy, cmap='gray')
    plt.subplot(3,2,2)
    plt.axis('off')
    plt.imshow(reference_left_mask_npy, cmap='gray')
    plt.subplot(3,2,3)
    plt.axis('off')
    plt.imshow(sample_right_mask_npy, cmap='gray')
    plt.subplot(3,2,4)
    plt.axis('off')
    plt.imshow(sample_left_mask_npy, cmap='gray')
    plt.subplot(3,2,5)
    plt.axis('off')
    plt.imshow(transformed_sample_right_frame, cmap='gray')
    plt.subplot(3,2,6)
    plt.axis('off')
    plt.imshow(transformed_sample_left_frame, cmap='gray')
    plt.tight_layout()
    plt.show()

    
