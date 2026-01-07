
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import scipy
from scipy.optimize import least_squares, minimize
import cv2


def _find_closest_point_in_mask(coord, mask):
    # Get the indices of all points in the mask
    mask_points = np.argwhere(mask)  # Returns (row, col) pairs
    # Calculate Euclidean distances to all points in the mask
    distances = np.linalg.norm(mask_points - np.array(coord)[::-1], axis=1)
    # Find the index of the closest point
    closest_idx = np.argmin(distances)
    # return closest (convert from (row, col) to (x, y))
    return tuple(mask_points[closest_idx][::-1])


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

    # make sure the midline coords are aligned to be within the mask of each hemisphere
    right_coords = {}
    left_coords = {}
    for landmark, coord in coords_dict.items():
        if landmark != 'midline_equation':
            if landmark in ['bregma', 'crosspoint_medianline_frontalpole', 'anterior_tip_interparietal_bone']:
                # Align midline landmarks to each mask
                # assign to right_mask
                if not right_mask[coord[1], coord[0]]:
                    coord = _find_closest_point_in_mask(coord, right_mask)
                right_coords[landmark] = coord
                if not left_mask[coord[1], coord[0]]:
                    coord = _find_closest_point_in_mask(coord, left_mask)
                left_coords[landmark] = coord
            else:
                if "right" in landmark:
                    right_coords[landmark] = coord
                else:  # left hemisphere
                    left_coords[landmark] = coord

    return right_mask, left_mask, right_coords, left_coords

def _apply_affine_to_points(points, transform_matrix):
    points = np.array(points)
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    # Apply transformation
    transformed = (transform_matrix @ points_homo.T).T
    # Return as 2D points
    return transformed[:, :2]

def _fit_affine_landmarks(source_points, target_points, weights=None):
    """
    source_points : array_like, shape (N, 2); [(x1,y1), (x2,y2), ...]
    target_points : array_like, shape (N, 2); [(x1,y1), (x2,y2), ...]
    weights : array_like, shape (N,), optional; Higher weights = more influence.

    transform_matrix : ndarray, shape (3, 3); 2D affine transformation matrix
    residual_error : float; Root mean square error of the fit
    """
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have same shape")
    n_points = source_points.shape[0]
    if n_points < 3:
        raise ValueError("Need at least 3 landmarks for affine transformation")
    # Set up the overdetermined system: A * params = b
    # For affine: [x’, y’] = [a b tx; c d ty] * [x, y, 1]
    # This gives us: x’ = a*x + b*y + tx, y’ = c*x + d*y + ty
    # Create design matrix A
    A = np.zeros((2 * n_points, 6))
    b = np.zeros(2 * n_points)

    for i in range(n_points):
        # For x’ equation: a*x + b*y + tx = x’
        A[2*i, 0] = source_points[i, 0]      # coefficient for ‘a’
        A[2*i, 1] = source_points[i, 1]      # coefficient for ‘b’
        A[2*i, 2] = 1                        # coefficient for ‘tx’
        b[2*i] = target_points[i, 0]         # target x’
        # For y’ equation: c*x + d*y + ty = y’
        A[2*i+1, 3] = source_points[i, 0]    # coefficient for ‘c’
        A[2*i+1, 4] = source_points[i, 1]    # coefficient for ‘d’
        A[2*i+1, 5] = 1                      # coefficient for ‘ty’
        b[2*i+1] = target_points[i, 1]       # target y’
    # Apply weights if provided
    if weights is not None:
        weights = np.array(weights)
        W = np.zeros(2 * n_points)
        for i in range(n_points):
            W[2*i] = weights[i]      # weight for x equation
            W[2*i+1] = weights[i]    # weight for y equation
        # Weight the system: W*A*params = W*b
        A = A * W[:, np.newaxis]
        b = b * W
    # Solve least squares: params = (A^T A)^-1 A^T b
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    # Reconstruct transformation matrix
    transform_matrix = np.array([
        [params[0], params[1], params[2]],  # [a, b, tx]
        [params[3], params[4], params[5]],  # [c, d, ty]
        [0, 0, 1]                           # [0, 0, 1]
    ])
    # Calculate residual error
    predicted = _apply_affine_to_points(source_points, transform_matrix)
    residual_error = np.sqrt(np.mean(np.sum((predicted - target_points)**2, axis=1)))
    return transform_matrix, residual_error

def _apply_affine_to_image(frame_npy, transform_matrix, output_shape=None, mask=None):
    if output_shape is None:
        output_shape = frame_npy.shape
    # scipy.ndimage expects the inverse transformation
    # Our matrix transforms source -> target, but ndimage needs target -> source
    inv_matrix = np.linalg.inv(transform_matrix)
    # Extract the 2x3 affine matrix (ndimage format)
    affine_matrix = inv_matrix[:2, :3]
    # Apply transformation
    transformed_frame = scipy.ndimage.affine_transform(
        frame_npy,
        affine_matrix[:, :2],  # 2x2 linear part
        offset=affine_matrix[:, 2],  # translation part
        output_shape=output_shape,
        order=1,  # linear interpolation
        cval=0.0  # fill value for outside regions
    )
    if mask is not None:
        transformed_mask = scipy.ndimage.affine_transform(
            mask.astype(float),
            affine_matrix[:, :2],
            offset=affine_matrix[:, 2],
            output_shape=output_shape,
            order=0,  # nearest neighbor for binary mask
            cval=0.0
        ) > 0.5
    else:
        transformed_mask = None
    
    return transformed_frame, transformed_mask

def register_brain_to_brain(sample_frame_npy, sample_coords_dict, reference_frame_npy, reference_coords_dict, visualize=False, verbose=True):
    
    sample_right_mask, sample_left_mask, sample_right_coords, sample_left_coords = _get_hemisphere_masks(sample_frame_npy, sample_coords_dict, verbose)
    reference_right_mask, reference_left_mask, reference_right_coords, reference_left_coords = _get_hemisphere_masks(reference_frame_npy, reference_coords_dict, verbose)

    if visualize: 
        # plot each hemisphere with it's coords
        plt.figure()
        plt.subplot(3,2,1)
        plt.imshow(np.where(sample_left_mask, sample_frame_npy, np.nan))
        plt.title('Sample Left Hemisphere')
        for landmark, coord in sample_left_coords.items():
            plt.scatter(coord[0], coord[1], label=landmark)
        plt.subplot(3,2,2)
        plt.imshow(np.where(sample_right_mask, sample_frame_npy, np.nan))
        plt.title('Sample Right Hemisphere')
        for landmark, coord in sample_right_coords.items():
            plt.scatter(coord[0], coord[1], label=landmark)
        plt.subplot(3,2,3)
        plt.imshow(np.where(reference_left_mask, reference_frame_npy, np.nan))
        plt.title('Reference Left Hemisphere')
        for landmark, coord in reference_left_coords.items():
            plt.scatter(coord[0], coord[1], label=landmark)
        plt.subplot(3,2,4)
        plt.imshow(np.where(reference_right_mask, reference_frame_npy, np.nan))
        plt.title('Reference Right Hemisphere')
        for landmark, coord in reference_right_coords.items():
            plt.scatter(coord[0], coord[1], label=landmark)
        # plt.legend()
        
    

    # now that we have the coords and masks, we can register the sample to the reference
    right_source_points = []
    left_source_points = []
    right_target_points = []
    left_target_points = []
    right_midline_source = [] # temp
    right_midline_target = [] # temp
    right_outer_source = [] # temp
    right_outer_target = [] # temp
    left_midline_source = [] # temp
    left_midline_target = [] # temp
    left_outer_source = [] # temp
    left_outer_target = [] # temp
    weights = []
    midline_landmarks = ['bregma', 'crosspoint_medianline_frontalpole', 'anterior_tip_interparietal_bone']
    for landmark in sample_right_coords.keys():
        right_source_points.append(sample_right_coords[landmark])
        right_target_points.append(reference_right_coords[landmark])
        if landmark in midline_landmarks:
            weights.append(10.0)
            right_midline_source.append(sample_right_coords[landmark])
            right_midline_target.append(reference_right_coords[landmark])
        else:
            weights.append(1.0)
            right_outer_source.append(sample_right_coords[landmark])
            right_outer_target.append(reference_right_coords[landmark])
    for landmark in sample_left_coords.keys():
        left_source_points.append(sample_left_coords[landmark])
        left_target_points.append(reference_left_coords[landmark])
        if landmark in midline_landmarks:
            left_midline_source.append(sample_left_coords[landmark])
            left_midline_target.append(reference_left_coords[landmark])
        else:
            left_outer_source.append(sample_left_coords[landmark])
            left_outer_target.append(reference_left_coords[landmark])


    right_affine_matrix, inliers = cv2.estimateAffine2D(
        np.array(right_source_points), 
        np.array(right_target_points),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,  # Max allowed reprojection error in pixels
        maxIters=2000,
        confidence=0.99
    )
    '''method 1'''
    #right_affine_matrix_3x3 = np.vstack([right_affine_matrix, [0, 0, 1]])
    #transformed_right_frame, transformed_right_mask = _apply_affine_to_image(sample_frame_npy, right_affine_matrix_3x3, output_shape=reference_frame_npy.shape, mask=sample_right_mask)
    '''method 2'''
    #right_transform_matrix, right_residual_error = _fit_affine_landmarks(right_source_points, right_target_points, weights=weights)
    #left_transform_matrix, left_residual_error = _fit_affine_landmarks(left_source_points, left_target_points, weights=weights)
    '''method 3'''
    #right_transform_matrix, right_similarity_transform = two_stage_registration(right_midline_source, right_midline_target, right_outer_source, right_outer_target)
    #left_transform_matrix, left_similarity_transform = two_stage_registration(left_midline_source, left_midline_target, left_outer_source, left_outer_target)
    '''method 4'''
    right_transform_matrix = boundary_aware_registration(right_midline_source, right_midline_target, right_outer_source,  right_outer_target)
    left_transform_matrix = boundary_aware_registration(left_midline_source, left_midline_target, left_outer_source, left_outer_target)

    
    transformed_right_frame, transformed_right_mask = _apply_affine_to_image(sample_frame_npy, right_transform_matrix, output_shape=reference_frame_npy.shape, mask=sample_right_mask)
    transformed_left_frame, transformed_left_mask = _apply_affine_to_image(sample_frame_npy, left_transform_matrix, output_shape=reference_frame_npy.shape, mask=sample_left_mask)

    if visualize:
        plt.subplot(3,2,5)
        plt.imshow(np.where(reference_left_mask, transformed_left_frame, np.nan)*reference_frame_npy)
        plt.title('Transformed Sample Left Hemisphere')
        plt.subplot(3,2,6)
        plt.imshow(np.where(reference_right_mask, transformed_right_frame, np.nan)*reference_frame_npy)
        plt.title('Transformed Sample Right Hemisphere')
        plt.show()
    
    # determine which hemisphere fits best and then instead of using two different transformations per hemisphere
    # find the reverse transformation and apply it to the other hemisphere

    return 0

"""
Brune method test! 
"""
def boundary_aware_registration(midline_source, midline_target, outer_source, outer_target):
    """
    Stage 1: Perfect midline alignment (similarity transform)
    Stage 2: Optimize boundary fit (with constraint that midline stays fixed)
    """
    
    # Stage 1: Exact midline alignment
    similarity_transform = fit_similarity_transform(midline_source, midline_target)
    
    # Stage 2: Among transforms that preserve midline, find best boundary fit
    outer_after_similarity = apply_transform(outer_source, similarity_transform)
    
    # Find the additional linear transform that best fits boundary
    # while keeping midline alignment intact
    residual_transform = fit_boundary_preserving_transform(
        outer_after_similarity, outer_target, 
        midline_source, midline_target, similarity_transform
    )
    
    final_transform = residual_transform @ similarity_transform

    return final_transform

def fit_boundary_preserving_transform(outer_source, outer_target, 
                                    midline_source, midline_target, similarity_transform):
    """
    Find linear transform that:
    1. Best fits outer_source -> outer_target
    2. Doesn't disturb the midline alignment from similarity_transform
    """
    
    def constraint_objective(linear_params):
        # Build 2x2 linear transform
        linear_matrix = np.array([[linear_params[0], linear_params[1]], 
                                 [linear_params[2], linear_params[3]]])
        
        # Apply to outer points
        outer_transformed = outer_source @ linear_matrix.T
        boundary_error = np.sum((outer_transformed - outer_target)**2)
        
        # Check that midline is still aligned (soft constraint)
        midline_after_similarity = apply_transform(midline_source, similarity_transform)[:, :2]
        midline_after_linear = midline_after_similarity @ linear_matrix.T
        midline_error = np.sum((midline_after_linear + similarity_transform[:2, 2] - midline_target)**2)
        
        # Heavily penalize midline disturbance
        return boundary_error + 1000 * midline_error
    
    # Start with identity
    initial = [1, 0, 0, 1]
    result = minimize(constraint_objective, initial)
    
    # Convert back to 3x3 matrix
    linear_2x2 = np.array([[result.x[0], result.x[1]], 
                          [result.x[2], result.x[3]]])
    linear_3x3 = np.eye(3)
    linear_3x3[:2, :2] = linear_2x2
    
    return linear_3x3

def similarity_residual(params, source_points, target_points):
        tx, ty, theta, scale = params
        # Build similarity transformation matrix
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        transform = np.array([
            [scale * cos_t, -scale * sin_t, tx],
            [scale * sin_t,  scale * cos_t, ty],
            [0, 0, 1]
        ])
        # Apply to source points
        source_homo = np.column_stack([source_points, np.ones(len(source_points))])
        predicted = (transform @ source_homo.T).T[:, :2]
        # Calculate residuals
        residuals = (predicted - target_points).flatten()
        return residuals

def fit_similarity_transform(source_points, target_points):
    """
    Fit similarity transform (translation + rotation + uniform scaling) using midline points.
    Similarity transform: 4 DOF (tx, ty, rotation, scale)
    Matrix form: [s*cos(θ) -s*sin(θ) tx]
                 [s*sin(θ)  s*cos(θ) ty]
                 [0         0        1 ]
    Parameters:
    -----------
    source_points : array, shape (N, 2) - at least 2 points needed
    target_points : array, shape (N, 2)
    Returns:
    --------
    transform_matrix : array, shape (3, 3)
    """
    source_points = np.array(source_points)
    target_points = np.array(target_points)
    
    # Initial guess: no transformation
    initial_params = [0, 0, 0, 1]  # tx, ty, theta, scale
    # Solve
    result = least_squares(similarity_residual, initial_params, args=(source_points, target_points))
    tx, ty, theta, scale = result.x
    # Build final transformation matrix
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    transform_matrix = np.array([
        [scale * cos_t, -scale * sin_t, tx],
        [scale * sin_t,  scale * cos_t, ty],
        [0, 0, 1]
    ])
    return transform_matrix

def fit_residual_transform(source_points, target_points):
    """
    Fit the remaining transformation (differential scaling + shearing) after similarity.
    This finds the 2x2 linear transformation that best maps source to target:
    [x’] = [a b] [x]  (no translation - that’s handled by similarity)
    [y’]   [c d] [y]
    Parameters:
    -----------
    source_points : array, shape (N, 2) - points after similarity transform
    target_points : array, shape (N, 2) - target points
    Returns:
    --------
    linear_transform : array, shape (2, 2) - the [a b; c d] matrix
    """
    source_points = np.array(source_points)
    target_points = np.array(target_points)
    # Set up least squares system for linear transformation
    # [x’] = [a b] [x]  becomes  x’ = a*x + b*y, y’ = c*x + d*y
    # [y’]   [c d] [y]
    n_points = len(source_points)
    A = np.zeros((2 * n_points, 4))
    b = np.zeros(2 * n_points)
    for i in range(n_points):
        # x’ equation: a*x + b*y = x’
        A[2*i, 0] = source_points[i, 0]      # coefficient for ‘a’
        A[2*i, 1] = source_points[i, 1]      # coefficient for ‘b’
        b[2*i] = target_points[i, 0]         # target x’
        # y’ equation: c*x + d*y = y’ 
        A[2*i+1, 2] = source_points[i, 0]    # coefficient for ‘c’
        A[2*i+1, 3] = source_points[i, 1]    # coefficient for ‘d’
        b[2*i+1] = target_points[i, 1]       # target y’
    # Solve least squares
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # Reconstruct 2x2 matrix
    linear_transform = np.array([
        [params[0], params[1]],  # [a, b]
        [params[2], params[3]]   # [c, d]
    ])
    return linear_transform

def two_stage_registration(midline_source, midline_target, outer_source, outer_target):
    """
    Two-stage registration: midline for global alignment, outer points for deformation.
    Parameters:
    -----------
    midline_source : array, shape (3, 2) - source midline points
    midline_target : array, shape (3, 2) - target midline points
    outer_source : array, shape (3, 2) - source outer points
    outer_target : array, shape (3, 2) - target outer points
    Returns:
    --------
    final_transform : array, shape (3, 3) - complete affine transformation
    similarity_transform : array, shape (3, 3) - just the similarity part
    """
    # Stage 1: Fit similarity transform using midline points
    similarity_transform = fit_similarity_transform(midline_source, midline_target)
    print(f"Similarity transform fitted with {len(midline_source)} midline points")
    # Apply similarity transform to outer points
    outer_homo = np.column_stack([outer_source, np.ones(len(outer_source))])
    outer_after_similarity = (similarity_transform @ outer_homo.T).T[:, :2]
    # Stage 2: Fit residual linear transform using outer points
    linear_residual = fit_residual_transform(outer_after_similarity, outer_target)
    print(f"Residual transform fitted with {len(outer_source)} outer points")
    # Combine transformations: final = linear_residual * similarity
    # Convert linear residual to 3x3 form
    residual_3x3 = np.array([
        [linear_residual[0,0], linear_residual[0,1], 0],
        [linear_residual[1,0], linear_residual[1,1], 0],
        [0, 0, 1]
    ])
    # Combine: apply similarity first, then residual
    final_transform = residual_3x3 @ similarity_transform
    return final_transform, similarity_transform

def apply_transform(points, transform_matrix):
    """Apply 3x3 transformation to 2D points."""
    points = np.array(points)
    points_homo = np.column_stack([points, np.ones(len(points))])
    transformed = (transform_matrix @ points_homo.T).T
    return transformed[:, :2]

    
if __name__ == "__main__":
    sample_frame_npy = None
    with h5py.File(r"D:\wfield\NicoleData\WT\7202\WT_7202_processedData.h5py", 'r') as f:
        sample_frame_npy = f["WT_7202_LED_530_R_F0.5_ND0_FW1/motion_corrected"][0,0,...] # single blue frame
    
    sample_coords_dict = json.load(open(r"D:\wfield\NicoleData\WT\7202\coordinates.json"))

    # Load sample and reference frames and coordinates
    reference_frame_npy = np.load(r"D:\allen_reference_atlas\640_540_binary_boundary_mask_nose_down_full.npy")
    reference_coords_dict = json.load(open(r"D:\allen_reference_atlas\atlas_landmark_coordinates.json"))["640_540_nose_down"]

    register_brain_to_brain(sample_frame_npy, sample_coords_dict, reference_frame_npy, reference_coords_dict, visualize=True)


    