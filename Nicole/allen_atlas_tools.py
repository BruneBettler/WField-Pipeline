
import numpy as np
import os
from pipeline_utils import get_hdf5
import h5py
import matplotlib.pyplot as plt
from allensdk.core.reference_space_cache import ReferenceSpaceCache
import ccf_streamlines.projection as ccfproj
import cv2
import nrrd
import json

"""
Allen Brain Atlas extraction functions
"""

def get_allen_top_and_segmented(allen_ref_path):
    template, _ = nrrd.read(os.path.join(allen_ref_path, "average_template_10.nrrd"))

    proj_top = ccfproj.Isocortex2dProjector(
        # Specify our view lookup file
        os.path.join(allen_ref_path, "top.h5"),

        # Specify our streamline file
        os.path.join(allen_ref_path, "surface_paths_10_v3.h5"), 

        # Specify that we want to project both hemispheres
        hemisphere="both",

        # The top view contains space for the right hemisphere, but is empty.
        # Therefore, we tell the projector to put both hemispheres side-by-side
        view_space_for_other_hemisphere=True)

    top_projection_max = proj_top.project_volume(template)
    
    plt.show()
    plt.imshow(
        top_projection_max.T, # transpose so that the rostral/caudal direction is up/down
        interpolation='none',
        cmap='Greys_r')
    
    plt.show()

    boundary_finder = ccfproj.BoundaryFinder(
    projected_atlas_file=os.path.join(allen_ref_path, "top.nrrd"),
    labels_file=os.path.join(allen_ref_path, "labelDescription_ITKSNAPColor.txt")),

    # We get the left hemisphere region boundaries with the default arguments
    left_boundaries = boundary_finder.region_boundaries()

    # And we can get the right hemisphere boundaries that match up with
    # our projection if we specify the same configuration
    right_boundaries = boundary_finder.region_boundaries(
        # we want the right hemisphere boundaries, but located in the right place
        # to plot both hemispheres at the same time
        hemisphere='right_for_both',

        # we also want the hemispheres to be adjacent
        view_space_for_other_hemisphere=True)
    
    plt.imshow(
    top_projection_max.T,
    interpolation='none',
    cmap='Greys_r')

    for k, boundary_coords in left_boundaries.items():
        plt.plot(*boundary_coords.T, c="white", lw=0.5)
    for k, boundary_coords in right_boundaries.items():
        plt.plot(*boundary_coords.T, c="white", lw=0.5)

    plt.show()

    # Get image shape
    height, width = top_projection_max.T.shape
    segmented_mask = np.zeros((height, width), dtype=np.uint8)

    # Function to draw a polyline on the mask
    def draw_boundaries_on_mask(mask, boundaries_dict):
        for coords in boundaries_dict.values():
            coords = np.round(coords).astype(np.int32)  # Ensure integer pixel coords
            for i in range(len(coords) - 1):
                pt1 = tuple(coords[i])
                pt2 = tuple(coords[i + 1])
                cv2.line(mask, pt1, pt2, color=1, thickness=1)

    # Draw left and right boundaries
    draw_boundaries_on_mask(segmented_mask, left_boundaries)
    draw_boundaries_on_mask(segmented_mask, right_boundaries)

    # Optional: convert to float and set background as 0.0
    segmented_mask = segmented_mask.astype(float)

    plt.imshow(segmented_mask)
    plt.show()

    print("done")

    return 0 


# animal_path contains a coordinates.json file (with or without midline) and experimental sessions
# register the brain/coords to a common allen atlas such that all brains will then be mapped onto a single atlas 
# extract the transformation to be done on the mouse brain frame (such that these can then be applied directly to a stack of brain frames) 
# save to the coordinates.json file 

def hdf5_to_allen_registration(mouse_path, reference_full_atlas_path, reference_atlas_coords_path, 
                              save_output=True, visualize=False, use_robust_estimation=True, verbose=True):
    """
    Register mouse brain data to Allen atlas using anatomical landmarks with hemisphere splitting.
    
    The brain is split along the midline defined by three central landmarks:
    - bregma
    - crosspoint_medianline_frontalpole  
    - anterior_tip_interparietal_bone
    
    Each hemisphere is registered separately using 6 landmarks:
    - 3 central (shared) landmarks
    - 3 hemisphere-specific outer landmarks
    
    Args:
        mouse_path: Path to mouse folder containing coordinates.json and HDF5 files
        reference_atlas_mask_path: Path to FULL atlas npy file (will auto-detect hemisphere versions)
        reference_atlas_coords_path: Path to json file with atlas landmark coordinates
        save_output: Whether to save transformation matrix and registered atlas
        visualize: Whether to show visualization plots
        use_robust_estimation: Use RANSAC for robust transformation estimation
        verbose: Print debug information
    
    Returns:
        dict: Contains separate results for left and right hemispheres
    """
    
    if verbose:
        print(f"Processing mouse: {mouse_path}")
    
    # Auto-detect hemisphere-specific atlas files
    base_path = reference_atlas_mask_path.replace('_full.npy', '').replace('.npy', '')
    full_atlas_path = base_path + '_full.npy' if '_full' not in base_path else reference_atlas_mask_path
    left_atlas_path = base_path + '_left_hemisphere.npy'
    right_atlas_path = base_path + '_right_hemisphere.npy'
    
    # Check if hemisphere-specific files exist
    atlas_files = {
        'full': full_atlas_path if os.path.exists(full_atlas_path) else reference_atlas_mask_path,
        'left': left_atlas_path if os.path.exists(left_atlas_path) else None,
        'right': right_atlas_path if os.path.exists(right_atlas_path) else None
    }
    
    if verbose:
        print(f"Atlas files detected:")
        for hemi, path in atlas_files.items():
            if path:
                print(f"  {hemi.title()}: {os.path.basename(path)}")
            else:
                print(f"  {hemi.title()}: Not found - will use full atlas")
    
    # Load coordinates from mouse folder
    coord_path = os.path.join(mouse_path, 'coordinates.json')
    if not os.path.exists(coord_path):
        raise FileNotFoundError(f"Coordinates file not found: {coord_path}")
        
    with open(coord_path, 'r') as f:
        frame_coord_dict = json.load(f)
    
    # Load sample brain image
    hdf5_path = get_hdf5(mouse_path, verbose=verbose)
    with h5py.File(hdf5_path, 'r') as f:
        dataset_keys = list(f.keys())
        if verbose:
            print(f"Available datasets: {dataset_keys}")
        sample_frame = f[dataset_keys[0]]['motion_corrected'][0, 0]
    
    # Load full atlas for reference and brain orientation detection
    full_atlas_frame = np.load(atlas_files['full'])
    brain_orientation = reference_atlas_mask_path.split("_")[-1].replace(".npy", "")
    if brain_orientation == 'full':
        # Extract orientation from earlier part of filename
        brain_orientation = reference_atlas_mask_path.split("_")[-2]
    
    # Load atlas coordinates
    with open(reference_atlas_coords_path, 'r') as f:
        atlas_coord_dict = json.load(f)
        atlas_key = f"{sample_frame.shape[0]}_{sample_frame.shape[1]}_nose_{brain_orientation}"
        if atlas_key not in atlas_coord_dict:
            raise KeyError(f"Atlas coordinates not found for key: {atlas_key}")
        atlas_coord_dict = atlas_coord_dict[atlas_key]
    
    # Extract midline equation from coordinates file or calculate from landmarks
    if 'midline_equation' in frame_coord_dict:
        # Use predefined midline from coordinates file
        midline_eq = frame_coord_dict['midline_equation']
        x_intercept = midline_eq['x_intercept']
        angle_degrees = midline_eq['angle_degrees']
        
        # Convert to slope-intercept form for processing
        angle_rad = np.radians(angle_degrees)
        if abs(np.cos(angle_rad)) > 1e-6:
            m = np.tan(angle_rad)
        else:
            m = 1e6 if angle_degrees > 0 else -1e6
        
        # Line passes through (x_intercept, height/2)
        reference_y = sample_frame.shape[0] / 2  # Use actual sample frame height
        b = reference_y - m * x_intercept
        
        if verbose:
            print(f"Using predefined midline: x_intercept={x_intercept:.1f}, angle={angle_degrees:.1f}°")
            print(f"Converted to equation: y = {m:.4f}*x + {b:.2f}")
            
        midline_source = 'predefined'
        midline_coords = None  # No landmark-based coordinates
        
    else:
        # Fallback: calculate midline from landmarks (original behavior)
        midline_landmarks = ['bregma', 'crosspoint_medianline_frontalpole', 'anterior_tip_interparietal_bone']
        
        # Check that midline landmarks exist
        missing_midline = [lm for lm in midline_landmarks if lm not in frame_coord_dict]
        if missing_midline:
            raise ValueError(f"No predefined midline found and missing required midline landmarks: {missing_midline}")
        
        # Extract midline coordinates and fit line
        midline_coords = np.array([frame_coord_dict[lm] for lm in midline_landmarks])
        x_mid = midline_coords[:, 0]
        y_mid = midline_coords[:, 1]
        
        # Fit line through midline points: y = m*x + b
        m, b = np.polyfit(x_mid, y_mid, deg=1)
        
        if verbose:
            print(f"Calculated midline from landmarks: y = {m:.4f}*x + {b:.2f}")
            print(f"Using midline landmarks: {midline_landmarks}")
            
        midline_source = 'calculated'
        x_intercept = None
        angle_degrees = None
    
    # Create hemisphere masks
    height, width = sample_frame.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    
    if brain_orientation == 'right':
        # Points below the line are typically right hemisphere, above are left
        # (assuming typical orientation where rostral is up)
        right_mask = (yy < (m * xx + b))
        left_mask = ~right_mask
    else:   
        # Points above the line are typically right hemisphere, below are left
        # (assuming typical orientation where rostral is up)
        right_mask = (yy > (m * xx + b))
        left_mask = ~right_mask
    
    # Define hemisphere-specific landmark groups
    left_outer_landmarks = []
    right_outer_landmarks = []
    
    # Get midline landmarks for hemisphere processing
    if midline_source == 'predefined':
        # When using predefined midline, we need to determine which landmarks to treat as midline
        # For compatibility, we'll still use the standard midline landmarks if they exist
        midline_landmarks = ['bregma', 'crosspoint_medianline_frontalpole', 'anterior_tip_interparietal_bone']
        available_midline = [lm for lm in midline_landmarks if lm in frame_coord_dict]
        
        if len(available_midline) < len(midline_landmarks):
            if verbose:
                print(f"Warning: Only {len(available_midline)} of {len(midline_landmarks)} standard midline landmarks found")
                print(f"Available: {available_midline}, Missing: {[lm for lm in midline_landmarks if lm not in frame_coord_dict]}")
        
        # Use whatever midline landmarks are available
        midline_landmarks = available_midline
    else:
        # Use the calculated midline landmarks
        midline_landmarks = ['bregma', 'crosspoint_medianline_frontalpole', 'anterior_tip_interparietal_bone']
    
    # Automatically categorize landmarks based on their position relative to midline
    for landmark_name in frame_coord_dict.keys():
        if landmark_name not in midline_landmarks and landmark_name != 'midline_equation':
            if "right" in landmark_name:  # Below line (typically right hemisphere)
                right_outer_landmarks.append(landmark_name)
            else:  # Above line (typically left hemisphere)
                left_outer_landmarks.append(landmark_name)
    
    if verbose:
        print(f"Midline landmarks: {midline_landmarks}")
        print(f"Left hemisphere outer landmarks: {left_outer_landmarks}")
        print(f"Right hemisphere outer landmarks: {right_outer_landmarks}")
    
    # Process each hemisphere separately
    results = {
        'left_hemisphere': None,
        'right_hemisphere': None,
        'midline_info': {
            'source': midline_source,
            'equation': {'slope': m, 'intercept': b}
        },
        'atlas_files_used': atlas_files
    }
    
    # Add source-specific midline information
    if midline_source == 'predefined':
        results['midline_info']['x_intercept'] = x_intercept
        results['midline_info']['angle_degrees'] = angle_degrees
    else:
        results['midline_info']['landmarks'] = midline_landmarks
        results['midline_info']['coordinates'] = midline_coords.tolist()
    
    for hemisphere in ['left', 'right']:
        try:
            # Load hemisphere-specific atlas or use full atlas
            hemisphere_atlas_path = atlas_files[hemisphere]
            if hemisphere_atlas_path is None:
                hemisphere_atlas_path = atlas_files['full']
                if verbose:
                    print(f"Using full atlas for {hemisphere} hemisphere (hemisphere-specific not found)")
            
            hemisphere_atlas_frame = np.load(hemisphere_atlas_path)
            
            if hemisphere == 'left':
                mask = left_mask
                hemisphere_landmarks = midline_landmarks + left_outer_landmarks
                hemisphere_coords = left_outer_landmarks
            else:
                mask = right_mask
                hemisphere_landmarks = midline_landmarks + right_outer_landmarks
                hemisphere_coords = right_outer_landmarks
            
            if len(hemisphere_landmarks) < 3:
                if verbose:
                    print(f"Warning: Only {len(hemisphere_landmarks)} landmarks for {hemisphere} hemisphere, skipping")
                continue
            
            # Extract coordinates for this hemisphere
            frame_coords = []
            atlas_coords = []
            matched_landmarks = []
            
            for landmark_name in hemisphere_landmarks:
                if landmark_name in frame_coord_dict and landmark_name in atlas_coord_dict:
                    frame_coords.append(frame_coord_dict[landmark_name])
                    atlas_coords.append(atlas_coord_dict[landmark_name])
                    matched_landmarks.append(landmark_name)
                elif verbose:
                    print(f"Warning: Landmark '{landmark_name}' not found in atlas coordinates")
            
            if len(matched_landmarks) < 3:
                if verbose:
                    print(f"Warning: Only {len(matched_landmarks)} matching landmarks for {hemisphere} hemisphere, skipping")
                continue
            
            frame_coords = np.array(frame_coords, dtype=np.float32)
            atlas_coords = np.array(atlas_coords, dtype=np.float32)
            
            if verbose:
                print(f"\n{hemisphere.title()} hemisphere: Using {len(matched_landmarks)} landmarks: {matched_landmarks}")
                print(f"  Atlas file: {os.path.basename(hemisphere_atlas_path)}")
            
            # Estimate transformation matrix for this hemisphere
            if use_robust_estimation and len(matched_landmarks) > 3:
                M, inliers = cv2.estimateAffine2D(
                    frame_coords, atlas_coords,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0,
                    maxIters=2000,
                    confidence=0.99
                )
                inlier_mask = inliers.flatten().astype(bool)
                if verbose:
                    print(f"{hemisphere.title()} RANSAC inliers: {np.sum(inlier_mask)}/{len(matched_landmarks)}")
            else:
                M, inliers = cv2.estimateAffine2D(frame_coords, atlas_coords)
                inlier_mask = inliers.flatten().astype(bool) if inliers is not None else np.ones(len(matched_landmarks), dtype=bool)
            
            if M is None:
                if verbose:
                    print(f"Warning: Failed to estimate transformation for {hemisphere} hemisphere")
                continue
            
            # Extract hemisphere-specific brain region
            hemisphere_brain = np.where(mask, sample_frame, 0)
            
            # Apply transformation to register hemisphere to atlas space (using hemisphere-specific atlas dimensions)
            output_size = (hemisphere_atlas_frame.shape[1], hemisphere_atlas_frame.shape[0])
            registered_hemisphere = cv2.warpAffine(hemisphere_brain, M, output_size)
            
            # Compute inverse transformation
            M_inv = cv2.invertAffineTransform(M)
            registered_atlas_to_hemisphere = cv2.warpAffine(
                hemisphere_atlas_frame.astype(np.float32), M_inv, 
                (sample_frame.shape[1], sample_frame.shape[0])
            )
            
            # Calculate registration quality metrics
            transformed_frame_coords = cv2.transform(frame_coords.reshape(-1, 1, 2), M).reshape(-1, 2)
            registration_error = np.linalg.norm(transformed_frame_coords - atlas_coords, axis=1)
            mean_error = np.mean(registration_error[inlier_mask])
            
            if verbose:
                print(f"{hemisphere.title()} hemisphere mean error: {mean_error:.2f} pixels")
            
            # Store hemisphere results
            hemisphere_results = {
                'transformation_matrix': M,
                'inverse_transformation_matrix': M_inv,
                'inliers': inliers,
                'inlier_mask': inlier_mask,
                'matched_landmarks': matched_landmarks,
                'registration_error': registration_error,
                'mean_error': mean_error,
                'registered_brain': registered_hemisphere,
                'registered_atlas_to_brain': registered_atlas_to_hemisphere,
                'hemisphere_mask': mask,
                'hemisphere_brain': hemisphere_brain,
                'hemisphere_atlas': hemisphere_atlas_frame,
                'atlas_file_used': hemisphere_atlas_path,
                'frame_coords': frame_coords,
                'atlas_coords': atlas_coords
            }
            
            results[f'{hemisphere}_hemisphere'] = hemisphere_results
            
        except Exception as e:
            if verbose:
                print(f"Error processing {hemisphere} hemisphere: {e}")
            continue
    
    # Visualization
    if visualize:
        # Determine how many hemispheres we have results for
        left_results = results['left_hemisphere']
        right_results = results['right_hemisphere']
        
        if left_results is None and right_results is None:
            print("No hemisphere registration results to visualize")
        else:
            # Create larger subplot grid for hemisphere comparison
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            
            # Show original brain with midline and landmarks
            axes[0, 0].imshow(sample_frame, cmap='gray')
            # Plot midlineF
            x_line = np.linspace(0, width-1, 100)
            y_line = m * x_line + b
            y_test = np.where((y_line >= 0) & (y_line < height), y_line, 0)  # Ensure y is within bounds
            #axes[0, 0].plot(x_line, y_test, 'white', linewidth=2, label='Midline')
            
            # Plot landmarks with hemisphere colors
            for landmark_name, coord in frame_coord_dict.items():
                if landmark_name == 'midline_equation':  # Skip the equation entry
                    continue
                    
                if landmark_name in midline_landmarks:
                    color = 'yellow'
                    marker = 's'  # square
                elif landmark_name in left_outer_landmarks:
                    color = 'cyan'
                    marker = 'o'
                elif landmark_name in right_outer_landmarks:
                    color = 'magenta'  
                    marker = 'o'
                else:
                    color = 'white'  # uncategorized
                    marker = 'x'
                axes[0, 0].scatter(coord[0], coord[1], c=color, s=30, marker=marker)
            
            title_suffix = f"({midline_source} midline)" 
            axes[0, 0].set_title(f'Original Brain + Midline + All Landmarks\n{title_suffix}')
            
            # Show hemisphere masks
            axes[0, 1].imshow(left_mask.astype(float)*sample_frame, cmap='gray', alpha=0.7)
            axes[0, 1].set_title('Left Hemisphere Mask')
            
            axes[0, 2].imshow(right_mask.astype(float)*sample_frame, cmap='gray', alpha=0.7)
            axes[0, 2].set_title('Right Hemisphere Mask')
            
            # Show atlas reference (use full atlas for reference)
            axes[0, 3].imshow(full_atlas_frame, cmap='gray')
            axes[0, 3].scatter(atlas_coords[:, 0], atlas_coords[:, 1], c='red', s=30, label='Atlas Landmarks')
            axes[0, 3].set_title('Full Atlas Reference')
            
            # Process each hemisphere visualization
            for i, (hemisphere, hemi_results) in enumerate([('left', left_results), ('right', right_results)]):
                if hemi_results is None:
                    # Fill empty plots for missing hemisphere
                    for j in range(4):
                        axes[i+1, j].text(0.5, 0.5, f'No {hemisphere}\nhemisphere\nresults', 
                                        horizontalalignment='center', verticalalignment='center',
                                        transform=axes[i+1, j].transAxes, fontsize=12)
                        axes[i+1, j].set_title(f'{hemisphere.title()} Hemisphere - No Data')
                    continue
                
                # Hemisphere-specific brain with landmarks
                axes[i+1, 0].imshow(hemi_results['hemisphere_brain'], cmap='gray')
                for j, (coord, name) in enumerate(zip(hemi_results['frame_coords'], hemi_results['matched_landmarks'])):
                    color = 'red' if hemi_results['inlier_mask'][j] else 'orange'
                    axes[i+1, 0].scatter(coord[0], coord[1], c=color, s=50)
                axes[i+1, 0].set_title(f'{hemisphere.title()} Hemisphere + Landmarks')
                
                # Registered hemisphere in atlas space
                axes[i+1, 1].imshow(hemi_results['registered_brain'], cmap='gray')
                axes[i+1, 1].set_title(f'{hemisphere.title()} → Atlas Space')
                
                # Atlas registered to hemisphere space
                axes[i+1, 2].imshow(hemi_results['registered_atlas_to_brain'], cmap='gray')
                axes[i+1, 2].set_title(f'Atlas → {hemisphere.title()} Space')
                
                # Registration errors for this hemisphere
                errors = hemi_results['registration_error']
                axes[i+1, 3].bar(range(len(errors)), errors)
                axes[i+1, 3].set_title(f'{hemisphere.title()} Registration Errors')
                axes[i+1, 3].set_xlabel('Landmark Index')
                axes[i+1, 3].set_ylabel('Error (pixels)')
                
                # Add error statistics as text
                mean_err = hemi_results['mean_error']
                axes[i+1, 3].text(0.02, 0.98, f'Mean: {mean_err:.1f}px', 
                                transform=axes[i+1, 3].transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
    
    # Save outputs
    if save_output:
        output_dir = os.path.join(mouse_path, 'registration_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save midline information
        midline_data = {
            'source': midline_source,
            'equation': {'slope': float(m), 'intercept': float(b)},
            'left_outer_landmarks': left_outer_landmarks,
            'right_outer_landmarks': right_outer_landmarks
        }
        
        # Add source-specific information
        if midline_source == 'predefined':
            midline_data['x_intercept'] = float(x_intercept)
            midline_data['angle_degrees'] = float(angle_degrees)
        else:
            midline_data['landmarks'] = midline_landmarks
            midline_data['coordinates'] = midline_coords.tolist()
        
        with open(os.path.join(output_dir, 'midline_info.json'), 'w') as f:
            json.dump(midline_data, f, indent=2)
        
        # Save hemisphere-specific results
        for hemisphere in ['left', 'right']:
            hemi_results = results[f'{hemisphere}_hemisphere']
            if hemi_results is None:
                continue
                
            hemi_dir = os.path.join(output_dir, f'{hemisphere}_hemisphere')
            os.makedirs(hemi_dir, exist_ok=True)
            
            # Save transformation matrices
            np.save(os.path.join(hemi_dir, 'transformation_matrix.npy'), hemi_results['transformation_matrix'])
            np.save(os.path.join(hemi_dir, 'inverse_transformation_matrix.npy'), hemi_results['inverse_transformation_matrix'])
            
            # Save registered images
            np.save(os.path.join(hemi_dir, 'registered_brain_to_atlas.npy'), hemi_results['registered_brain'])
            np.save(os.path.join(hemi_dir, 'registered_atlas_to_brain.npy'), hemi_results['registered_atlas_to_brain'])
            np.save(os.path.join(hemi_dir, 'hemisphere_brain.npy'), hemi_results['hemisphere_brain'])
            np.save(os.path.join(hemi_dir, 'hemisphere_mask.npy'), hemi_results['hemisphere_mask'])
            
            # Save metadata
            metadata = {
                'matched_landmarks': hemi_results['matched_landmarks'],
                'mean_registration_error': float(hemi_results['mean_error']),
                'inlier_landmarks': [hemi_results['matched_landmarks'][i] for i in range(len(hemi_results['matched_landmarks'])) if hemi_results['inlier_mask'][i]],
                'brain_shape': sample_frame.shape,
                'atlas_shape': hemi_results['hemisphere_atlas'].shape,
                'atlas_file_used': hemi_results['atlas_file_used'],
                'hemisphere': hemisphere
            }
            
            with open(os.path.join(hemi_dir, 'registration_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        if verbose:
            print(f"Hemisphere registration outputs saved to: {output_dir}")
    
    return results


def apply_transformation_to_hemisphere_stack(brain_stack, hemisphere_results, atlas_shape, interpolation=cv2.INTER_LINEAR):
    """
    Apply hemisphere-specific transformation matrix to a stack of brain frames.
    
    Args:
        brain_stack: 3D array (time, height, width) or 4D array (time, channels, height, width)
        hemisphere_results: Results dict from hemisphere registration (contains mask and transformation)
        atlas_shape: Target shape (height, width) for output
        interpolation: OpenCV interpolation method
    
    Returns:
        np.ndarray: Transformed hemisphere stack in atlas space
    """
    if hemisphere_results is None:
        raise ValueError("No hemisphere results provided")
    
    transformation_matrix = hemisphere_results['transformation_matrix']
    hemisphere_mask = hemisphere_results['hemisphere_mask']
    
    original_shape = brain_stack.shape
    is_4d = len(original_shape) == 4
    
    if is_4d:
        n_time, n_channels, height, width = original_shape
        brain_stack = brain_stack.reshape(n_time * n_channels, height, width)
    else:
        n_time, height, width = original_shape
        n_channels = 1
    
    output_size = (atlas_shape[1], atlas_shape[0])  # (width, height)
    transformed_stack = np.zeros((brain_stack.shape[0], atlas_shape[0], atlas_shape[1]), dtype=brain_stack.dtype)
    
    for i in range(brain_stack.shape[0]):
        # Apply hemisphere mask first
        masked_frame = np.where(hemisphere_mask, brain_stack[i], 0)
        # Then apply transformation
        transformed_stack[i] = cv2.warpAffine(masked_frame, transformation_matrix, output_size, flags=interpolation)
    
    if is_4d:
        transformed_stack = transformed_stack.reshape(n_time, n_channels, atlas_shape[0], atlas_shape[1])
    
    return transformed_stack


def apply_transformation_to_stack(brain_stack, transformation_matrix, atlas_shape, interpolation=cv2.INTER_LINEAR):
    """
    Apply transformation matrix to a stack of brain frames (legacy function for backward compatibility).
    
    For hemisphere-specific registration, use apply_transformation_to_hemisphere_stack instead.
    
    Args:
        brain_stack: 3D array (time, height, width) or 4D array (time, channels, height, width)
        transformation_matrix: 2x3 affine transformation matrix
        atlas_shape: Target shape (height, width) for output
        interpolation: OpenCV interpolation method
    
    Returns:
        np.ndarray: Transformed brain stack in atlas space
    """
    original_shape = brain_stack.shape
    is_4d = len(original_shape) == 4
    
    if is_4d:
        n_time, n_channels, height, width = original_shape
        brain_stack = brain_stack.reshape(n_time * n_channels, height, width)
    else:
        n_time, height, width = original_shape
        n_channels = 1
    
    output_size = (atlas_shape[1], atlas_shape[0])  # (width, height)
    transformed_stack = np.zeros((brain_stack.shape[0], atlas_shape[0], atlas_shape[1]), dtype=brain_stack.dtype)
    
    for i in range(brain_stack.shape[0]):
        transformed_stack[i] = cv2.warpAffine(brain_stack[i], transformation_matrix, output_size, flags=interpolation)
    
    if is_4d:
        transformed_stack = transformed_stack.reshape(n_time, n_channels, atlas_shape[0], atlas_shape[1])
    
    return transformed_stack


def batch_register_animals(animal_paths, reference_atlas_mask_path, reference_atlas_coords_path, 
                          output_dir=None, save_individual=True, visualize=False, verbose=True):
    """
    Register multiple animals to the same atlas space with hemisphere-specific processing.
    
    Args:
        animal_paths: List of paths to animal folders
        reference_atlas_mask_path: Path to reference atlas
        reference_atlas_coords_path: Path to atlas coordinates  
        output_dir: Directory to save batch results
        save_individual: Whether to save individual animal results
        visualize: Whether to show plots
        verbose: Print progress
    
    Returns:
        dict: Batch registration results with hemisphere-specific quality metrics
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    batch_results = {
        'animal_results': {},
        'left_hemisphere_metrics': {
            'mean_errors': [],
            'transformation_matrices': [],
            'successful_animals': [],
            'failed_animals': []
        },
        'right_hemisphere_metrics': {
            'mean_errors': [],
            'transformation_matrices': [],
            'successful_animals': [],
            'failed_animals': []
        }
    }
    
    for i, animal_path in enumerate(animal_paths):
        if verbose:
            print(f"\nProcessing animal {i+1}/{len(animal_paths)}: {os.path.basename(animal_path)}")
        
        try:
            results = hdf5_to_allen_registration(
                animal_path, 
                reference_atlas_mask_path, 
                reference_atlas_coords_path,
                save_output=save_individual,
                visualize=visualize,
                verbose=verbose
            )
            
            animal_id = os.path.basename(animal_path)
            batch_results['animal_results'][animal_id] = results
            
            # Process each hemisphere separately
            for hemisphere in ['left', 'right']:
                hemi_results = results[f'{hemisphere}_hemisphere']
                hemi_metrics = batch_results[f'{hemisphere}_hemisphere_metrics']
                
                if hemi_results is not None:
                    hemi_metrics['mean_errors'].append(hemi_results['mean_error'])
                    hemi_metrics['transformation_matrices'].append(hemi_results['transformation_matrix'])
                    hemi_metrics['successful_animals'].append(animal_id)
                else:
                    hemi_metrics['failed_animals'].append((animal_id, f"No {hemisphere} hemisphere results"))
            
        except Exception as e:
            animal_id = os.path.basename(animal_path)
            # Add to both hemisphere failure lists
            batch_results['left_hemisphere_metrics']['failed_animals'].append((animal_id, str(e)))
            batch_results['right_hemisphere_metrics']['failed_animals'].append((animal_id, str(e)))
            if verbose:
                print(f"Failed to process {animal_id}: {e}")
    
    # Compute batch statistics for each hemisphere
    for hemisphere in ['left', 'right']:
        hemi_metrics = batch_results[f'{hemisphere}_hemisphere_metrics']
        
        if hemi_metrics['mean_errors']:
            hemi_metrics['overall_mean_error'] = np.mean(hemi_metrics['mean_errors'])
            hemi_metrics['overall_std_error'] = np.std(hemi_metrics['mean_errors'])
            
            if verbose:
                print(f"\n{hemisphere.title()} Hemisphere Summary:")
                print(f"Successful: {len(hemi_metrics['successful_animals'])}/{len(animal_paths)}")
                print(f"Mean registration error: {hemi_metrics['overall_mean_error']:.2f} ± {hemi_metrics['overall_std_error']:.2f} pixels")
    
    # Save batch results
    if output_dir:
        for hemisphere in ['left', 'right']:
            hemi_metrics = batch_results[f'{hemisphere}_hemisphere_metrics']
            
            # Save hemisphere-specific results
            save_data = {
                'successful_animals': hemi_metrics['successful_animals'],
                'failed_animals': hemi_metrics['failed_animals'],
                'mean_errors': hemi_metrics['mean_errors'],
                'overall_mean_error': hemi_metrics.get('overall_mean_error', None),
                'overall_std_error': hemi_metrics.get('overall_std_error', None)
            }
            
            with open(os.path.join(output_dir, f'{hemisphere}_hemisphere_batch_results.json'), 'w') as f:
                json.dump(save_data, f, indent=2)
            
            # Save transformation matrices
            if hemi_metrics['transformation_matrices']:
                transformation_stack = np.stack(hemi_metrics['transformation_matrices'])
                np.save(os.path.join(output_dir, f'{hemisphere}_hemisphere_transformation_matrices.npy'), transformation_stack)
    
    return batch_results


def analyze_registration_consistency(batch_results, hemisphere='both', visualize=True):
    """
    Analyze consistency of hemisphere-specific registrations across animals.
    
    Args:
        batch_results: Output from batch_register_animals
        hemisphere: 'left', 'right', or 'both' to analyze
        visualize: Whether to create plots
    
    Returns:
        dict: Analysis results for specified hemisphere(s)
    """
    
    def analyze_hemisphere_data(hemi_metrics, hemi_name):
        """Analyze data for a single hemisphere."""
        if not hemi_metrics['transformation_matrices']:
            print(f"No successful {hemi_name} hemisphere registrations to analyze")
            return {}
        
        matrices = np.array(hemi_metrics['transformation_matrices'])
        errors = np.array(hemi_metrics['mean_errors'])
        
        # Analyze transformation parameters
        scales_x = np.sqrt(matrices[:, 0, 0]**2 + matrices[:, 1, 0]**2)
        scales_y = np.sqrt(matrices[:, 0, 1]**2 + matrices[:, 1, 1]**2)
        rotations = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]) * 180 / np.pi
        translations_x = matrices[:, 0, 2]
        translations_y = matrices[:, 1, 2]
        
        return {
            'n_animals': len(matrices),
            'error_stats': {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors)
            },
            'scale_stats': {
                'scale_x': {'mean': np.mean(scales_x), 'std': np.std(scales_x)},
                'scale_y': {'mean': np.mean(scales_y), 'std': np.std(scales_y)}
            },
            'rotation_stats': {
                'mean': np.mean(rotations),
                'std': np.std(rotations),
                'range': np.max(rotations) - np.min(rotations)
            },
            'translation_stats': {
                'x': {'mean': np.mean(translations_x), 'std': np.std(translations_x)},
                'y': {'mean': np.mean(translations_y), 'std': np.std(translations_y)}
            },
            'matrices': matrices,
            'errors': errors
        }
    
    # Analyze specified hemisphere(s)
    analysis_results = {}
    
    if hemisphere in ['left', 'both']:
        left_metrics = batch_results['left_hemisphere_metrics']
        analysis_results['left_hemisphere'] = analyze_hemisphere_data(left_metrics, 'left')
    
    if hemisphere in ['right', 'both']:
        right_metrics = batch_results['right_hemisphere_metrics']
        analysis_results['right_hemisphere'] = analyze_hemisphere_data(right_metrics, 'right')
    
    # Visualization
    if visualize and analysis_results:
        # Determine subplot layout
        n_hemispheres = len(analysis_results)
        fig, axes = plt.subplots(n_hemispheres, 6, figsize=(24, 6*n_hemispheres))
        
        if n_hemispheres == 1:
            axes = axes.reshape(1, -1)
        
        hemisphere_names = list(analysis_results.keys())
        colors = ['blue' if 'left' in name else 'red' for name in hemisphere_names]
        
        for i, (hemi_name, hemi_analysis) in enumerate(analysis_results.items()):
            if not hemi_analysis:  # Skip empty analyses
                continue
                
            matrices = hemi_analysis['matrices']
            errors = hemi_analysis['errors']
            color = colors[i]
            
            # Recalculate parameters for plotting
            scales_x = np.sqrt(matrices[:, 0, 0]**2 + matrices[:, 1, 0]**2)
            scales_y = np.sqrt(matrices[:, 0, 1]**2 + matrices[:, 1, 1]**2)
            rotations = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]) * 180 / np.pi
            translations_x = matrices[:, 0, 2]
            translations_y = matrices[:, 1, 2]
            
            # Registration errors
            axes[i, 0].hist(errors, bins=10, alpha=0.7, color=color)
            axes[i, 0].set_xlabel('Mean Registration Error (pixels)')
            axes[i, 0].set_ylabel('Number of Animals')
            axes[i, 0].set_title(f'{hemi_name.replace("_", " ").title()}\nError Distribution')
            
            # Scales
            axes[i, 1].scatter(scales_x, scales_y, alpha=0.7, color=color)
            axes[i, 1].set_xlabel('Scale X')
            axes[i, 1].set_ylabel('Scale Y')
            axes[i, 1].set_title(f'{hemi_name.replace("_", " ").title()}\nScale Factors')
            axes[i, 1].plot([0.8, 1.2], [0.8, 1.2], 'k--', alpha=0.5)
            
            # Rotations
            axes[i, 2].hist(rotations, bins=10, alpha=0.7, color=color)
            axes[i, 2].set_xlabel('Rotation (degrees)')
            axes[i, 2].set_ylabel('Number of Animals')
            axes[i, 2].set_title(f'{hemi_name.replace("_", " ").title()}\nRotation Distribution')
            
            # Translations
            axes[i, 3].scatter(translations_x, translations_y, alpha=0.7, color=color)
            axes[i, 3].set_xlabel('Translation X (pixels)')
            axes[i, 3].set_ylabel('Translation Y (pixels)')
            axes[i, 3].set_title(f'{hemi_name.replace("_", " ").title()}\nTranslation Offsets')
            
            # Error vs Scale
            axes[i, 4].scatter(scales_x * scales_y, errors, alpha=0.7, color=color)
            axes[i, 4].set_xlabel('Total Scale Factor')
            axes[i, 4].set_ylabel('Registration Error')
            axes[i, 4].set_title(f'{hemi_name.replace("_", " ").title()}\nError vs Scale')
            
            # Error vs Rotation
            axes[i, 5].scatter(np.abs(rotations), errors, alpha=0.7, color=color)
            axes[i, 5].set_xlabel('Absolute Rotation (degrees)')
            axes[i, 5].set_ylabel('Registration Error')
            axes[i, 5].set_title(f'{hemi_name.replace("_", " ").title()}\nError vs Rotation')
        
        plt.tight_layout()
        plt.show()
    
    return analysis_results

# ------------------- JULY 23 -------------------
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def fit_affine_landmarks(source_points, target_points, weights=None):
    """
    Fit 2D affine transformation using least squares with multiple landmarks.
    
    Parameters:
    -----------
    source_points : array_like, shape (N, 2)
        Source landmark coordinates [(x1,y1), (x2,y2), ...]
    target_points : array_like, shape (N, 2)  
        Target landmark coordinates [(x1,y1), (x2,y2), ...]
    weights : array_like, shape (N,), optional
        Weights for each landmark pair. Higher weights = more influence.
        
    Returns:
    --------
    transform_matrix : ndarray, shape (3, 3)
        2D affine transformation matrix
    residual_error : float
        Root mean square error of the fit
    """
    source_points = np.array(source_points)
    target_points = np.array(target_points)
    
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have same shape")
    
    n_points = source_points.shape[0]
    if n_points < 3:
        raise ValueError("Need at least 3 landmarks for affine transformation")
    
    # Set up the overdetermined system: A * params = b
    # For affine: [x', y'] = [a b tx; c d ty] * [x, y, 1]
    # This gives us: x' = a*x + b*y + tx, y' = c*x + d*y + ty
    
    # Create design matrix A
    A = np.zeros((2 * n_points, 6))
    b = np.zeros(2 * n_points)
    
    for i in range(n_points):
        # For x' equation: a*x + b*y + tx = x'
        A[2*i, 0] = source_points[i, 0]      # coefficient for 'a'
        A[2*i, 1] = source_points[i, 1]      # coefficient for 'b' 
        A[2*i, 2] = 1                        # coefficient for 'tx'
        b[2*i] = target_points[i, 0]         # target x'
        
        # For y' equation: c*x + d*y + ty = y'
        A[2*i+1, 3] = source_points[i, 0]    # coefficient for 'c'
        A[2*i+1, 4] = source_points[i, 1]    # coefficient for 'd'
        A[2*i+1, 5] = 1                      # coefficient for 'ty'
        b[2*i+1] = target_points[i, 1]       # target y'
    
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
    predicted = apply_affine_to_points(source_points, transform_matrix)
    residual_error = np.sqrt(np.mean(np.sum((predicted - target_points)**2, axis=1)))
    
    return transform_matrix, residual_error

def apply_affine_to_points(points, transform_matrix):
    """Apply affine transformation to a set of points."""
    points = np.array(points)
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    # Apply transformation
    transformed = (transform_matrix @ points_homo.T).T
    # Return as 2D points
    return transformed[:, :2]

def apply_affine_to_image(image, transform_matrix, output_shape=None, mask=None):
    """
    Apply affine transformation to an image.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    transform_matrix : ndarray, shape (3, 3)
        Affine transformation matrix
    output_shape : tuple, optional
        Output image shape. If None, uses input shape.
    mask : ndarray, optional
        Binary mask. Transformed mask is returned along with image.
        
    Returns:
    --------
    transformed_image : ndarray
        Transformed image
    transformed_mask : ndarray, optional
        Transformed mask (if mask was provided)
    """
    if output_shape is None:
        output_shape = image.shape
    
    # scipy.ndimage expects the inverse transformation
    # Our matrix transforms source -> target, but ndimage needs target -> source
    inv_matrix = np.linalg.inv(transform_matrix)
    
    # Extract the 2x3 affine matrix (ndimage format)
    affine_matrix = inv_matrix[:2, :3]
    
    # Apply transformation
    transformed_image = ndimage.affine_transform(
        image, 
        affine_matrix[:, :2],  # 2x2 linear part
        offset=affine_matrix[:, 2],  # translation part
        output_shape=output_shape,
        order=1,  # linear interpolation
        cval=0.0  # fill value for outside regions
    )
    
    if mask is not None:
        transformed_mask = ndimage.affine_transform(
            mask.astype(float),
            affine_matrix[:, :2],
            offset=affine_matrix[:, 2],
            output_shape=output_shape,
            order=0,  # nearest neighbor for binary mask
            cval=0.0
        ) > 0.5
        return transformed_image, transformed_mask
    
    return transformed_image

def evaluate_registration_quality(source_points, target_points, transform_matrix):
    """Evaluate registration quality with detailed metrics."""
    predicted = apply_affine_to_points(source_points, transform_matrix)
    errors = np.sqrt(np.sum((predicted - target_points)**2, axis=1))
    
    return {
        'individual_errors': errors,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'predicted_points': predicted
    }

# Example usage
if __name__ == "__main__":
    # Example with 6 landmarks
    # In your case, these would be your actual landmark coordinates

    source_landmarks = np.array([
        [100, 150],  # outer landmark 1
        [300, 120],  # outer landmark 2  
        [500, 180],  # outer landmark 3
        [200, 250],  # midline landmark 1
        [320, 270],  # midline landmark 2
        [420, 260]   # midline landmark 3
    ])
    
    # Simulated target landmarks (with some transformation + noise)
    angle = np.radians(5)  # 5 degree rotation
    scale = 1.02          # 2% scaling
    translation = [10, -15]
    
    # Create "true" transformation for testing
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    true_transform = np.array([
        [scale * cos_a, -scale * sin_a, translation[0]],
        [scale * sin_a,  scale * cos_a, translation[1]],
        [0, 0, 1]
    ])
    
    target_landmarks = apply_affine_to_points(source_landmarks, true_transform)
    # Add some noise to make it realistic
    target_landmarks += np.random.normal(0, 1, target_landmarks.shape)
    
    # Fit affine transformation
    fitted_transform, residual = fit_affine_landmarks(source_landmarks, target_landmarks)
    
    # Evaluate quality
    quality = evaluate_registration_quality(source_landmarks, target_landmarks, fitted_transform)
    
    print("Registration Results:")
    print(f"RMSE: {quality['rmse']:.2f} pixels")
    print(f"Mean error: {quality['mean_error']:.2f} pixels") 
    print(f"Max error: {quality['max_error']:.2f} pixels")
    print(f"Individual errors: {quality['individual_errors']}")
    
    print("\nFitted transformation matrix:")
    print(fitted_transform)
    
    # Optional: weight outer landmarks more heavily
    weights = [2, 2, 2, 1, 1, 1]  # Higher weight for outer landmarks
    weighted_transform, weighted_residual = fit_affine_landmarks(
        source_landmarks, target_landmarks, weights=weights
    )
    
    weighted_quality = evaluate_registration_quality(
        source_landmarks, target_landmarks, weighted_transform
    )
    print(f"\nWith weighted fitting:")
    print(f"RMSE: {weighted_quality['rmse']:.2f} pixels")

#if __name__ == "__main__":
#    # Example usage
#    mouse_path = r"D:\wfield\NicoleData\WT\7204"
#    atlas_mask_path = r"D:\allen_reference_atlas\640_540_binary_segmented_mask_nose_down.npy"
#    atlas_coords_path = r"D:\allen_reference_atlas\atlas_landmark_coordinates.json"
#    
#    # Single animal registration
#    results = hdf5_to_allen_registration(
#        mouse_path, 
#        atlas_mask_path, 
#        atlas_coords_path,
#        visualize=True,
#        verbose=True
#    )
    
    # Example batch processing (uncomment when you have multiple animals)
    # animal_paths = [
    #     r"D:\wfield\NicoleData\WT\7204",
    #     r"D:\wfield\NicoleData\WT\7205",
    #     # Add more animal paths...
    # ]
    # 
    # batch_results = batch_register_animals(
    #     animal_paths,
    #     atlas_mask_path,
    #     atlas_coords_path,
    #     output_dir=r"D:\wfield\batch_registration_results",
    #     visualize=False
    # )
    # 
    # analysis = analyze_registration_consistency(batch_results)