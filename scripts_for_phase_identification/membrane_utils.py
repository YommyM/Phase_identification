import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional
from MDAnalysis import Universe

def remove_isolated_points_and_smooth(labels: np.ndarray, iterations: int) -> np.ndarray:
    """
    Remove isolated points (NaNs) in a 2D label array by replacing them with the 
    most frequent value among their neighbors in a 5x5 window using periodic boundaries.
    This operation is repeated for a specified number of iterations to improve smoothing.

    Parameters:
    ----------
    labels : np.ndarray
        A 2D NumPy array containing labels (can include NaN values).
    iterations : int
        The number of smoothing iterations to perform.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array of the same shape as `labels` with isolated points replaced 
        and labels smoothed.
    """
    rows, cols = labels.shape

    for _ in range(iterations):
        # Create a copy to store updated labels in this iteration
        new_labels = np.copy(labels)

        for i in range(rows):
            for j in range(cols):
                # Collect neighbors from a 5x5 window centered at (i, j)
                neighbors = []
                for di in range(-2, 3):  # Row offset range: [-2, 2]
                    for dj in range(-2, 3):  # Column offset range: [-2, 2]
                        if di == 0 and dj == 0:
                            continue  # Skip the center point itself
                        # Apply periodic boundary conditions
                        ni = (i + di) % rows
                        nj = (j + dj) % cols
                        neighbor_value = labels[ni, nj]
                        neighbors.append(neighbor_value)

                # Count occurrences of each non-NaN label among neighbors
                counts = {}
                for value in neighbors:
                    if not np.isnan(value):  # Ignore NaN values
                        counts[value] = counts.get(value, 0) + 1

                # Find the most frequent label in the neighborhood
                max_count = -1
                most_frequent = np.nan
                for key, count in counts.items():
                    if count > max_count:
                        max_count = count
                        most_frequent = key

                # Replace current label with the most frequent neighbor label
                new_labels[i, j] = most_frequent

        # Update labels for the next iteration
        labels = np.copy(new_labels)

    return new_labels

def get_lip_grid_ndx(
    atoms_grid_indices: List[List[Tuple[int, int]]],
    atoms_lipid: List[int],
    leaflet: int
) -> Dict[int, Tuple[Counter, int]]:
    """
    Assigns a unique grid index to each lipid molecule based on the most frequent 
    grid location of its atoms over multiple frames (e.g., 5 ns).

    Parameters
    ----------
    atoms_grid_indices : List[List[Tuple[int, int]]]
        A list of size (n_atoms), where each element is a list of (grid_x, grid_y)
        indices representing the atom's grid positions over time.
    atoms_lipid : List[int]
        A list of lipid IDs corresponding to each atom (length = n_atoms).
    leaflet : int
        The leaflet ID (e.g., 0 for upper or 1 for lower) to be assigned to each lipid.

    Returns
    -------
    Dict[int, Tuple[Counter, int]]
        A dictionary mapping each lipid ID to a tuple containing:
            - A Counter of grid indices and their occurrence counts among its atoms
            - The assigned leaflet ID
    """
    # Step 1: Find the most frequent grid index for each atom over time
    atoms_most_grid_index = []
    for atom_grid_timeseries in atoms_grid_indices:
        # Count occurrences of each grid index tuple for this atom
        counter = Counter(atom_grid_timeseries)
        # Select the most common grid index (mode)
        most_grid_index = counter.most_common(1)[0][0]
        atoms_most_grid_index.append(most_grid_index)
    # Now atoms_most_grid_index is a list of (x, y) tuples: length = n_atoms

    # Step 2: Aggregate most common grid indices per lipid
    lip_grid_indices: Dict[int, Counter] = {}

    for grid_index, lipid_id in zip(atoms_most_grid_index, atoms_lipid):
        if lipid_id not in lip_grid_indices:
            lip_grid_indices[lipid_id] = Counter()
        lip_grid_indices[lipid_id][grid_index] += 1

    # Step 3: Attach leaflet info to each lipid
    for lipid_id in lip_grid_indices:
        lip_grid_counter = lip_grid_indices[lipid_id]
        lip_grid_indices[lipid_id] = (lip_grid_counter, leaflet)

    return lip_grid_indices

def get_atom_grid_position(
    atom_coords: np.ndarray,
    x_grid_points: np.ndarray,
    y_grid_points: np.ndarray
) -> List[List[Tuple[int, int]]]:
    """
    Determines the 2D grid cell index (x, y) for each atom based on its x, y coordinates.

    Parameters
    ----------
    atom_coords : np.ndarray
        An (n_atoms, 3) array containing the (x, y, z) coordinates of atoms.
        Only the first two columns (x and y) are used.
    x_grid_points : np.ndarray
        A 1D array of x-axis grid boundaries.
    y_grid_points : np.ndarray
        A 1D array of y-axis grid boundaries.

    Returns
    -------
    List[List[Tuple[int, int]]]
        A list of length `n_atoms`, where each element is a list containing a single tuple (x_index, y_index)
        representing the grid cell in which the atom is located.
    """
    # Ensure the first grid point is zero 
    x_grid_points[0] = 0
    y_grid_points[0] = 0

    grid_indices: List[List[Tuple[int, int]]] = []

    for atom in atom_coords[:, :2]:
        x_coord, y_coord = atom

        # Find grid index along each axis
        x_index = np.searchsorted(x_grid_points, x_coord, side='left') - 1
        y_index = np.searchsorted(y_grid_points, y_coord, side='left') - 1

        # Clip index to ensure it is within valid range
        x_index = max(0, min(x_index, len(x_grid_points) - 2))
        y_index = max(0, min(y_index, len(y_grid_points) - 2))

        # Store grid index as a list of a tuple
        grid_indices.append([(x_index, y_index)])

    return grid_indices

def simulate_PBC(
    points_leaflet: np.ndarray,
    box: Tuple[float, float],
    r: float = 0.3,
    edge: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    Simulates 2D periodic boundary conditions (PBC) by generating additional 
    shifted copies of atoms near the x/y boundaries of the simulation box.

    Parameters
    ----------
    points_leaflet : np.ndarray
        An (N, 3) array of atomic coordinates (x, y, z).
    box : Tuple[float, float]
        The box size in the x and y dimensions.
    r : float, optional
        Ratio of the box edge (default: 0.3) to determine how close atoms must be to
        the boundary to trigger periodic copies.
    edge : Optional[Tuple[np.ndarray, np.ndarray]]
        Optional edge boundaries along x and y axes:
        edge = (x_edges, y_edges), each as 1D arrays.
        If not provided, the default range is extended slightly beyond the box.

    Returns
    -------
    np.ndarray
        An (M, 3) array of original and PBC-shifted coordinates within the edge window.
    """
    PBC_points: List[List[float]] = []

    # Determine output boundaries
    if edge is not None:
        x_min, x_max = edge[0][0], edge[0][-1]
        y_min, y_max = edge[1][0], edge[1][-1]
    else:
        x_min = 0
        x_max = (1 + r) * box[0]
        y_min = 0
        y_max = (1 + r) * box[1]

    for x,y,z in points_leaflet:
        # Always include the original point if within output range
        if x_min < x < x_max and y_min < y < y_max:
            PBC_points.append([x, y, z])

        shifts = []
        # Determine necessary PBC shifts in x or y directions
        if x > (1 - r) * box[0]: shifts.append([-box[0], 0])   # Near right edge → shift left
        if x < r * box[0]:       shifts.append([ box[0], 0])   # Near left edge → shift right
        if y > (1 - r) * box[1]: shifts.append([0, -box[1]])   # Near top edge → shift down
        if y < r * box[1]:       shifts.append([0,  box[1]])   # Near bottom edge → shift up

        # Diagonal combinations (corners)
        if x > (1 - r) * box[0] and y < r * box[1]:
            shifts.append([-box[0], box[1]])   # Bottom-right corner → top-left
        if x > (1 - r) * box[0] and y > (1 - r) * box[1]:
            shifts.append([-box[0], -box[1]])  # Top-right corner → bottom-left
        if x < r * box[0] and y < r * box[1]:
            shifts.append([ box[0], box[1]])   # Bottom-left corner → top-right
        if x < r * box[0] and y > (1 - r) * box[1]:
            shifts.append([ box[0], -box[1]])  # Top-left corner → bottom-right

        # Apply valid shifts and append new coordinates if within edge limits
        for dx, dy in shifts:
            new_x, new_y = x + dx, y + dy
            if x_min < new_x < x_max and y_min < new_y < y_max:
                PBC_points.append([new_x, new_y, z])

    return np.array(PBC_points)

def cal_density(
    u: Universe,
    sel: str,
    sel_key: str,
    diffnp: np.ndarray,
    edge: Tuple[np.ndarray, np.ndarray, np.ndarray],
    box_fr: np.ndarray,
    pr_id: Optional[List[int]] = None
):
    """
    Calculate 2D density maps (sum and mean) from a 3D system for lipids,
    with optional exclusion of protein densities.

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe object containing atom data.
    sel : str
        Atom selection string.
    sel_key : str
        Leaflet key: 'atom_upper', 'atom_lower'.
    diffnp : np.ndarray
        Centering vector to move COM to box center.
    edge : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Grid edge bins for x, y, z axes.
    box_fr : np.ndarray
        Box size array for current frame (length 3).
    pr_id : Optional[List[int]], default=None
        Protein residue IDs to exclude (if any).

    Returns
    -------
    dens2d_sum : np.ndarray
        2D density map by summing along z-axis.
    dens2d_mean : np.ndarray
        2D density map by averaging top values along z-axis.
    atom_to_grid : Optional[List[List[Tuple[int, int]]]]
        Grid indices for each selected atom (if sel_key is 'atom_upper' or 'atom_lower').
    atom_resids : Optional[np.ndarray]
        Residue IDs for selected atoms (if sel_key is 'atom_upper' or 'atom_lower').
    """
    sel_atoms_obj = u.select_atoms(sel)
    sel_atoms = sel_atoms_obj.positions
    sel_atoms_com = np.add(sel_atoms, diffnp)
    sel_atoms_pbc = np.mod(sel_atoms_com, box_fr)

    # Simulate periodic boundary conditions
    PBC_points = simulate_PBC(sel_atoms_pbc, box_fr, r=0.3, edge=edge)
    h, _ = np.histogramdd(PBC_points, bins=edge, normed=False)

    # If protein exclusion is enabled
    if pr_id is not None:
        sel_pr = 'resid ' + ' '.join(map(str, pr_id))
        sel_pr_positions = u.select_atoms(sel_pr).positions
        sel_pr_com = np.add(sel_pr_positions, diffnp)
        h_pr, _ = np.histogramdd(sel_pr_com, bins=edge, normed=False)
    else:
        h_pr = np.zeros_like(h)

    # Convert to densities
    x_edges, y_edges, z_edges = edge
    dx, dy, dz = np.diff(x_edges), np.diff(y_edges), np.diff(z_edges)
    dx_mesh, dy_mesh, dz_mesh = np.meshgrid(dx, dy, dz, indexing='ij')
    volumes = dx_mesh * dy_mesh * dz_mesh
    dens3d = h / volumes
    dens3d_pr = h_pr / volumes if pr_id is not None else None

    # Collapse along z-axis to get 2D maps
    dens2d_mean = np.zeros((dens3d.shape[0], dens3d.shape[1]))

    if pr_id is not None:
        z_min = np.min(np.where(dens3d > 0)[2])
        z_max = np.max(np.where(dens3d > 0)[2])

        for i in range(dens3d.shape[0]):
            for j in range(dens3d.shape[1]):
                dens_vals = dens3d[i, j, :]
                pr_vals = dens3d_pr[i, j, :]

                if np.sum(pr_vals[z_min:z_max+1]) != 0:
                    nonzero = dens_vals[dens_vals > 0]
                    if len(nonzero) > 0:
                        sorted_vals = np.sort(nonzero)
                        top_n = max(1, len(sorted_vals) // 4)
                        dens2d_mean[i, j] = np.mean(sorted_vals[-top_n:])
                else:
                    nonzero = dens_vals[dens_vals > 0]
                    if len(nonzero) > 0:
                        sorted_vals = np.sort(nonzero)
                        top_n = max(1, len(sorted_vals) // 2)
                        dens2d_mean[i, j] = np.mean(sorted_vals[-top_n:])
    else:
        # Original behavior: take top 1/2 of nonzero z-values for each (x, y)
        xy = dens3d.shape[0] * dens3d.shape[1]
        reshaped = dens3d.reshape(xy, dens3d.shape[2])
        mean_vals = np.zeros(xy)
        for i in range(xy):
            row = reshaped[i]
            nonzero = row[row > 0]
            if len(nonzero) > 0:
                sorted_vals = np.sort(nonzero)
                top_n = max(1, len(sorted_vals) // 2)
                mean_vals[i] = np.mean(sorted_vals[-top_n:])
        dens2d_mean = mean_vals.reshape(dens3d.shape[0], dens3d.shape[1])

    dens2d_sum = dens3d.sum(axis=2)

    # Return atom grid indices and resids if leaflet-specific
    if sel_key == 'atom_upper' or sel_key == 'atom_lower':
        atom_to_grid = get_atom_grid_position(sel_atoms_pbc, edge[0], edge[1])
        atom_resids = sel_atoms_obj.resids
        return dens2d_sum, dens2d_mean, atom_to_grid, atom_resids
    else:
        return dens2d_sum, dens2d_mean
    
def spatial_smooth_with_nan(matrix, n_iter=2, n_size =3):
	n, m = matrix.shape
	half_kernel = n_size // 2
	smoothed_matrix = np.copy(matrix)
	for _ in range(n_iter):
		temp_smoothed = np.zeros_like(smoothed_matrix)
		# 遍历矩阵的每个元素
		for i in range(n):
			for j in range(m):
				if np.isnan(smoothed_matrix[i,j]):
					temp_smoothed[i, j] = np.nan
					continue
				# 确定滤波器窗口的范围
				i_start = max(0, i - half_kernel)
				i_end = min(n, i + half_kernel + 1)
				j_start = max(0, j - half_kernel)
				j_end = min(m, j + half_kernel + 1)

				# 提取当前窗口内的数据
				window = smoothed_matrix[i_start:i_end, j_start:j_end]
				valid_values = window[ ~np.isnan(window)]
				if len(valid_values) > 0:
					mean_value = np.mean(valid_values)
				else:
					mean_value = np.nan  # 如果没有有效的值，则结果仍为NaN
				# 将计算得到的均值赋值给临时结果矩阵
				temp_smoothed[i, j] = mean_value
		# 更新smoothed_matrix用于下一次迭代
		smoothed_matrix = np.copy(temp_smoothed)
	return smoothed_matrix
def spatial_smooth(matrix, n_iter=2, n_size =3):
	new_matrix = deepcopy(matrix)
	for i in range(n_iter):
		new_matrix = np.around(uniform_filter(new_matrix, size = n_size, mode='wrap'), decimals=6)
	return new_matrix
def get_lipis_tags(tags_up_all, tags_low_all, lips_grid_ndx):
	lips_tag = []
	for t in range(len(lips_grid_ndx)): #200
		lips_tag_fr = []
		for lip_index in range(len(lips_grid_ndx[0])):  #对于每个lip的n个grid index
			lip_grids_counter, leaflet = lips_grid_ndx[t][lip_index]
			lip_tag_counts = Counter()
			for grid_index, count in lip_grids_counter.items():
				(i, j) = grid_index
				if (leaflet == 0): #gird index 转为tag
					tag = int(tags_up_all[t][i][j])
				else:
					tag = int(tags_low_all[t][i][j])
				if(np.isnan(tag)):
					continue
				else:
					lip_tag_counts[tag] += count
			most_common_tag, _ = lip_tag_counts.most_common(1)[0]
			lips_tag_fr.append(most_common_tag)
		lips_tag.append(lips_tag_fr)
	return lips_tag