import MDAnalysis
import argparse
import json
import os
from scipy.ndimage import uniform_filter
import numpy as np
from copy import deepcopy
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('-trj', type = str, help='name of trajectory')
parser.add_argument('-pdb', type = str, help = 'file name of system structure, e.g., pdb, gro, psf, etc...')
parser.add_argument('-start', type = int, help='begin frame number')
parser.add_argument('-end', type = int, help='end frame number')
parser.add_argument('-bin_width', type = int, help='bin width (Å)')
parser.add_argument('-n_gap', type = int, help='Calculate the mean of the density per n_gap frame')
parser.add_argument('-leaflet', type = str, help = 'leaflet file')
parser.add_argument('-sys', type = str, help = 'name of system')
parser.add_argument('-cal_ratio', type = str, help = 'calculate ratio or not')
parser.add_argument('-outpath', type = str, help = 'name of output dictionary')
parser.add_argument('-param', type = str, help = 'parameter file')
parser.add_argument('-primary_lipid', type = str, help = 'resid of primary lipid')
args = parser.parse_args()
trj = args.trj
pdb = args.pdb
start = args.start
end = args.end
bin_width = args.bin_width
n_gap = args.n_gap
fn_leaflet = args.leaflet
sys = args.sys
cal_ratio = args.cal_ratio
if cal_ratio.lower() == 'true':
	cal_ratio = True
else:
	cal_ratio = False
outpath = args.outpath
param = args.param
primary_lipid_id = [str(int(x.strip())) for x in args.primary_lipid.split(',')]
print('param:',param)
def compute_loss(threshold, data):
	# 将数据按阈值分成两类
	density_data = data[:, 0]
	# density_data = np.where (density_data <1.1, density_data, 1.1)
	# density_data = np.where (density_data >0.9, density_data, 0.9)
	lo_class = data[data[:, 0] >= threshold]  # 高密度的样本
	ld_class = data[data[:, 0] < threshold]   # 低密度的样本
	
	# 计算组内方差
	lo_variance = np.var(lo_class[:, 0]) if len(lo_class) > 0 else 0
	ld_variance = np.var(ld_class[:, 0]) if len(ld_class) > 0 else 0
	within_group_variance = (len(lo_class) * lo_variance + \
						  len(ld_class) * ld_variance*2) / (len(lo_class) + len(ld_class))
	
	# 计算组间方差
	lo_mean = np.mean(lo_class[:, 0]) if len(lo_class) > 0 else 0
	ld_mean = np.mean(ld_class[:, 0]) if len(ld_class) > 0 else 0
	between_group_variance = (len(lo_class) * (lo_mean - np.mean(density_data))**2 + 
							  len(ld_class) * (ld_mean - np.mean(density_data))**2) / (len(lo_class) + len(ld_class))
	
	# 计算loss (组内方差/组间方差)
	loss = within_group_variance / between_group_variance if between_group_variance != 0 else float('inf')
	
	return loss
def remove_isolated_points_and_smooth(labels, iter):
	if(iter == 0):
		return labels
	else:
		rows, cols = labels.shape
		for _ in range(iter):
			new_labels = np.copy(labels)
			for i in range(rows):
				for j in range(cols):
					# 获取周围 5x5 的邻居，考虑周期性边界条件
					neighbors = []
					for di in range(-2, 3):  # 行偏移范围 [-2, 2]
						for dj in range(-2, 3):  # 列偏移范围 [-2, 2]
							if di == 0 and dj == 0:  # 排除中心元素本身
								continue
							neighbor_value = labels[(i + di) % rows, (j + dj) % cols]
							neighbors.append(neighbor_value)
					# 统计出现次数的字典，忽略 NaN
					counts = {}
					for value in neighbors:
						if not np.isnan(value):  # 忽略 NaN
							counts[value] = counts.get(value, 0) + 1
					# 找到出现最多次的值
					max_count = -1
					most_frequent = None
					for key, count in counts.items():
						if count > max_count:
							max_count = count
							most_frequent = key
					# 替换 NaN
					new_labels[i, j] = most_frequent
			labels = np.copy(new_labels)
	return new_labels
def get_lip_grid_ndx(atoms_grid_indexs, atoms_lipid, leaflet):
	# atoms_grid_indexs: natoms*5*2
	# atoms_lipid: natoms*1
	####第一步：获得每个原子在5ns中最多次定位在哪个grid,作为该原子的唯一grid index
	atoms_most_grid_index = []
	# 遍历每个原子
	for atom_grid_indexs in atoms_grid_indexs:
		# 统计当前原子的grid indexs元组出现的次数
		counter = Counter(atom_grid_indexs)
		# 得到出现次数最多的grid index元组
		most_grid_index = counter.most_common(1)[0][0]
		# 将结果加入列表
		atoms_most_grid_index.append(most_grid_index)
	# 此时atoms_most_grid_index: natoms*2....[(0,1),(0,1),...,(4,5)]
	####第二步：对于每个lipid，根据其原子，确定lipid所属的grid indexs列表
	# 创建一个字典，存放每个resid的grid indexs列表及计数
	# 类似：
	# {
	#     lipid_id_1: Counter({grid_index_1: count_1, grid_index_2: count_2, ...}),
	#     lipid_id_2: Counter({grid_index_3: count_3, grid_index_4: count_4, ...}),
	#     ...
	# }
	lip_grid_indexs = {}
	for grid_index, lipid_id in zip(atoms_most_grid_index, atoms_lipid):
		# 如果分子 ID 不在字典中，则将其添加到字典，并初始化为一个空的 Counter
		if lipid_id not in lip_grid_indexs:
			lip_grid_indexs[lipid_id] = Counter()
		# 将当前原子的标签添加到对应分子的 Counter 中
		lip_grid_indexs[lipid_id][grid_index] += 1
	for key in lip_grid_indexs.keys():
		lip_grid_counter = lip_grid_indexs[key]
		lip_grid_indexs[key] = [lip_grid_counter, leaflet]
	return lip_grid_indexs 
def get_atom_grid_position(atom_coords, x_grid_points, y_grid_points):
	x_grid_points[0] = 0
	y_grid_points[0] = 0
	grids_index = []
	for atom in atom_coords[:,:2]:
		x_coord, y_coord = atom
		# 查找在x轴和y轴上的索引
		x_index = np.searchsorted(x_grid_points, x_coord, side='left') - 1
		y_index = np.searchsorted(y_grid_points, y_coord, side='left') - 1
		grids_index.append([(x_index, y_index)]) 
		# grids_index.append([(y_index, x_index)])
	return grids_index
def simulate_PBC(points_leaflet, box, r=0.3, edge=None):
	PBC_points = []
	# 使用 edge 来限制输出范围
	if edge is not None:
		x_min, x_max = edge[0][0], edge[0][-1]
		y_min, y_max = edge[1][0], edge[1][-1]
	else:
		x_min = 0
		x_max = (1 + r) * box[0]
		y_min = 0
		y_max = (1 + r) * box[1]
	for x, y, z in points_leaflet:
		# 原始点
		if x_min < x < x_max and y_min < y < y_max:
			PBC_points.append([x, y, z])
		# 方向平移（仅 x/y）
		shifts = []
		if x > (1 - r) * box[0]: shifts.append([-box[0], 0])
		if x < r * box[0]:        shifts.append([ box[0], 0])
		if y > (1 - r) * box[1]: shifts.append([0, -box[1]])
		if y < r * box[1]:        shifts.append([0,  box[1]])
		# 斜角方向组合（左上、右上、左下、右下）
		if x > (1 - r) * box[0] and y < r * box[1]:
			shifts.append([-box[0], box[1]])
		if x > (1 - r) * box[0] and y > (1 - r) * box[1]:
			shifts.append([-box[0], -box[1]])
		if x < r * box[0] and y < r * box[1]:
			shifts.append([ box[0], box[1]])
		if x < r * box[0] and y > (1 - r) * box[1]:
			shifts.append([ box[0], -box[1]])
		# 平移后坐标加入
		for dx, dy in shifts:
			new_x, new_y = x + dx, y + dy
			if x_min < new_x < x_max and y_min < new_y < y_max:
				PBC_points.append([new_x, new_y, z])
	return np.array(PBC_points)
def cal_density(u, sel, sel_key, diffnp, edge, box_fr):
	# Get the coordinates of the selected atoms
	sel_atoms_obj = u.select_atoms(sel)
	sel_atoms = sel_atoms_obj.positions
	# Move the com of system to the center of present box
	sel_atoms_com = np.add(sel_atoms, diffnp)
	# Rescale the coordinates between 0 - present boxsize
	sel_atoms_pbc = np.mod(sel_atoms_com, box_fr)
	# Generate some fake points to simulate PBC to fill the max box
	PBC_points = simulate_PBC(sel_atoms_pbc, box_fr, r=0.3, edge = edge) 
	# Get the number of atoms within each grid
	# h, _ = np.histogramdd(sel_atoms_pbc, bins=edge, normed = False)
	h, _ = np.histogramdd(PBC_points, bins=edge, normed = False)

	assert np.all(h >= 0), "Histogram contains negative counts!"
	# Convert histogram to densities 
	# x_edges, y_edges, z_edges = edge
	# dx = np.diff(x_edges) 
	# dy = np.diff(y_edges) 
	# dz = np.diff(z_edges) 
	# dx_mesh, dy_mesh, dz_mesh = np.meshgrid(dx, dy, dz, indexing='ij')
	# volumes = dx_mesh * dy_mesh * dz_mesh 
	# dens3d = h / volumes

	volume = bin_width*bin_width*bin_width/6
	dens3d = h / volume

	# Collapse along the z-axis
	# top 1/2 mean
	xy = dens3d.shape[0] * dens3d.shape[1]
	z = dens3d.shape[2]
	reshaped = dens3d.reshape(xy, z)
	mean_vals = np.zeros(xy)
	for i in range(xy):
		row = reshaped[i]
		nonzero = row[row > 0]
		if len(nonzero) > 0:
			sorted_vals = np.sort(nonzero)
			num_keep = max(1, len(sorted_vals) // 2)
			mean_vals[i] = np.mean(sorted_vals[-num_keep:])
	dens2d_mean = mean_vals.reshape(dens3d.shape[0], dens3d.shape[1])
	# # 策略： top 1/2 mean
	# dens2d_mean = np.zeros(shape = (dens3d.shape[0], dens3d.shape[1]))

	# for i in range(0, dens3d.shape[0]):
	#     for j in range(0, dens3d.shape[1]):
	#         all_z = dens3d[i, j, :]
	#         nonzero = all_z[all_z>0]
	#         if len(nonzero) > 0:
	#             sorted_z = sorted(nonzero, reverse=True)
	#             top_n = max(1, len(sorted_z) // 2)
	#             top_z = sorted_z[:top_n]
	#             dens2d_mean[i,j] = np.array(top_z).mean()
	# sum
	dens2d_sum = dens3d.sum(axis=2)

	# Get the grid index and resid for each atom.
	if(sel_key == 'atom_upper'):
		sel_atoms2grids_index_up = get_atom_grid_position(sel_atoms_pbc, edge[0][:], edge[1][:])
		sel_atoms2lips_up =  sel_atoms_obj.resids
		return dens2d_sum, dens2d_mean, sel_atoms2grids_index_up, sel_atoms2lips_up
	elif(sel_key == 'atom_lower'):
		sel_atoms2grids_index_low = get_atom_grid_position(sel_atoms_pbc, edge[0][:], edge[1][:])
		sel_atoms2lips_low = sel_atoms_obj.resids
		return dens2d_sum, dens2d_mean, sel_atoms2grids_index_low, sel_atoms2lips_low
	else:
		return dens2d_sum, dens2d_mean
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
				lip_tag_counts[tag] += count
			most_common_tag, _ = lip_tag_counts.most_common(1)[0]
			lips_tag_fr.append(most_common_tag)
		lips_tag.append(lips_tag_fr)
	return lips_tag
def spatial_smooth(matrix, n_iter=2, n_size =3):
	new_matrix = deepcopy(matrix)
	for i in range(n_iter):
		new_matrix = np.around(uniform_filter(new_matrix, size = n_size, mode='wrap'), decimals=6)
	return new_matrix
def get_tag_by_first_feature(data,raw_tags):
	# 计算每个聚类中第一个特征的均值
	means = []
	for cluster in set(raw_tags):
		cluster_points = data[raw_tags == cluster]  # 提取当前标签对应的样本
		mean_value = cluster_points[:, 0].mean()  # 计算第一个特征的均值
		means.append((cluster, mean_value))
	# 根据均值大小重新定义标签
	means.sort(key=lambda x: x[1])  # 按均值排序，较小均值排在前面
	mapping = {means[0][0]: 0, means[1][0]: 1}  # 均值小的映射为 2，均值大的映射为 1
	# 映射原始标签到新的标签
	new_tags = np.array([mapping[tag] for tag in raw_tags])
	return new_tags
def combine_tags(tags_2, new_tags_of1):
	# 创建一个与原始标签相同长度的数组
	final_tags = np.copy(tags_2)
	# 找到原始标签中为1的索引
	indices_of_1 = np.where(tags_2 == 1)[0]
	# 将二次分类的结果放回原来的位置
	# 如果二次分类为1的部分要标记为2，可以将new_tags_of1加1
	final_tags[indices_of_1] = new_tags_of1
	return final_tags
def get_gmm_threshold(data_1d, plot=True):
	X = data_1d[~np.isnan(data_1d)].reshape(-1, 1)
	gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
	gmm.fit(X)

	means = gmm.means_.flatten()
	stds = np.sqrt(gmm.covariances_.flatten())
	weights = gmm.weights_.flatten()

	# 保证顺序：means[0] 是 Ld，means[1] 是 Lo
	if means[0] > means[1]:
		means = means[::-1]
		stds = stds[::-1]
		weights = weights[::-1]

	# 尝试获取交点
	try:
		def f(x): return weights[0]*norm.pdf(x, means[0], stds[0]) - weights[1]*norm.pdf(x, means[1], stds[1])
		intersection = brentq(f, means[0], means[1])
	except:
		intersection = np.mean(means)  # fallback

	# 所有候选阈值
	candidates = {
		"intersection": intersection,
		"mean": np.mean(means),
		"weighted": np.dot(means, weights),
		"hybrid": 0.5 * intersection + 0.5 * np.dot(means, weights)
		# "hybrid": 0.5 * intersection + 0.5 * np.mean(means)
	}
	return candidates['hybrid']
def cluster_all(h_atom_up_mean_all, h_atom_low_mean_all, \
				iterations=0, param="None", cal_ratio = cal_ratio):
	# 得到非平衡态的均值和方差
	if param != 'None':
		with open(param, 'r') as json_file:
			params = json.load(json_file)
		density_mean = params['density']['mean']
		threshold = params['phase']['threshold']
	else:
		density_mean = np.mean(h_atom_up_mean_all)/2+np.mean(h_atom_low_mean_all)/2
	print('start:',start,'\tend:',end,'\ndensity_mean:',density_mean)
	h_atom_up_mean_all_normed = [matrix / density_mean for matrix in h_atom_up_mean_all]
	h_atom_low_mean_all_normed = [matrix / density_mean for matrix in h_atom_low_mean_all]
	h_atom_up_mean_all_normed_filtered = [spatial_smooth(matrix) for matrix in h_atom_up_mean_all_normed]
	h_atom_low_mean_all_normed_filtered = [spatial_smooth(matrix) for matrix in h_atom_low_mean_all_normed] 

	if(cal_ratio):
		h_ratio_up_all = [h_primary_lipid_up_sum_all[fr]/h_atom_up_sum_all[fr] for fr in range(len(h_primary_lipid_up_sum_all))]
		h_ratio_low_all = [h_primary_lipid_low_sum_all[fr]/h_atom_low_sum_all[fr] for fr in range(len(h_primary_lipid_low_sum_all))]
		h_ratio_up_all_filtered = [spatial_smooth(matrix) for matrix in h_ratio_up_all]
		h_ratio_low_all_filtered = [spatial_smooth(matrix) for matrix in h_ratio_low_all]
	
	h_up_all = np.array(h_atom_up_mean_all_normed_filtered)
	h_low_all = np.array(h_atom_low_mean_all_normed_filtered)
	n_leaflet_samples = h_up_all.shape[0]*h_up_all.shape[1]*h_up_all.shape[2]
	density_up_flattened = h_up_all.reshape(n_leaflet_samples, -1).tolist()
	density_low_flattened = h_low_all.reshape(n_leaflet_samples, -1).tolist()
	density_flattened = density_up_flattened + density_low_flattened
	if(cal_ratio):
		h_ratio_up_all_filtered = np.array(h_ratio_up_all_filtered)
		h_ratio_low_all_filtered = np.array(h_ratio_low_all_filtered)
		r_up_flattened = h_ratio_up_all_filtered.reshape(n_leaflet_samples, -1).tolist()
		r_low_flattened = h_ratio_low_all_filtered.reshape(n_leaflet_samples, -1).tolist()
		r_flattened = r_up_flattened + r_low_flattened
		flattened_matrix = np.concatenate((density_flattened, r_flattened), axis=1)
	else:
		flattened_matrix = np.array(deepcopy(density_flattened)).reshape(-1,1)
	if param == 'None':
		best_threshold = get_gmm_threshold(flattened_matrix[:, 0])
		tags = np.where(list(flattened_matrix[:,0]) > best_threshold, 1, 0)
		# Step 3: 提取每个标签的联合分布参数
		output_data = {}
		output_data['density'] = {
				"mean":density_mean
			}
		output_data['phase'] = {
				"threshold": best_threshold
			}
		# print(output_data['phase'])
		with open(os.path.join(os.path.dirname(os.path.abspath(outpath)), 'parameters.json'), 'w') as json_file:
			json.dump(output_data, json_file, indent=4)
	else:
		tags = np.where(flattened_matrix[:,0] > threshold, 1, 0)

	split_idx = len(tags) // 2
	tag_matrixs_up = tags[:split_idx].reshape(h_up_all.shape[0],h_up_all.shape[1],h_up_all.shape[2])
	tag_matrixs_low = tags[split_idx:].reshape(h_up_all.shape[0],h_up_all.shape[1],h_up_all.shape[2])
	# 在像素化平面针对grid的tag进行空间平滑
	final_tag_matrixs_up = deepcopy(tag_matrixs_up)
	final_tag_matrixs_low = deepcopy(tag_matrixs_low)
	for n_fr in range(h_up_all.shape[0]):
		tag_matrix_up_fr = tag_matrixs_up[n_fr]
		tag_matrix_low_fr = tag_matrixs_low[n_fr]
		final_tag_matrixs_up[n_fr] = remove_isolated_points_and_smooth(tag_matrix_up_fr, iter=iterations) #控制是否平滑
		final_tag_matrixs_low[n_fr] = remove_isolated_points_and_smooth(tag_matrix_low_fr, iter=iterations) #控制是否平滑
	print('end of clustering all')
	return final_tag_matrixs_up, final_tag_matrixs_low
def split_into_segments(sequence):
	segments = []
	current_char = sequence[0]
	count = 1
	
	for char in sequence[1:]:
		if char == current_char:
			count += 1
		else:
			segments.append((current_char, count))
			current_char = char
			count = 1
	segments.append((current_char, count))
	return segments
def merge_segments(segments):
	tmp = list(''.join([char * count for char, count in segments]))
	return tmp
def correct_min_repeat_tags(segments, repeat = 2):
	for r in range(1, repeat+1):
		# print('r = ',r)
		iter = 0
		while True:
			iter += 1
			# print('1st: iter: ' + str(iter))
			if iter > 2000:
				raise RuntimeError("Error: Iteration limit exceeded, potential infinite loop detected!")
			corrected_segments = segments.copy()
			for i in range(len(segments)):
				char, count = segments[i]
				if(len(segments) >= 3):
					if count <= r:
						if i == 0 and len(segments) > 1:
							# 首个segment的处理
							right_char, right_count = segments[i+1]
							if right_count > repeat:
								corrected_segments[i] = (right_char, count)
						elif i == len(segments) - 1 and len(segments) > 1:
							# 末尾segment的处理
							left_char, left_count = segments[i-1]
							if left_count > repeat:
								corrected_segments[i] = (left_char, count)
						else:
							# 中间segment的处理
							left_char, left_count = segments[i-1]
							right_char, right_count = segments[i+1]
							
							if left_count > r or right_count > r:
								if left_count >= right_count:
									corrected_segments[i] = (left_char, count)
								else:
									corrected_segments[i] = (right_char, count)
				else:
					if count <= repeat:
						if (i + 1 < len(segments)):
							right_char, right_count = segments[i+1]
							corrected_segments[i] = (right_char, count)
						else:
							left_char, left_count = segments[i-1]
							corrected_segments[i] = (left_char, count)
			cor_seq = merge_segments(corrected_segments)
			# print(cor_seq)
			segments = split_into_segments(cor_seq)
			
			if r < repeat:
				if all(count > r for char, count in segments[1:-1]) \
				   and len(segments)>=3:
					break
				elif( len(segments)<3):
					break
			else:
				if all(count > r for char, count in segments):
					break
	return segments
def temporal_smoothing (matrix):
	columns = [matrix[:, i].tolist() for i in range(matrix.shape[1])]
	new_columns = []
	for sequence in columns:
		# print('raw sequence:',end = ' ')
		# print(sequence)
		sequence    = [str(x) for x in sequence]
		segments = split_into_segments(sequence)
		corrected_segs = correct_min_repeat_tags(segments)
		corrected_sequence = merge_segments(corrected_segs)
		corrected_sequence = [int(x) for x in corrected_sequence]
		new_columns.append(corrected_sequence) 
		# print('new sequence:',end = ' ')
		# print(final_corrected_sequence)
	new_array = np.array(list(map(np.array, zip(*new_columns))))
	return new_array
def process_frame(pdb, trj, frame_index, sel_atom_upper, sel_atom_lower, sel_primary_lipid_upper, sel_primary_lipid_lower, edge, cal_ratio):
	"""
	独立加载 Universe 并处理每一帧数据
	"""
	# 跳转到指定帧
	u = MDAnalysis.Universe(pdb, trj)
	ts = u.trajectory[frame_index]
	box_fr = ts.dimensions[:3]
	sel_all_com = u.select_atoms('all').center_of_mass()
	diffx = box_fr[0] / 2 - sel_all_com[0]
	diffy = box_fr[1] / 2 - sel_all_com[1]
	diffz = box_fr[2] / 2 - sel_all_com[2]
	diffnp = np.array([diffx, diffy, diffz])

	# 计算原子密度
	atom_up_sum, atom_up_mean, atoms2grid_up_fr, atoms2lip_up = cal_density(u, 
		sel_atom_upper, 'atom_upper', diffnp, edge, box_fr)
	atom_low_sum, atom_low_mean, atoms2grid_low_fr, atoms2lip_low = cal_density(u, 
		sel_atom_lower, 'atom_lower', diffnp, edge, box_fr)

	# 计算 primary lipid 密度（根据 cal_ratio）
	primary_lipid_up_sum, primary_lipid_up_mean = None, None
	primary_lipid_low_sum, primary_lipid_low_mean = None, None
	if cal_ratio:
		primary_lipid_up_sum, primary_lipid_up_mean = cal_density(u, 
			sel_primary_lipid_upper, 'primary_lipid_upper', diffnp, edge, box_fr)
		primary_lipid_low_sum, primary_lipid_low_mean = cal_density(u, 
			sel_primary_lipid_lower, 'primary_lipid_lower', diffnp, edge, box_fr)

	# 返回每一帧的结果
	return atom_up_sum, atom_up_mean, atom_low_sum, atom_low_mean, \
		   primary_lipid_up_sum, primary_lipid_low_sum, \
		   atoms2grid_up_fr, atoms2lip_up, atoms2grid_low_fr, atoms2lip_low
def process_block(pdb, trj, start_block, end_block, lip_leaflet_raw, primary_lipid_id, edge, cal_ratio):
	"""
	处理一个 n_gap 范围块的任务，并返回所有需要的结果
	"""
	lips_grid_ndx_block = []
	h_atom_up_mean_all = []
	h_atom_low_mean_all = []
	h_atom_up_sum_all = []
	h_atom_low_sum_all = []
	h_primary_lipid_up_sum_all = []
	h_primary_lipid_low_sum_all = []
# 创建一个进度条对象，范围是从 start_block 到 end_block
	block_progress = tqdm(range(start_block, end_block, n_gap), desc="Processing Blocks", leave=True, position=0)
	for b in block_progress:
		e = min(b + n_gap, end_block)  # 确保不超出范围
		fr5_leaflets = lip_leaflet_raw[b:e, :]
		most_common_leaflet_tag = [Counter(column).most_common(1)[0][0] for column in zip(*fr5_leaflets)]
		upper_resid_list = [str(index+1) for index, value in enumerate(most_common_leaflet_tag) if value == 0]
		lower_resid_list = [str(index+1) for index, value in enumerate(most_common_leaflet_tag) if value == 1]
		sel_atom_upper = 'resid ' + ' '.join(upper_resid_list)
		sel_atom_lower = 'resid ' + ' '.join(lower_resid_list)
		upper_primary_lipid_list = [str(x) for x in sorted(list(set([int(idx) for idx in primary_lipid_id]) & set([int(idx) for idx in upper_resid_list])))]
		lower_primary_lipid_list = [str(x) for x in sorted(list(set([int(idx) for idx in primary_lipid_id]) & set([int(idx) for idx in lower_resid_list])))]
		sel_primary_lipid_upper = 'resid ' + ' '.join([str(x) for x in upper_primary_lipid_list])
		# print(sel_primary_lipid_upper)
		sel_primary_lipid_lower = 'resid ' + ' '.join([str(x) for x in lower_primary_lipid_list])
		# if(b == 9500):
		#     print(sel_atom_upper)
		#     print(sel_primary_lipid_upper)
		# 创建一个帧级别的进度条对象
		frame_indices = list(range(b, e))
		frame_progress = tqdm(frame_indices, desc=f"Frames {b}-{e}", leave=False, position=1)
		# 串行处理每一帧
		results = [
			process_frame(pdb, trj, frame, sel_atom_upper, sel_atom_lower,
						  sel_primary_lipid_upper, sel_primary_lipid_lower, edge, cal_ratio)
			for frame in frame_progress
		]

		atoms2lip_up = results[0][7]
		atoms2lip_low = results[0][9]

		for i, (atoms2grid_up_fr, atoms2grid_low_fr) in enumerate(zip([r[6] for r in results], [r[8] for r in results])):
			if i == 0:
				atoms2grid_up = deepcopy(atoms2grid_up_fr)
				atoms2grid_low = deepcopy(atoms2grid_low_fr)
			else:
				atoms2grid_up = [raw + new for raw, new in zip(atoms2grid_up, atoms2grid_up_fr)]
				atoms2grid_low = [raw + new for raw, new in zip(atoms2grid_low, atoms2grid_low_fr)]

		lip2grid_up = get_lip_grid_ndx(atoms2grid_up, atoms2lip_up, 0)
		lip2grid_low = get_lip_grid_ndx(atoms2grid_low, atoms2lip_low, 1)
		all_lips2grid_ndx = {**lip2grid_up, **lip2grid_low}
		sorted_keys = sorted(all_lips2grid_ndx.keys())
		sorted_grid_ndx = [all_lips2grid_ndx[key] for key in sorted_keys]
		lips_grid_ndx_block.append(sorted_grid_ndx)

		# 添加到结果列表
		h_atom_up_mean = np.mean([r[1] for r in results], axis=0)
		h_atom_low_mean = np.mean([r[3] for r in results], axis=0)
		h_atom_up_mean_all.append(h_atom_up_mean)
		h_atom_low_mean_all.append(h_atom_low_mean)
		if cal_ratio:
			h_atom_up_sum = np.mean([r[0] for r in results], axis=0)
			h_atom_low_sum = np.mean([r[2] for r in results], axis=0)
			h_primary_lipid_up_sum = np.mean([r[4] for r in results], axis=0)
			h_primary_lipid_low_sum = np.mean([r[5] for r in results], axis=0)
			h_atom_up_sum_all.append(h_atom_up_sum)
			h_atom_low_sum_all.append(h_atom_low_sum)
			h_primary_lipid_up_sum_all.append(h_primary_lipid_up_sum)
			h_primary_lipid_low_sum_all.append(h_primary_lipid_low_sum)

	# 返回块的所有结果
	return lips_grid_ndx_block, h_atom_up_mean_all, h_atom_low_mean_all, \
		   h_atom_up_sum_all, h_atom_low_sum_all, h_primary_lipid_up_sum_all, h_primary_lipid_low_sum_all
def process_block_wrapper_with_index(args):
	start_block = args[2]
	result = process_block(*args)
	return start_block, result

u = MDAnalysis.Universe(pdb, trj)
box_x = box_y = box_z = 0
for ts in u.trajectory[start:end]:
	box = ts.dimensions[:3]
	box_x = max(box_x, box[0])
	box_y = max(box_y, box[1])
	box_z = max(box_z, box[2])
max_box = (box_x, box_y, box_z)
x_bins = int(np.ceil(max_box[0] / bin_width))
y_bins = int(np.ceil(max_box[1] / bin_width))
z_bins = int(np.ceil(max_box[2] / 0.5))
x_edges = np.linspace(0, 0 + x_bins * bin_width, x_bins + 1)
y_edges = np.linspace(0, 0 + y_bins * bin_width, y_bins + 1)
z_edges = np.linspace(0, 0 + z_bins * 0.5, z_bins + 1)
edge3d = (x_edges, y_edges, z_edges)

lip_leaflet_raw = np.loadtxt(fn_leaflet)[:, 1:]

num_blocks = 10 
block_size = (end - start) // num_blocks
blocks = [(start + i * block_size, start + (i + 1) * block_size) for i in range(num_blocks)]
blocks[-1] = (blocks[-1][0], end)

# 外层并行化
with Pool(processes=num_blocks) as pool:
	all_results = list(tqdm(
		pool.imap_unordered(
			process_block_wrapper_with_index,
			[(pdb, trj, start_block, end_block, lip_leaflet_raw, primary_lipid_id, edge3d, cal_ratio)
			 for start_block, end_block in blocks]
		),
		total=len(blocks),
		desc="Processing Blocks"
	))
# 按起始索引排序
all_results.sort(key=lambda x: x[0])  # 按 start_block 排序
# 提取排序后的结果
all_results = [result for _, result in all_results]
# 合并所有块的结果
lips_grid_ndx = [item for block_result in all_results for item in block_result[0]]
h_atom_up_mean_all = [item for block_result in all_results for item in block_result[1]]
h_atom_low_mean_all = [item for block_result in all_results for item in block_result[2]]

if cal_ratio:
	h_atom_up_sum_all = [item for block_result in all_results for item in block_result[3]]
	h_atom_low_sum_all = [item for block_result in all_results for item in block_result[4]]
	h_primary_lipid_up_sum_all = [item for block_result in all_results for item in block_result[5]]
	h_primary_lipid_low_sum_all = [item for block_result in all_results for item in block_result[6]]
print('End of density calculation')
tag_matrixs_up, tag_matrixs_low = cluster_all(h_atom_up_mean_all, h_atom_low_mean_all,\
											  iterations=0, param=param, \
												cal_ratio = cal_ratio)
lips_tag = get_lipis_tags(tag_matrixs_up, tag_matrixs_low, lips_grid_ndx)
lips_tag = np.array(lips_tag)
lips_tag_smoothed = temporal_smoothing(lips_tag)
lips_tag_smoothed = np.array(lips_tag_smoothed)
e_list_up = edge3d
print('end of phase identification')
print('shape of lips_tag:\t',lips_tag.shape)
print('---------------------------------\n')
outfn_raw = outpath+sys+'-rawdata.xvg'
outfn_upper_phase1 = outpath+sys+'-upper-phase1.xvg'
outfn_upper_phase2 = outpath+sys+'-upper-phase2.xvg'
outfn_lower_phase1 = outpath+sys+'-lower-phase1.xvg'
outfn_lower_phase2 = outpath+sys+'-lower-phase2.xvg'
outf_raw = open(outfn_raw, 'w')
outf_upper_phase1 = open(outfn_upper_phase1, 'w')
outf_upper_phase2 = open(outfn_upper_phase2, 'w')
outf_lower_phase1 = open(outfn_lower_phase1, 'w')
outf_lower_phase2 = open(outfn_lower_phase2, 'w')

lip_leaflet = []
for i in range(start,end,n_gap):
	fn_leaflets = lip_leaflet_raw[i:i+n_gap, :]
	most_common_leaflet_tag = [Counter(column).most_common(1)[0][0] for column in zip(*fn_leaflets)]
	lip_leaflet.append(most_common_leaflet_tag)
lip_leaflet = np.array(lip_leaflet)

for i in range(0, len(lips_tag_smoothed)):
	print('%d' % (start + 5*i),file=outf_raw,end=' ')
	for j in range(0, len(lips_tag_smoothed[0])):
		print('%d' % (lips_tag_smoothed[i][j]),file= outf_raw,end=' ')
	print('\n',file= outf_raw,end='')

for i in range(0, len(lips_tag_smoothed)):
	print('%d' % (start + 5*i),file=outf_upper_phase1,end=' ')
	print('%d' % (start + 5*i),file=outf_upper_phase2,end=' ')
	print('%d' % (start + 5*i),file=outf_lower_phase1,end=' ')
	print('%d' % (start + 5*i),file=outf_lower_phase2,end=' ')
	for j in range(0, len(lips_tag_smoothed[0])):
		if lips_tag_smoothed[i][j] == 0 and (lip_leaflet[i, j] == 0):
			print('%d' % (j+1),file=outf_upper_phase1,end=' ')
		elif lips_tag_smoothed[i][j] == 1 and (lip_leaflet[i, j] == 0):
			print('%d' % (j+1),file=outf_upper_phase2,end=' ')
		elif lips_tag_smoothed[i][j] == 0 and (lip_leaflet[i, j] == 1):
			print('%d' % (j+1), file=outf_lower_phase1, end=' ')
		elif lips_tag_smoothed[i][j] == 1 and (lip_leaflet[i, j] == 1):
			print('%d' % (j+1),file=outf_lower_phase2,end=' ')
	print( '\n',file=outf_upper_phase1,end='')
	print( '\n',file=outf_upper_phase2,end='')
	print( '\n',file=outf_lower_phase1,end='')
	print( '\n',file=outf_lower_phase2,end='')

