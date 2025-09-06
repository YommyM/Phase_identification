# sys = 'dpdo280k'
# trj = "/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/traj-dpdo280k-0-10us-pbc.xtc"
# pdb = "/data/gulab/yzdai/dyz_project1/data/dppc_dopc_280k/dppc_dopc_280k_0us.gro"
# fn_leaflet = '/data/gulab/yzdai/data4/phase_identification/leaflet/dpdo280k-leaflet.xvg'
# start = 9000
# end = 10000
# primary_lipid_id = list(range(1, 347)) + list(range(577, 923))
# primary_lipid_id = [str(x) for x in primary_lipid_id]
# chol_id = None



sys='psmdopochl'
pdb="/data/gulab/yzdai/dyz_project1/data/psmdopochl/psmdopochl-rho0.8.gro"
trj="/data/gulab/yzdai/dyz_project1/data/psmdopochl/trjcat-psmdopochl-rho0.8-1ns-1-22.xtc"
fn_leaflet='/data/gulab/yzdai/data4/phase_identification/leaflet/psmdopochl300k-0.8-0-20us-leaflet.xvg'
start = 19000
end = 20000
primary_lipid_id = list(range(1, 91)) + list(range(91, 181))
primary_lipid_id = [str(x) for x in primary_lipid_id]

psm_id = list(range(1,181))
dopc_id = list(range(181, 325))
chol_id = list(range(325, 477))
popc_id = list(range(477, 513))

cal_ratio = False
n_gap = 5
bin_width = 2
outpath = '/data/gulab/yzdai/data4/phase_identification/phase_out/' + sys + '/' + str(start) + '-' + str(end) + '/'
param = "None"


import matplotlib
from joblib import Parallel, delayed
import MDAnalysis
import seaborn as sns
import json
import os
from scipy.stats import norm
import argparse
from sklearn.cluster import KMeans
from matplotlib.ticker import LinearLocator
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from multiprocessing import Pool
from tqdm import tqdm

def remove_isolated_points_and_smooth(labels, iter):
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
def plot_phase(edge_list, cluster_matrix):
	 # Make figure with regular contour plot 
	levels = LinearLocator(numticks=16).tick_values(0, 1)      #这里需要改
	cmap = plt.get_cmap('RdYlBu')
	xMatrix = np.zeros((cluster_matrix.shape[0], cluster_matrix.shape[1]))
	yMatrix = np.zeros((cluster_matrix.shape[0], cluster_matrix.shape[1]))
	xaxis = edge_list[0][1:]
	yaxis = edge_list[1][1:]
	for i in range(0, cluster_matrix.shape[0]):
		xMatrix[i, :] = xaxis
	for j in range(0, cluster_matrix.shape[1]):
		yMatrix[:, j] = yaxis

	im = plt.contourf(yMatrix,
			 xMatrix, cluster_matrix, levels=levels,
			 cmap=cmap)
	densmap = plt.gca()
	densmap.set_aspect(1)
	#set border width
	densmap.spines['bottom'].set_linewidth(2)
	densmap.spines['left'].set_linewidth(2)
	densmap.spines['top'].set_linewidth(2)
	densmap.spines['right'].set_linewidth(2)
	densmap.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
	return im
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

	volume = bin_width*bin_width*0.5
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
def cluster_all(atom_dens_mean_up_all_raw, atom_dens_mean_low_all_raw,
				k=2, iterations=1, gmm_param=None):
	if gmm_param != 'None':
		# Step 1: 读取 JSON 文件
		with open(gmm_param, 'r') as json_file:
			params = json.load(json_file)
		# 提取 Ld 和 Lo 的参数
		density_mean = params['density']['mean']
	else:
		density_mean = np.mean(atom_dens_mean_up_all_raw)/2+np.mean(atom_dens_mean_low_all_raw)/2
	# 数据处理，空间上均值滤波
	density_up_filtered_all = []
	density_low_filtered_all = []
	for n_fr in range(atom_dens_mean_up_all_raw.shape[0]):
		density_mean_up_fr = atom_dens_mean_up_all_raw[n_fr]
		density_mean_low_fr = atom_dens_mean_low_all_raw[n_fr]
		if np.any(density_mean_up_fr == 0):
			density_mean_up_fr = replace_zeros_with_neighbors_mean(density_mean_up_fr)
		if np.any(density_mean_low_fr == 0):
			density_mean_low_fr = replace_zeros_with_neighbors_mean(density_mean_low_fr)
		# 对原始数值进行空间上均值滤波
		density_up_normed_fr = density_mean_up_fr/density_mean
		density_low_normed_fr = density_mean_low_fr/density_mean
		density_up_filtered_fr = spatial_smooth(density_up_normed_fr)
		density_low_filtered_fr = spatial_smooth(density_low_normed_fr)

		density_up_filtered_all.append(density_up_filtered_fr)
		density_low_filtered_all.append(density_low_filtered_fr)
	density_up_filtered_all = np.array(density_up_filtered_all)
	density_low_filtered_all = np.array(density_low_filtered_all)
	# 将数组转换为2D形式，以便进行KMeans聚类
	n_leaflet_samples = atom_dens_mean_up_all_raw.shape[0]*atom_dens_mean_up_all_raw.shape[1]*atom_dens_mean_up_all_raw.shape[2]
	density_up_flattened = density_up_filtered_all.reshape(n_leaflet_samples, -1).tolist()
	density_low_flattened = density_low_filtered_all.reshape(n_leaflet_samples, -1).tolist()
	density_flattened = density_up_flattened + density_low_flattened
	# [atom]
	flattened_matrix = density_flattened
	# 去除NA
	flattened_matrix = np.where(np.isnan(flattened_matrix), 0, flattened_matrix)
	# 检查 NaN
	has_nan = np.isnan(flattened_matrix).any()
	if has_nan:
		print("Array contains NaN values.")
	# 检查 Infinity
	has_inf = np.isinf(flattened_matrix).any()
	if has_inf:
		print("Array contains infinite values.")
	###画分布图###
	plot_density_distr(flattened_matrix, title=sys+' '+str(start)+'-'+str(end))
	# 开始分类
	if (gmm_param != 'None'):
		n_fr  = density_up_filtered_all.shape[0]
		tags_up_all = []; tags_low_all = []
		for fr in range(n_fr):
			filter_fr = threshold_max - (threshold_max - threshold_min) * fr/n_fr
			density_up_filtered_fr = density_up_filtered_all[fr]
			tags_up_fr = (density_up_filtered_fr > filter_fr).astype(int)
			tags_up_all.append(tags_up_fr)
			density_low_filtered_fr = density_low_filtered_all[fr]
			tags_low_fr = (density_low_filtered_fr > filter_fr).astype(int)
			tags_low_all.append(tags_low_fr)
		tags_up_flattened = np.array(tags_up_all).reshape(n_leaflet_samples, -1).tolist()
		tags_low_flattened = np.array(tags_low_all).reshape(n_leaflet_samples, -1).tolist()
		tags = tags_up_flattened + tags_low_flattened
		tags = np.array(tags)

	if (gmm_param == 'None'):
		# Step 1: 高斯混合模型拟合
		gmm = GaussianMixture(n_components=k, random_state=0).fit(flattened_matrix)   
		raw_tags = gmm.predict(flattened_matrix)

		# Step 2: 对 raw_tags 进行处理，得到最终标签 tags
		tags = get_tag_by_first_feature(flattened_matrix,raw_tags)

		# Step 3: 根据 tags 提取 Lo 和 Ld 的分布参数
		ld_indices = (tags == 0)  # Ld 标签为 0
		lo_indices = (tags == 1)  # Lo 标签为 1
		# 提取 Ld 和 Lo 样本
		ld_samples = flattened_matrix[ld_indices]
		lo_samples = flattened_matrix[lo_indices]
		# 计算 Ld 和 Lo 的均值和标准差
		mu_ld, sigma_ld = ld_samples.mean(), ld_samples.std()
		mu_lo, sigma_lo = lo_samples.mean(), lo_samples.std()
		min_lo = min(flattened_matrix[tags == 1])
		max_ld = max(flattened_matrix[tags == 0])
		# Step 4: 保存到 JSON 文件
		output_data = {
			"Ld": {
				"tag": 0,
				"mean": float(mu_ld),
				"std": float(sigma_ld),
				"max density": float(max_ld)
			},
			"Lo": {
				"tag": 1,
				"mean": float(mu_lo),
				"std": float(sigma_lo),
				"min density": float(min_lo)
			},
			"density": {
				"mean": float(density_mean)
			}
		}
		with open(os.path.join(os.path.dirname(os.path.abspath(outpath)), 'gmm_parameters.json'), 'w') as json_file:
			json.dump(output_data, json_file, indent=4)
	# else:
		####方法1：看哪个概率密度大
		# Step 1: 读取 JSON 文件
		# with open(gmm_param, 'r') as json_file:
		#     params = json.load(json_file)
		# # 提取 Ld 和 Lo 的参数
		# mu_ld = params['Ld']['mean']
		# sigma_ld = params['Ld']['std']
		# mu_lo = params['Lo']['mean']
		# sigma_lo = params['Lo']['std']
		# mu_ubld = params['unbalanced_Ld']['mean']
		# sigma_ubld = params['unbalanced_Ld']['std']
		# # Step 2: 计算每个样本的概率密度
		# prob_ld_1 = norm.pdf(flattened_matrix, loc=mu_ld, scale=sigma_ld)  # 属于 Ld 的概率密度
		# prob_ld_2 = norm.pdf(flattened_matrix, loc=mu_ubld, scale=sigma_ubld)  # 属于 Ld 的概率密度
		# prob_ld = np.maximum(prob_ld_1, prob_ld_2)
		# prob_lo = norm.pdf(flattened_matrix, loc=mu_lo, scale=sigma_lo)  # 属于 Lo 的概率密度
		# # Step 3: 分类规则
		# tags = np.where(prob_lo > prob_ld, 1, 
		#         np.where(prob_lo < prob_ld, 0, np.nan))
		
		# ####方法2：单侧检验
		# mu_lo = params['Lo']['mean']
		# sigma_lo = params['Lo']['std']
		# z_score = norm.ppf(0.80)  # 单侧 90% 显著性水平对应的 Z 值（可根据需求调整）
		# # 计算特征值较大的阈值
		# threshold = mu_lo - z_score * sigma_lo  # 比较大的样本特征值阈值
		# print(f"分类阈值 (80% 左单侧显著): {threshold:.3f}")
		# tags = np.where(flattened_matrix > threshold, 1, 0)

		# ####方法3：动态threshold
		# tags = np.where(flattened_matrix > threshold, 1, 0)
		# print('Threshold:\t',threshold)
	print('Lo min:\t',min(flattened_matrix[tags == 1]))
	print('Ld max:\t',flattened_matrix[tags == 0].max())
	plot_3density_distr(flattened_matrix,
					flattened_matrix[tags == 0], 
					flattened_matrix[tags == 1], 
					title=sys+' '+str(start)+'-'+str(end))
	# 将标签数组重新分割为tag_up和tag_low，并还原其形状
	split_idx = len(tags) // 2
	tag_matrixs_up = tags[:split_idx].reshape(atom_dens_mean_up_all_raw.shape[0],atom_dens_mean_up_all_raw.shape[1],atom_dens_mean_up_all_raw.shape[2])
	tag_matrixs_low = tags[split_idx:].reshape(atom_dens_mean_up_all_raw.shape[0],atom_dens_mean_up_all_raw.shape[1],atom_dens_mean_up_all_raw.shape[2])
	# 在像素化平面针对grid的tag进行空间平滑
	final_tag_matrixs_up = deepcopy(tag_matrixs_up)
	final_tag_matrixs_low = deepcopy(tag_matrixs_low)
	for n_fr in range(atom_dens_mean_up_all_raw.shape[0]):
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
print(lip_leaflet_raw.shape)

# 将 start 到 end 分成若干块
num_blocks = 10  # 并行块的数量
block_size = (end - start) // num_blocks
blocks = [(start + i * block_size, start + (i + 1) * block_size) for i in range(num_blocks)]
blocks[-1] = (blocks[-1][0], end)  # 确保最后一块包含剩余的帧

def process_block_wrapper_with_index(args):
	start_block = args[2]
	result = process_block(*args)
	return start_block, result

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

# 得到非平衡态的均值和方差
if param != 'None':
	with open(param, 'r') as json_file:
		params = json.load(json_file)
	density_mean = params['density']['mean']
else:
	density_mean = np.mean(h_atom_up_mean_all)/2+np.mean(h_atom_low_mean_all)/2
	# density_mean = 0.3492629423089485
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

n_t, n_x, n_y = h_up_all.shape

if(cal_ratio):
	h_ratio_up_all_filtered = np.array(h_ratio_up_all_filtered)
	h_ratio_low_all_filtered = np.array(h_ratio_low_all_filtered)

n_leaflet_samples = n_t*n_x*n_y

density_up_flattened = h_up_all.reshape(n_leaflet_samples, -1).tolist()
density_low_flattened = h_low_all.reshape(n_leaflet_samples, -1).tolist()
density_flattened = density_up_flattened + density_low_flattened
if(cal_ratio):
	r_up_flattened = h_ratio_up_all_filtered.reshape(n_leaflet_samples, -1).tolist()
	r_low_flattened = h_ratio_low_all_filtered.reshape(n_leaflet_samples, -1).tolist()
	r_flattened = r_up_flattened + r_low_flattened

# [atom]
if(cal_ratio):
	flattened_matrix = np.concatenate((density_flattened, r_flattened), axis=1)
else:
	flattened_matrix = np.array(deepcopy(density_flattened)).reshape(-1,1)

file_path = '/data/gulab/yzdai/data4/phase_identification/plot/input/last1us/'+sys+'_bin'+str(bin_width)+'.xvg'
# 将 flattened_matrix 写入文件
with open(file_path, 'w') as file:
	if cal_ratio:
		for density, ratio in flattened_matrix:
			print(density, ratio, file=file)
	else:
		for density in flattened_matrix:
			print(str(float(density)), file=file)
print(f"Data written to {file_path}")










