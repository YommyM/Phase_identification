#考虑周期性边界条件的Voronoi tessellation
import MDAnalysis
from scipy.spatial import Voronoi
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
parser = argparse.ArgumentParser()
parser.add_argument('-trj', type = str, help='name of trajectory')
parser.add_argument('-pdb', type = str, help = 'file name of system structure, e.g., pdb, gro, psf, etc...')
parser.add_argument('-start', type = int, help='begin frame number')
parser.add_argument('-end', type = int, help='end frame number')
parser.add_argument('-n_gap', type = int, help='The interval of plot')
parser.add_argument('-sys', type = str, help = 'name of system')
parser.add_argument('-leaflet', type = str, help = 'name of leaflet path')
parser.add_argument('-phasepath', type = str, help = 'name of phase path')
parser.add_argument('-HMMphasepath', type = str, help = 'name of HMM phase path')
parser.add_argument('-voronoi_out', type = str, help = 'name of output path')
args = parser.parse_args()

sys = args.sys
pdb = args.pdb
trj = args.trj
start = args.start
end = args.end
fn_leaflet = args.leaflet
atom_phase_fn = args.phasepath
HMM_phase_fn = args.HMMphasepath
n_gap = args.n_gap
voronoi_out = args.voronoi_out

def get_sn(sel_lip, sel_lip_type):
    sel_atom =[]
    for i in range(len(sel_lip)):
        if(sel_lip_type[i] == 'DPPC'):
            tmp_sn1 = sel_lip[i] + ' and (name C32 H2X H2Y C33 H3X H3Y C34 H4X H4Y C35 H5X H5Y C36 H6X H6Y C37 H7X H7Y\
                                      C38 H8X H8Y C39 H9X H9Y C310 H10X H10Y C311 H11X H11Y C312 H12X H12Y C313 H13X H13Y\
                                      C314 H14X H14Y C315 H15X H15Y C316 H16X H16Y H16Z)'
            tmp_sn2 = sel_lip[i] + ' and (name C22 H2R H2S C23 H3R H3S C24 H4R H4S C25 H5R H5S C26 H6R H6S C27 H7R H7S\
                                      C28 H8R H8S C29 H9R H9S C210 H10R H10S C211 H11R H11S C212 H12R H12S C213 H13R H13S\
                                      C214 H14R H14S C215 H15R H15S C216 H16R H16S H16T)'
        elif(sel_lip_type[i] == 'DOPC'):
            tmp_sn1 = sel_lip[i] + ' and (name C32 H2X H2Y C33 H3X H3Y C34 H4X H4Y C35 H5X H5Y C36 H6X H6Y C37 H7X H7Y\
                                      C38 H8X H8Y C39 H9X C310 H10X C311 H11X H11Y C312 H12X H12Y C313 H13X H13Y\
                                      C314 H14X H14Y C315 H15X H15Y C316 H16X H16Y C317 H17X H17Y C318 H18X H18Y H18Z)'
            tmp_sn2 = sel_lip[i] + ' and (name C22 H2R H2S C23 H3R H3S C24 H4R H4S C25 H5R H5S C26 H6R H6S C27 H7R H7S\
                                      C28 H8R H8S C29 H9R C210 H10R C211 H11R H11S C212 H12R H12S C213 H13R H13S\
                                      C214 H14R H14S C215 H15R H15S C216 H16R H16S C217 H17R H17S C218 H18R H18S H18T)' 
        elif(sel_lip_type[i] == 'POPC'):
            tmp_sn1 = sel_lip[i] + ' and (name C32 H2X H2Y C33 H3X H3Y C34 H4X H4Y C35 H5X H5Y C36 H6X H6Y C37 H7X H7Y\
                                      C38 H8X H8Y C39 H9X H9Y C310 H10X H10Y C311 H11X H11Y C312 H12X H12Y C313 H13X H13Y\
                                      C314 H14X H14Y C315 H15X H15Y C316 H16X H16Y H16Z)'
            tmp_sn2 = sel_lip[i] + ' and (name C22 H2R H2S C23 H3R H3S C24 H4R H4S C25 H5R H5S C26 H6R H6S C27 H7R H7S\
                                      C28 H8R H8S C29 H91 C210 H101 C211 H11R H11S C212 H12R H12S C213 H13R H13S\
                                      C214 H14R H14S C215 H15R H15S C216 H16R H16S C217 H17R H17S C218 H18R H18S H18T)'
        elif(sel_lip_type[i] == 'PSM'):
            tmp_sn1 = sel_lip[i] + ' and (name C2F H2F C3F H3F C4F H4F C5F H5F C6F H6F C7F H7F \
                                      C8F H8F C9F H9F C10F H10F C11F H11F C12F H12F C13F H13F \
                                      C14F H14F C15F H15F C16F H16F C2F H2G C3F H3G C4F H4G C5F H5G C6F H6G C7F H7G \
                                       C8F H8G C9F H9G C10F H10G C11F H11G C12F H12G C13F H13G \
                                       C14F H14G C15F H15G C16F H16G H16H)'
            tmp_sn2 = sel_lip[i] + ' and (name C6S H6S C7S H7S C8S H8S C9S H9S C10S H10S C11S H11S C12S H12S C13S H13S \
                                      C14S H14S C15S H15S C16S H16S C17S H17S C18S H18S C6S H6T C7S H7T C8S H8T C9S H9T C10S H10T C11S H11T C12S H12T C13S H13T \
                                      C14S H14T C15S H15T C16S H16T C17S H17T C18S H18T H18U)'
        sel_atom.append(tmp_sn1)
        sel_atom.append(tmp_sn2)
    return sel_atom
def getPolygonArea(points):
    sizep = len(points)
    if sizep<3:
        return 0.0
    area = points[-1][0] * points[0][1] - points[0][0] * points[-1][1]
    for i in range(1, sizep):
        v = i - 1
        area += (points[v][0] * points[i][1])
        area -= (points[i][0] * points[v][1])
    return abs(0.5 * area)
def calculate_lipid_Voronoi_area(points):

    vor = Voronoi(points)
    sorted_regions = [vor.regions[vor.point_region[i]] for i in range(len(points))]

    areas = []
    boundary_points = []
    for region in sorted_regions:
        if -1 in region or len(region) == 0:

            areas.append(0)
            boundary_points.append(list(points[i]))
            continue
        # 获取区域的顶点坐标
        vertices = vor.vertices[region]
        # 计算区域的凸包
        area=getPolygonArea(vertices)
        # print('area',area)
        areas.append(area)
    # 检查
    if (len(areas) == len(points)):
        print('                     V_AREAS NUM CORRECT')
    else:
        print('!                    V_AREAS NUM ERROR')

    return areas, vor, sorted_regions, boundary_points
def plot_Voronoi(ax, vor, sorted_regions, points, n_count, box, phase): #n_count = lip_num*2 + chol_num in one leaflet

    for i in range(len(sorted_regions)):
        region = sorted_regions[i]
        if -1 in region or len(region) == 0:
            continue

        vertices = vor.vertices[region]
        if(i < len(phase)):
            if(phase[i] == 1):
                ax.fill(*zip(*vertices), alpha=0.2, color='blue')
            elif(phase[i] == 0):
                ax.fill(*zip(*vertices), alpha=0.2, color='red')
        else:
            ax.fill(*zip(*vertices), alpha=0.0, color='white')      

    if ('dpdochl' in sys):
        ax.plot(points[:404, 0], points[:404, 1], 'b.', markersize='2')          # dppc
        ax.plot(points[404:808, 0], points[404:808, 1], 'r.', markersize='2')    # dopc
        ax.plot(points[808:n_count, 0], points[808:n_count, 1], 'y.', markersize='2')    # chol O3 
        ax.plot(points[n_count:, 0], points[n_count:, 1], '.', color='white', markersize='2')            # PBC fake points

        square = plt.Rectangle((0, 0), box[0], box[1], fill=False)
        ax.add_patch(square)
        ax.axis('off')
        for i in range(20, 151, 40):
            ax.plot([0, 2], [i, i], color='black')
            ax.text(-3, i, str(i), ha='right', va='center')  
            ax.plot([i, i], [0, 2], color='black')  
            ax.text(i, -3, str(i), ha='center', va='top') 
    elif('psmdopochl' in sys):
        ax.plot(points[:180, 0], points[:180, 1], 'b.', markersize='2')               # psm 
        ax.plot(points[180:324, 0], points[180:324, 1], 'r.', markersize='2')     # dopc
        ax.plot(points[324:360, 0], points[324:360, 1], 'g.', markersize='2')     # popc 
        ax.plot(points[360:n_count, 0], points[360:n_count, 1], 'y.', markersize='2') # chol O3        
        ax.plot(points[n_count:, 0], points[n_count:, 1], '.', color='white', markersize='2')            # PBC fake points

        square = plt.Rectangle((0, 0), box[0], box[1], fill=False)

        ax.add_patch(square)

        ax.axis('off')

        for i in range(20, 101, 40):

            plt.plot([0, 2], [i, i], color='black')
            plt.text(-3, i, str(i), ha='right', va='center') 

            plt.plot([i, i], [0, 2], color='black') 
            plt.text(i, -3, str(i), ha='center', va='top')  
    x_min = 0-0.1*box[0]
    x_max = 1.1*box[0]
    y_min = 0-0.1*box[1]
    y_max = 1.1*box[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max) 
    return ax
def simulate_PBC(points_leaflet, box): 
    PBC_points = []
    x_min = 0-0.1*box[0]
    x_max = 1.1*box[0]
    y_min = 0-0.1*box[1]
    y_max = 1.1*box[1]
    for i in range(len(points_leaflet)):

        if(points_leaflet[i, 0] > box[0]*0.9):
            x = points_leaflet[i, 0] - box[0]
            y = points_leaflet[i, 1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])

        if(points_leaflet[i, 0] < box[0]*0.1):
            x = points_leaflet[i, 0] + box[0]
            y = points_leaflet[i, 1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])

        if(points_leaflet[i, 1] < box[1]*0.1):
            x = points_leaflet[i, 0]
            y = points_leaflet[i, 1] + box[1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])            

        if(points_leaflet[i, 1] > box[1]*0.9):
            x = points_leaflet[i, 0]
            y = points_leaflet[i, 1] - box[1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])            

        if(points_leaflet[i, 0] > box[0]*0.9 and points_leaflet[i, 1] < box[1]*0.1):
            x = points_leaflet[i, 0] - box[0]
            y = points_leaflet[i, 1] + box[1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])            

        if(points_leaflet[i, 0] > box[0]*0.9 and points_leaflet[i, 1] > box[1]*0.9):
            x = points_leaflet[i, 0] - box[0]
            y = points_leaflet[i, 1] - box[1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])            

        if(points_leaflet[i, 0] < box[0]*0.1 and points_leaflet[i, 1] < box[1]*0.1):
            x = points_leaflet[i, 0] + box[0]
            y = points_leaflet[i, 1] + box[1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])           

        if(points_leaflet[i, 0] < box[0]*0.1 and points_leaflet[i, 1] > box[1]*0.9):
            x = points_leaflet[i, 0] + box[0]
            y = points_leaflet[i, 1] - box[1]
            if(x > x_min and x < x_max and y > y_min and y < y_max):
                PBC_points.append([x, y])           
    PBC_points = np.array(PBC_points)
    return PBC_points
def leaflet_Vor(sel_atom_leaflet, count_leaflet_lip, box):

    points_leaflet = []
    for i in range(0, len(sel_atom_leaflet)):
        point = u.select_atoms(sel_atom_leaflet[i]).center_of_mass()
        points_leaflet.append(list(point[0:2]))
    points_leaflet = np.array(points_leaflet)
    #  n_count == lip_num*2 + chol_num
    n_count = len(points_leaflet)


    PBC_points = simulate_PBC(points_leaflet, box)

    points_with_PBC = np.array(list(points_leaflet) + list(PBC_points))  

    print('PBC_points:\t' + str(len(PBC_points)) + '\tpoints_with_PBC:\t' + str(len(points_with_PBC)))

    areas, vor, sorted_regions, _boundary_points = calculate_lipid_Voronoi_area(points_with_PBC)
    areas = areas[:n_count]
    chol_area = areas[-count_leaflet_lip*2:]
    lip_area = []
    for i in range(0,count_leaflet_lip*2,2):
        if(areas[i]==0 or areas[i+1]==0):
            print('!                    FINAL AREAS ZERO ERROR')
            break
        else:
            tmp = areas[i] + areas[i+1]
            lip_area.append(tmp)
    print('lip_area:\t' + str(len(lip_area)))
    return lip_area, chol_area, vor, sorted_regions, points_with_PBC

if ('dpdochl' in sys):
    count_leaflet_lip = 404 
    sel_lip_type_upper = ['DPPC' for x in range(202)] + ['DOPC' for x in range(202)] 
    sel_lip_type_lower = ['DPPC' for x in range(202)] + ['DOPC' for x in range(202)] 
    lip_upper_ndx = list(np.array(list(range(1,405)))-1)
    sel_lip_upper = list(range(1,405)) 
    for i in range(0,len(sel_lip_upper)):
        sel_lip_upper[i] = 'resid '+str(sel_lip_upper[i])
    lip_lower_ndx = list(np.array(list(range(577,981)))-1)
    sel_lip_lower = list(range(577,981))
    for i in range(0,len(sel_lip_lower)):
        sel_lip_lower[i] = 'resid '+str(sel_lip_lower[i])
    all_chol_id = [int(x) for x in range(405,577)] + [int(x) for x in range(981,1153)]
    if ('280' in sys):
        tit = 'Ternary mixture 280 K'
    if ('290' in sys):
        tit = 'Ternary mixture 290 K'
elif('psmdopochl' in sys):
    count_leaflet_lip = 180  
    sel_lip_type_upper = ['PSM' for x in range(90)] + ['DOPC' for x in range(72)] + ['POPC' for x in range(18)]
    sel_lip_type_lower = ['PSM' for x in range(90)] + ['DOPC' for x in range(72)] + ['POPC' for x in range(18)]

    lip_upper_ndx = list(np.array(list(range(1,91)) + list(range(181,253)) + list(range(477, 495)))-1)
    sel_lip_upper = list(range(1,91)) + list(range(181,253)) + list(range(477, 495))
    for i in range(0,len(sel_lip_upper)):
        sel_lip_upper[i] = 'resid '+str(sel_lip_upper[i])

    lip_lower_ndx = list(np.array(list(range(91, 181)) + list(range(253, 325)) + list(range(495, 513)))-1)
    sel_lip_lower = list(range(91, 181)) + list(range(253, 325)) + list(range(495, 513))
    for i in range(0,len(sel_lip_lower)):
        sel_lip_lower[i] = 'resid '+str(sel_lip_lower[i])
    all_chol_id = [int(x) for x in range(325,401)] + [int(x) for x in range(401,477)]  
    tit = 'Quaternary mixture 300 K'
u = MDAnalysis.Universe(pdb, trj)
lip_leaflet_raw = np.loadtxt(fn_leaflet)[:,1:]
fr=0
for b in range(start, end, n_gap):
    print("fr:", str(fr))
    e = b + n_gap
    fn_leaflets = lip_leaflet_raw[b:e, :]
    most_common_leaflet_tag = [Counter(column).most_common(1)[0][0] for column in zip(*fn_leaflets)]
    chl_resname = 'CHL1'
    sel_upper_chol=[]; sel_lower_chol=[]
    atom_phase_upper_chol=[]; atom_phase_lower_chol=[]
    HMM_phase_upper_chol=[]; HMM_phase_lower_chol=[]

    atom_phase_all = np.loadtxt(atom_phase_fn)[:,1:]
    # if (sum(atom_phase_all[:,0] == 0) > sum(atom_phase_all[:,0] == 1)):
    #     atom_phase_all = 1 - atom_phase_all
    atom_phase_fr = list(atom_phase_all[fr,:])

    HMM_phase_all = np.loadtxt(HMM_phase_fn)[:,1:]
    if (sum(HMM_phase_all[:,0] == 0) > sum(HMM_phase_all[:,0] == 1)):
        HMM_phase_all = 1 - HMM_phase_all
    HMM_phase_fr = list(HMM_phase_all[fr,:])

    for chol_id in all_chol_id:
        index = chol_id - 1
        leaflet_tag = most_common_leaflet_tag[index]
        atom_phase_tag = atom_phase_fr[index]
        HMM_phase_tag = HMM_phase_fr[index]
        if leaflet_tag == 0:
            sel_upper_chol.append(chol_id)
            atom_phase_upper_chol.append(atom_phase_tag)
            HMM_phase_upper_chol.append(HMM_phase_tag)
        else:
            sel_lower_chol.append(chol_id)
            atom_phase_lower_chol.append(atom_phase_tag)
            HMM_phase_lower_chol.append(HMM_phase_tag)

    atom_phase_upper_lip = [item for item in np.array(atom_phase_fr)[lip_upper_ndx] for _ in range(2)]
    atom_phase_lower_lip = [item for item in np.array(atom_phase_fr)[lip_lower_ndx] for _ in range(2)]
    HMM_phase_upper_lip = [item for item in np.array(HMM_phase_fr)[lip_upper_ndx] for _ in range(2)]
    HMM_phase_lower_lip = [item for item in np.array(HMM_phase_fr)[lip_lower_ndx] for _ in range(2)]

    atom_phase_of_points_up = atom_phase_upper_lip + atom_phase_upper_chol
    atom_phase_of_points_low = atom_phase_lower_lip + atom_phase_lower_chol
    HMM_phase_of_points_up = HMM_phase_upper_lip + HMM_phase_upper_chol
    HMM_phase_of_points_low = HMM_phase_lower_lip + HMM_phase_lower_chol
    count_upper_chol = len(sel_upper_chol)
    count_lower_chol = len(sel_lower_chol)

    sel_atom_upper = get_sn(sel_lip_upper, sel_lip_type_upper)
    sel_atom_lower = get_sn(sel_lip_lower, sel_lip_type_lower)

    for i in range(count_upper_chol):
        tmp = 'resname ' + chl_resname + ' and resid ' + str(sel_upper_chol[i]) + ' '# + 'and name ' + chl_atomname
        sel_atom_upper.append(tmp)
    for i in range(count_lower_chol):
        tmp = 'resname ' + chl_resname + ' and resid ' + str(sel_lower_chol[i]) + ' '# + ' and name ' + chl_atomname
        sel_atom_lower.append(tmp)
    print('upper atoms:\t' + str(len(sel_atom_upper)) + '\tlower atoms:\t' + str(len(sel_atom_lower)))
    n_count_up =  count_leaflet_lip*2 + count_upper_chol
    n_count_low = count_leaflet_lip*2 + count_lower_chol
    for ts in u.trajectory[b:e:n_gap]:
        box = ts.dimensions[:3]
        print('upper--------------------------------------------------')
        lip_area_up, chol_area_up, vor_up, sorted_regions_up, points_all_up = leaflet_Vor(sel_atom_upper, count_leaflet_lip, box)
        print('lower--------------------------------------------------')
        lip_area_low, chol_area_low, vor_low, sorted_regions_low, points_all_low = leaflet_Vor(sel_atom_lower, count_leaflet_lip, box)
        print('-------------------------------------------------------')

        fig1 = plt.figure(figsize=(6, 6))#
        gs = matplotlib.gridspec.GridSpec(3, 3, width_ratios=[0.1, 1, 1,], height_ratios=[1, 1,0.1], wspace=0.03, hspace=0)

        for i, title in enumerate(['Upper leaflet','Lower leaflet']):
            ax = fig1.add_subplot(gs[i, 0]) 
            ax.axis('off') 
            ax.text(0.5, 0.5, title, ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize = 15)

        for i, title in enumerate(['This method', 'HMM method']):
            ax = fig1.add_subplot(gs[2, i+1]) 
            ax.axis('off')
            ax.text(0.5, 0.5, title, ha='center', va='center', transform=ax.transAxes, fontsize = 15) 
        ax = plt.subplot(gs[0, 1])
        im = plot_Voronoi(ax, vor_up, sorted_regions_up, points_all_up, n_count_up, box, atom_phase_of_points_up)
        ax.set_aspect('equal', 'box') 

        ax = plt.subplot(gs[1, 1])
        im = plot_Voronoi(ax, vor_low, sorted_regions_low, points_all_low, n_count_low, box, atom_phase_of_points_low)
        ax.set_aspect('equal', 'box') 

        ax = plt.subplot(gs[0,2])
        im = plot_Voronoi(ax, vor_up, sorted_regions_up, points_all_up, n_count_up, box, HMM_phase_of_points_up)  
        ax.set_aspect('equal', 'box') 
        
        ax = plt.subplot(gs[1,2])
        im = plot_Voronoi(ax, vor_low, sorted_regions_low, points_all_low, n_count_low, box, HMM_phase_of_points_low)  
        ax.set_aspect('equal', 'box')  
        
        fig1.suptitle(tit + ' ts = ' + str(b) +' ns', fontsize=20, fontweight='bold')

        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
        plt.savefig(
        voronoi_out+sys+'-'+str(b)+'.png',     
        dpi=350,         
        format='png',     
        bbox_inches='tight', 
        pad_inches=0.0,     
        # facecolor='white',  
        # edgecolor='black', 
        transparent=False,  
        ) 
        plt.close()
    fr += 1