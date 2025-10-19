import MDAnalysis
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import LinearLocator
parser = argparse.ArgumentParser()
parser.add_argument('-trj', type = str, help='name of trajectory')
parser.add_argument('-pdb', type = str, help = 'file name of system structure, e.g., pdb, gro, psf, etc...')
parser.add_argument('-b', type = int, help='begin frame number')
parser.add_argument('-e', type = int, help='end frame number')
parser.add_argument('-inte', type = int, help='The interval of plot')
parser.add_argument('-bin_width', type = int, help='bin width (Å)')
parser.add_argument('-sys', type = str, help = 'name of system')
parser.add_argument('-phasepath', type = str, help = 'name of output dictionary')
args = parser.parse_args()

sys = args.sys
pdb = args.pdb
trj = args.trj
b = args.b
e = args.e
phasepath = args.phasepath
inte = args.inte
bins = args.bin_width

up1file = phasepath + sys + "-upper-phase1.xvg"
up2file = phasepath + sys + "-upper-phase2.xvg"
low1file = phasepath + sys + "-lower-phase1.xvg"
low2file = phasepath + sys + "-lower-phase2.xvg"
with open(up1file, 'r') as file:
    up2 = [' '.join(line.strip().split(' ')[1:]) for line in file]
with open(up2file, 'r') as file:
    up1 = [' '.join(line.strip().split(' ')[1:]) for line in file]
with open(low1file, 'r') as file:
    low2 = [' '.join(line.strip().split(' ')[1:]) for line in file]
with open(low2file, 'r') as file:
    low1 = [' '.join(line.strip().split(' ')[1:]) for line in file]

def cal_densmap(sel, bin_x_num, bin_y_num, bin_z_num, box_x_ave, box_y_ave, box_z_ave, bins=0):
    #cal the box size of this frame
    box_fr = u.dimensions[:3]
    #get the coordinates of the selected atoms
    sel_atoms = u.select_atoms(sel).positions
    if(sel !=''):
        #move the com of system to the center of present box
        sel_all_com = u.select_atoms('all').center_of_mass()
        diffx = box_fr[0]/2 - sel_all_com[0]
        diffy = box_fr[1]/2 - sel_all_com[1]
        diffz = box_fr[2]/2 - sel_all_com[2]
        diffnp = np.array([diffx, diffy, diffz])
        sel_atoms_com = np.add(sel_atoms, diffnp)
        #rescale the coordinates between 0 - present boxsize
        #this step is low efficient, modify the code
        sel_atoms_pbc1 = np.where(sel_atoms_com > 0, sel_atoms_com, np.add(sel_atoms_com, box_fr))
        sel_atoms_pbc2 = np.where(sel_atoms_pbc1 < box_fr, sel_atoms_pbc1, np.add(sel_atoms_pbc1, -1*box_fr))
        #rescale the coordinates according to the average boxsize
        scale_np = np.array([box_x_ave / box_fr[0], box_y_ave / box_fr[1], box_z_ave / box_fr[2]])
        sel_atoms_scale = np.multiply(sel_atoms_pbc2, scale_np)
        #histogram the adjusted coordinates
        if bins == 0:
            h_fr, edge_fr = np.histogramdd(sel_atoms_scale, bins=(bin_x_num, bin_y_num, bin_z_num), normed = False)
        else:
            h_fr, edge_fr = np.histogramdd(sel_atoms_scale, bins=bins, normed = False)
    else:
        h_fr, edge_fr = np.histogramdd(np.empty((0, 3)), bins=bins, normed = False)
    #convert histogram to densities 
    volumn = (box_x_ave*box_y_ave*box_z_ave)/(bin_x_num*bin_y_num*bin_z_num)
    dens3d = h_fr / volumn
    #average along z axis to get 2D densmap
    dens2d = np.ndarray(shape = (dens3d.shape[0], dens3d.shape[1]), dtype = float)
    for i in range(0, dens3d.shape[0]):
        for j in range(0, dens3d.shape[1]):
            dens2d[i,j] = dens3d[i, j, :].mean()
    #return values
    return dens2d, edge_fr
def ratio_gel(hist0, hist1):
    #hist0, phase0; hist1, phase1;
    #0 in hist_comp present phase0
    hist_comp = np.zeros(hist0.shape)
    for i in range(0, hist0.shape[0]):
        for j in range(0, hist0.shape[1]):
            if hist0[i, j] > hist1[i, j]:
                hist_comp[i, j] = 0
            elif hist0[i, j] == hist1[i, j]:
                hist_comp[i, j] = np.nan
            else:
                hist_comp[i, j] = 1
    rows, cols = hist_comp.shape
    for i in range(rows):
        for j in range(cols):
            if np.isnan(hist_comp[i, j]): 

                neighbors = [
                    hist_comp[(i-1) % rows, (j-1) % cols],
                    hist_comp[(i-1) % rows, j % cols],
                    hist_comp[(i-1) % rows, (j+1) % cols],
                    hist_comp[i % rows, (j-1) % cols],
                    hist_comp[i % rows, (j+1) % cols],
                    hist_comp[(i+1) % rows, (j-1) % cols],
                    hist_comp[(i+1) % rows, j % cols],
                    hist_comp[(i+1) % rows, (j+1) % cols]
                ]

                counts = {}
                for value in neighbors:
                    if not np.isnan(value):
                        counts[value] = counts.get(value, 0) + 1

                max_count = -1
                most_frequent = None
                for key, count in counts.items():
                    if count > max_count:
                        max_count = count
                        most_frequent = key

                hist_comp[i, j] = most_frequent
    count_p0 = np.sum(hist_comp == 0)
    return hist_comp, float(count_p0)/float(hist0.shape[0]*hist0.shape[1])
    # return hist_comp, count/float(hist0.shape[0]*hist0.shape[1])
def ratio_regi(histu, histl):
    #histu, upper leaflet; histl, lower leaflet;
    #0 both phase0, 1 both phase1, 0.25 upper phase0 lower phase1, 0.75 upper phase1 lower phase0
    hist_comp = np.zeros(histu.shape)
    count = [0, 0, 0, 0] # both phase0, both phase1, upper phase0 lower phase1, lower phase0 upper phase1
    for i in range(0, histu.shape[0]):
        for j in range(0, histu.shape[1]):
            if histu[i, j] == 0 and histl[i, j] == 0:
                hist_comp[i, j] = 0
                count[0]+=1.0
            elif histu[i, j] == 1 and histl[i, j] == 1:
                hist_comp[i, j] = 1
                count[1]+=1.0
            elif histu[i, j] == 0 and histl[i, j] == 1:
                hist_comp[i, j] = 0.25
                count[2]+=1.0
            elif histu[i, j] == 1 and histl[i, j] == 0:
                hist_comp[i, j] = 0.75
                count[3]+=1.0
 
    ratio = np.array(count)/float(histu.shape[0]*histl.shape[1])

    return hist_comp, ratio

def plotDensmap(h, e):
    selectData = h
    for i in range(0, selectData.shape[0]):
        for j in range(0, selectData.shape[1]):
            if selectData[i, j] == 1.00:
                selectData[i, j] = -(1.0/15.0)-0.001
            if selectData[i, j] == 0.75:
                selectData[i, j] = -(3.0/15.0)-0.001
                # selectData[i, j] = -(1.0/15.0)-0.001
            if selectData[i, j] == 0.25:
                # selectData[i, j] = -(1.0/15.0)+0.001
                # selectData[i, j] = -(1.0/15.0)-0.001
                selectData[i, j] = -(12.0/15.0)+0.001
            if selectData[i, j] == 0.00:
                selectData[i, j] = -(14.0/15.0)+0.001
    normData = np.where (selectData <2, selectData, 2)
    normData = 0-normData
    # Make figure with regular contour plot
    levels = LinearLocator(numticks=16).tick_values(0.0, 1.0)
    # levels = LinearLocator(numticks=16).tick_values(-1.0, 0.0)
    cmap = plt.get_cmap('RdBu')
    xMatrix = np.zeros((normData.shape[0], normData.shape[1]))
    yMatrix = np.zeros((normData.shape[0], normData.shape[1]))
    xaxis = e[0][1:]
    yaxis = e[1][1:]
    for i in range(0, normData.shape[0]):
        xMatrix[i, :] = xaxis
    for j in range(0, normData.shape[1]):
        yMatrix[:, j] = yaxis
    im = plt.contourf(yMatrix,
             xMatrix, normData, levels=levels,
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
#read the trajectory for cal
u = MDAnalysis.Universe(pdb, trj)
#cal the average size of the box,
print ("calculating the average size of the box ... ")
box_x = 0
box_y = 0
box_z = 0
count = 0
for ts in u.trajectory[b:e:inte]:
    box = ts.dimensions[:3]
    box_x += box[0]
    box_y += box[1]
    box_z += box[2]
    count += 1
box_x_ave = box_x / float(count)
box_y_ave = box_y / float(count)
box_z_ave = box_z / float(count)
#cal the num of bins
bin_x_num = int(box_x_ave / bins)
bin_y_num = int(box_y_ave / bins)
bin_z_num = int(box_z_ave / bins)
outf_ratio = open(phasepath+sys+'-ratio.xvg', 'w')
h_up_list = []; e_list = []
h_low_list = [];
h_list=[]
for ts in u.trajectory[b:e:inte]:
    index = int((ts.frame-b)/inte) 
    sel_up1 = 'resid ' + up1[index]
    sel_up2 = 'resid ' + up2[index]
    sel_lp1 = 'resid ' + low1[index]
    sel_lp2 = 'resid ' + low2[index]

    hist_all, edge_fr = cal_densmap('all', bin_x_num, bin_y_num, bin_z_num, box_x_ave, box_y_ave, box_z_ave)

    hist_up1, _ = cal_densmap(sel_up1, bin_x_num, bin_y_num, bin_z_num, box_x_ave, box_y_ave, box_z_ave, bins=edge_fr) if len(sel_up1) != 6 else (np.zeros_like(hist_all), edge_fr)
    hist_up2, _ = cal_densmap(sel_up2, bin_x_num, bin_y_num, bin_z_num, box_x_ave, box_y_ave, box_z_ave, bins=edge_fr) if len(sel_up2) != 6 else (np.zeros_like(hist_all), edge_fr)
    hist_lp1, _ = cal_densmap(sel_lp1, bin_x_num, bin_y_num, bin_z_num, box_x_ave, box_y_ave, box_z_ave, bins=edge_fr) if len(sel_lp1) != 6 else (np.zeros_like(hist_all), edge_fr)
    hist_lp2, _ = cal_densmap(sel_lp2, bin_x_num, bin_y_num, bin_z_num, box_x_ave, box_y_ave, box_z_ave, bins=edge_fr) if len(sel_lp2) != 6 else (np.zeros_like(hist_all), edge_fr)

    #cal ratio of gel/lo phase in each leaflet
    p0_hist_u, p0_ratio_u = ratio_gel(hist_up1, hist_up2)
    p0_hist_l, p0_ratio_l = ratio_gel(hist_lp1, hist_lp2)
    #cal ratio of registration
    regi_hist, regi_ratio = ratio_regi(p0_hist_u, p0_hist_l)

    h_up_list.append(p0_hist_u)
    h_low_list.append(p0_hist_l)
    h_list.append(regi_hist)
    e_list.append(edge_fr)
    print ('%d %.4f %.4f %.4f %.4f %.4f %.4f' % (\
            ts.frame, p0_ratio_u, p0_ratio_l, regi_ratio[0], regi_ratio[1], regi_ratio[2], regi_ratio[3]), file=outf_ratio),
    
 
print("Finish calculating!")
if(sys == 'dpdo280k'):
    tit = 'Binary mixture 280 K'
elif(sys == 'dpdo290k'):
    tit = 'Binary mixture 290 K'
elif(sys == 'dpdocchl280k'):
    tit = 'Ternary mixture 280 K'
elif(sys == 'dpdochl290k'):
    tit = 'Ternary mixture 290 K'
elif(sys == 'psmdopochl'):
    tit = 'Quaternary mixture 300 K'


for i in range(len(h_up_list)): 
    fig1 = plt.figure(figsize=(3, 3.5))
    h_up = h_up_list[i]
    e_up = e_list[i]
    h_low = h_low_list[i]
    e_low = e_list[i]
    h = h_list[i]

    im = plotDensmap(h_up, e_up)
    fig1.suptitle(tit+'\nupper ts = ' + str(b+i*5) +'ns')
    fig1.savefig(phasepath+"phaseplot/upper/"+sys+'-'+str(b+i*5)+'-upper.png', 
                dpi=350,
                format='png',     
                bbox_inches='tight',
                pad_inches=0.0,    
                # facecolor='white',   
                # edgecolor='black',  
                transparent=False, 
                )
    
    # im = plotDensmap(h_low, e_low)
    # fig1.suptitle(tit+' lower ts = ' + str(b+i*5) +'ns')
    # fig1.savefig(phasepath+"phaseplot/lower/"+sys+'-'+str(b+i*5)+'-lower.png', dpi=350)
    
    im = plotDensmap(h, e_low)
    fig1.suptitle(tit+"\nts = " + str(b+i*5) +'ns')
    fig1.savefig(phasepath+"phaseplot/regi/"+sys+'-'+str(b+i*5)+'.png', 
                dpi=350,
                format='png',      
                bbox_inches='tight', 
                pad_inches=0.0,   
                # facecolor='white',  
                # edgecolor='black', 
                transparent=False,  
                )
    plt.close()
print("Finish plotting!")
