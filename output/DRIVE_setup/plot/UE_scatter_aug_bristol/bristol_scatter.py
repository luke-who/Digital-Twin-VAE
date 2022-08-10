import numpy as np
import pandas as pd
from nn_utilities import data_load_processing
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [12.00, 9.0]
plt.rcParams["figure.autolayout"] = True
im = plt.imread("bristol.PNG")
x_bristol=[0,1128]
x_brisRange=x_bristol[1]-x_bristol[0]
y_bristol=[0,824]
y_brisRange=y_bristol[1]-y_bristol[0]

fig, ax = plt.subplots()
im = ax.imshow(im, extent=[0, x_bristol[1], 0, y_bristol[1]], aspect='auto')
# im = ax.imshow(im)


data_A_jan = pd.read_csv('MDT data_jan/mdt_10231A11_10m_Jan.csv')
data_B_jan= pd.read_csv('MDT data_jan/mdt_10231B11_10m_Jan.csv')
data_C_jan = pd.read_csv('MDT data_jan/mdt_10231C11_10m_Jan.csv')
data_A_aug = pd.read_csv('MDT data_aug/mdt_10231A11_10m_aug.csv')
data_B_aug = pd.read_csv('MDT data_aug/mdt_10231B11_10m_aug.csv')
data_C_aug = pd.read_csv('MDT data_aug/mdt_10231C11_10m_aug.csv')

xlim=[356940,361021]
xrange = xlim[1]-xlim[0]
ylim=[172200,174400]
# ylim=[171658,173912]
yrange = ylim[1]-ylim[0]

# filter for x axis
f_ymin = ylim[0]+650
f_ymax = ylim[1]-100
f_xmin = xlim[0]
f_xmax = xlim[1]-1371

# filter data points to a certain frame
filtered_A_aug_xmin, filtered_A_aug_ymin = [], []
filtered_B_aug_xmin, filtered_B_aug_ymin = [], []
filtered_C_aug_xmin, filtered_C_aug_ymin = [], []

# for i in list(range(len(data_A_aug.xmin))):
#     # if (data_A_aug.xmin[i]>xlim[0] and data_A_aug.xmin[i]<xlim[1]) and (data_A_aug.ymin[i]>ylim[0] and data_A_aug.ymin[i]<ylim[1]): # confine to a certain frame
#     if (data_A_aug.xmin[i]>f_xmin and data_A_aug.xmin[i]<f_xmax) and (data_A_aug.ymin[i]>f_ymin and data_A_aug.ymin[i]<f_ymax): # remove certain outliers
#         filtered_A_aug_xmin.append(data_A_aug.xmin[i]), filtered_A_aug_ymin.append(data_A_aug.ymin[i])
# for i in list(range(len(data_B_aug.xmin))):
#     if (data_B_aug.xmin[i]>f_xmin and data_B_aug.xmin[i]<f_xmax) and (data_B_aug.ymin[i]>f_ymin and data_B_aug.ymin[i]<f_ymax):
#         filtered_B_aug_xmin.append(data_B_aug.xmin[i]), filtered_B_aug_ymin.append(data_B_aug.ymin[i])
# for i in list(range(len(data_C_aug.xmin))):
#     if (data_C_aug.xmin[i]>f_xmin and data_C_aug.xmin[i]<f_xmax) and (data_C_aug.ymin[i]>f_ymin and data_C_aug.ymin[i]<f_ymax):
#         filtered_C_aug_xmin.append(data_C_aug.xmin[i]), filtered_C_aug_ymin.append(data_C_aug.ymin[i])

# Hampel function to remove outlier
def remove_outlier_Hampel(x,y):
    deviation_factor = 12
    x = [float(xx) for xx in x]
    y = [float(yx) for yx in y]

    med_x=np.median(x)
    List_x=abs(x-med_x)
    cond_x=np.median(List_x)*deviation_factor

    good_index_x = [i for i,l in enumerate(List_x) if ~(l>cond_x)]

    mid_x = [x[i] for i in good_index_x]
    mid_y = [y[i] for i in good_index_x]

    med_y=np.median(mid_y)
    List_y=abs(mid_y-med_y)
    cond_y=np.median(List_y)*deviation_factor

    good_index_y = [i for i,l in enumerate(List_y) if ~(l>cond_y)]
    x_without_outliers = [mid_x[i] for i in good_index_y]
    y_without_outliers = [mid_y[i] for i in good_index_y]

    return x_without_outliers, y_without_outliers

filtered_A_aug_xmin, filtered_A_aug_ymin = remove_outlier_Hampel(data_A_aug.xmin,data_A_aug.ymin)
filtered_B_aug_xmin, filtered_B_aug_ymin = remove_outlier_Hampel(data_B_aug.xmin,data_B_aug.ymin)
filtered_C_aug_xmin, filtered_C_aug_ymin = remove_outlier_Hampel(data_C_aug.xmin,data_C_aug.ymin)

# # scatter scale to bristol pic
filtered_A_aug_xmin_scaled = [int(((f_A_aug_x-xlim[0])/xrange)*x_brisRange) for f_A_aug_x in filtered_A_aug_xmin]
filtered_A_aug_ymin_scaled = [int(((f_A_aug_y-ylim[0])/yrange)*y_brisRange) for f_A_aug_y in filtered_A_aug_ymin]
filtered_B_aug_xmin_scaled = [int(((f_B_aug_x-xlim[0])/xrange)*x_brisRange) for f_B_aug_x in filtered_B_aug_xmin]
filtered_B_aug_ymin_scaled = [int(((f_B_aug_y-ylim[0])/yrange)*y_brisRange) for f_B_aug_y in filtered_B_aug_ymin]
filtered_C_aug_xmin_scaled = [int(((f_C_aug_x-xlim[0])/xrange)*x_brisRange) for f_C_aug_x in filtered_C_aug_xmin]
filtered_C_aug_ymin_scaled = [int(((f_C_aug_y-ylim[0])/yrange)*y_brisRange) for f_C_aug_y in filtered_C_aug_ymin]
# plt.scatter(data_A_jan.xmin, data_A_jan.ymin, color='red', s=1)
# plt.scatter(data_B_jan.xmin, data_B_jan.ymin, color='blue', s=1)
# plt.scatter(data_C_jan.xmin, data_C_jan.ymin, color='green', s=1)
ax.scatter(filtered_A_aug_xmin_scaled, filtered_A_aug_ymin_scaled,label='Sector A', color='red', s=7)
ax.scatter(filtered_B_aug_xmin_scaled, filtered_B_aug_ymin_scaled,label='Sector B', color='blue', s=7)
ax.scatter(filtered_C_aug_xmin_scaled, filtered_C_aug_ymin_scaled,label='Sector C', color='green', s=7)
ax.annotate("BS", (417, 498), xytext=(417-210, 489+30), fontsize=15, arrowprops = dict(arrowstyle="fancy"))
plt.xlim(x_bristol[0],x_bristol[1])
plt.ylim(y_bristol[0],y_bristol[1])
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.legend()
# plt.show() # comment this line if you want to save the figure as a svg file

plt.tight_layout()
plt.savefig("bristol_UE_scatter.svg")
plt.close()


