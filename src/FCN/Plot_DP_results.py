"""
This script aims to show the training results
"""
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Sector A, 10 fold cross-validation, 4D latest


def plot_results(dim3MAE, dim2MAE, dim3MAE_P, dim2MAE_P, dim3Loss, dim2Loss, ProcessData, isplotting, month, sector):

    imp = (dim2MAE.mean()-dim3MAE.mean())/(dim2MAE.mean())
    imp_in_P = (dim2MAE_P.mean()-dim3MAE_P.mean())/(dim2MAE_P.mean())

    imp2 = (dim2Loss.mean()-dim3Loss.mean())/(dim2Loss.mean())

    MAE_Loss3D = np.zeros((len(ProcessData), 2))
    MAE_Loss2D = np.zeros((len(ProcessData), 2))

    t = 20

    for i in range(len(ProcessData)):
        tmp1 = np.sort(np.asarray(ProcessData[i, 2]))
        tmp2 = np.sort(np.asarray(ProcessData[i, 6]))
        MAE_Loss3D[i, 0], MAE_Loss3D[i, 1] = np.mean(
            tmp1[0:t]), np.var(tmp1[0:t])
        MAE_Loss2D[i, 0], MAE_Loss2D[i, 1] = np.mean(
            tmp2[0:t]), np.var(tmp2[0:t])
        # MAE_Loss2D = np.asarray(ProcessData[i, 5])

    x = np.arange(t)
    x_fold = np.arange(1, t+1)
    # print(x_fold)
    error = 0.1 + 0.2 * MAE_Loss3D[:, 0]

    # plt.errorbar(x, y, e, linestyle='None', marker='^')
    if isplotting:
        # plt.errorbar(x, MAE_Loss3D[:, 0], yerr=MAE_Loss3D[:, 1], fmt='-o')
        # plt.errorbar(x, MAE_Loss2D[:, 0], yerr=MAE_Loss2D[:, 1], fmt='-*')

        plt.plot(x_fold, MAE_Loss2D[:, 0],
                 color='orange', marker='o', label='MLP')
        plt.plot(x_fold, MAE_Loss3D[:, 0],
                 color='purple', marker='*', label='2-stage NN')
        plt.xlabel('# fold cross-validation')
        plt.ylabel('Test MAE (dBm)')
        plt.xticks(np.arange(min(x_fold), max(x_fold)+1, 1))
        plt.legend()
        # plt.show()
        plt.savefig(f"plot_result/MAE_line/MAE_{month}_{sector}.svg")
        plt.close()

        # plt.figure(figsize=(10,5))#size of plot
        # plt.title('Comparsion of MAE on test set (20-fold Cross validation)-Sector C',fontsize=15)# title and font size
        # labels = '3 Features','2 Features'
        # plt.boxplot([MAE_Loss3D[:, 0], MAE_Loss2D[:, 0]], labels = labels, vert=False,showmeans=True)#grid=False：without displaying the grid
        # # data.boxplot()#another way to plot the box line, with less parameters and it only accepts dataframe, which is not commonly used.
        # plt.xlabel('MAE')

        # plt.show()#display the plot
        print(f'The of MAE of 2-stage NN is {dim3MAE.mean()}')
        print(f'The of MAE of MLP is {dim2MAE.mean()}')
        print(f'The improvement of MAE is {imp} ({imp_in_P*100:.2f}%)')
        print("----------------------------------------------------------")

    return MAE_Loss3D[:, 0], MAE_Loss2D[:, 0], dim3MAE.mean(), dim2MAE.mean()


def bar_plot(MAE_Mean3D, MAE_Mean2D, sector):
    # set width of bars
    barWidth = 0.25

    # set heights of bars
    MAE_Mean3D_Jan = [MAE_Mean3D['Jan']['A'],
                      MAE_Mean3D['Jan']['B'], MAE_Mean3D['Jan']['C']]
    MAE_Mean2D_Jan = [MAE_Mean2D['Jan']['A'],
                      MAE_Mean2D['Jan']['B'], MAE_Mean2D['Jan']['C']]
    MAE_Mean3D_Aug = [MAE_Mean3D['Aug']['A'],
                      MAE_Mean3D['Aug']['B'], MAE_Mean3D['Aug']['C']]
    MAE_Mean2D_Aug = [MAE_Mean2D['Aug']['A'],
                      MAE_Mean2D['Aug']['B'], MAE_Mean2D['Aug']['C']]

    # Set position of bar on X axis
    x1 = np.arange(len(MAE_Mean3D_Jan))
    x2 = [x + barWidth for x in x1]

    ###### Make the plot for all Sectors in Jan ######
    plt.bar(x1, MAE_Mean3D_Jan, color='purple', width=barWidth,
            edgecolor='white', label='2-stage NN')
    plt.bar(x2, MAE_Mean2D_Jan, color='orange', width=barWidth,
            edgecolor='white', label='MLP')
    # Add xticks on the middle of the group bars
    plt.xlabel('Jan', fontweight='bold')
    plt.xticks([m + (barWidth/2)
               for m in range(len(MAE_Mean3D_Jan))], ['A', 'B', 'C'])
    plt.ylabel('Mean MAE (dBm)')
    # Create legend & Show graphic
    plt.legend()
    # plt.show()
    plt.savefig(f"plot_result/MAE_bar/Mean_MAE_Jan.svg")
    plt.close()

    ###### Make the plot for all Sectors in Aug ######
    plt.bar(x1, MAE_Mean3D_Aug, color='purple', width=barWidth,  # 3C88C0
            edgecolor='white', label='2-stage NN')
    plt.bar(x2, MAE_Mean2D_Aug, color='orange', width=barWidth,  # 6BDD7F
            edgecolor='white', label='MLP')
    # Add xticks on the middle of the group bars
    plt.xlabel('Aug', fontweight='bold')
    plt.xticks([m + (barWidth/2)
               for m in range(len(MAE_Mean3D_Aug))], ['A', 'B', 'C'])
    plt.ylabel('Mean MAE (dBm)')
    # Create legend & Show graphic
    plt.legend()
    # plt.show()
    plt.savefig(f"plot_result/MAE_bar/Mean_MAE_Aug.svg")
    plt.close()

    #-------------- ALternatively use panda --------------#
    plt.figure(figsize=(10, 8))
    dfs = pd.DataFrame(data={'Sectors': sector,
                             '2-stage NN': MAE_Mean3D_Jan,
                             'MLP': MAE_Mean2D_Jan})
    dfs1 = pd.melt(dfs, id_vars="Sectors")
    dfs1.rename(columns={'variable': 'Model'}, inplace=True)
    dfs1.rename(columns={'value': 'Mean MAE (dBm)'}, inplace=True)
    plot = sns.barplot(x='Sectors', y='Mean MAE (dBm)', hue='Model', data=dfs1)
    plt.legend(loc='upper right')
    for bar in plot.patches:
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
        plot.annotate(format(bar.get_height(), '.2f'),
                      (bar.get_x() + bar.get_width() / 2,
                       bar.get_height()), ha='center', va='center',
                      size=15, xytext=(0, 5),
                      textcoords='offset points')
    # plt.show()
    plt.savefig(f"plot_result/MAE_bar/Mean_MAE_Jan_panda.svg")
    plt.close()

    plt.figure(figsize=(10, 8))
    dfs = pd.DataFrame(data={'Sectors': sector,
                             '2-stage NN': MAE_Mean3D_Aug,
                             'MLP': MAE_Mean2D_Aug})
    dfs1 = pd.melt(dfs, id_vars="Sectors")
    dfs1.rename(columns={'variable': 'Model'}, inplace=True)
    dfs1.rename(columns={'value': 'Mean MAE (dBm)'}, inplace=True)
    plot = sns.barplot(x='Sectors', y='Mean MAE (dBm)', hue='Model', data=dfs1)
    plt.legend(loc='upper right')
    for bar in plot.patches:
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
        plot.annotate(format(bar.get_height(), '.2f'),
                      (bar.get_x() + bar.get_width() / 2,
                       bar.get_height()), ha='center', va='center',
                      size=15, xytext=(0, 5),
                      textcoords='offset points')
    # plt.show()
    plt.savefig(f"plot_result/MAE_bar/Mean_MAE_Aug_panda.svg")
    plt.close()


def box_plot(Loss3D, Loss2D):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 9))
    y_font_size = 14
    x_font_size = 13
    # plt.annotate('August', xy=(-2.7, 0.55), xycoords='axes fraction')

    bplot1 = axes[0][0].boxplot([Loss3D['Jan']['A'], Loss2D['Jan']['A']],
                                vert=True,
                                patch_artist=True,
                                showmeans=True,
                                labels=['2-stage NN', 'MLP'])

    axes[0][0].set_title('Sector A', fontsize=x_font_size)
    axes[0][0].set_ylabel('January', fontsize=y_font_size)
    bplot2 = axes[0][1].boxplot([Loss3D['Jan']['B'], Loss2D['Jan']['B']],
                                vert=True,
                                patch_artist=True,
                                showmeans=True,
                                labels=['2-stage NN', 'MLP']
                                )
    axes[0][1].set_title('Sector B', fontsize=x_font_size)

    bplot3 = axes[0][2].boxplot([Loss3D['Jan']['C'], Loss2D['Jan']['C']],
                                vert=True,
                                patch_artist=True,
                                showmeans=True,
                                labels=['2-stage NN', 'MLP']
                                )
    axes[0][2].set_title('Sector C', fontsize=x_font_size)
    # axes[0][2].yaxis.set_label_position("right")
    # axes[0][2].yaxis.tick_right()

    bplot4 = axes[1][0].boxplot([Loss3D['Aug']['A'], Loss2D['Aug']['A']],
                                vert=True,
                                patch_artist=True,
                                showmeans=True,
                                labels=['2-stage NN', 'MLP'])
    # axes[1][0].set_title('Aug-Sector A')
    axes[1][0].set_ylabel('August', fontsize=y_font_size)

    bplot5 = axes[1][1].boxplot([Loss3D['Aug']['B'], Loss2D['Aug']['B']],
                                vert=True,
                                patch_artist=True,
                                showmeans=True,
                                labels=['2-stage NN', 'MLP']
                                )
    # axes[1][1].set_title('Aug-Sector B')
    bplot6 = axes[1][2].boxplot([Loss3D['Aug']['C'], Loss2D['Aug']['C']],
                                vert=True,
                                patch_artist=True,
                                showmeans=True,
                                labels=['2-stage NN', 'MLP']
                                )
    # axes[1][2].set_title('Aug-Sector C')
    # 颜色填充
    colors = ['lightpink', 'skyblue']
    for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5, bplot6):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    fig.legend([bplot1["boxes"][0], bplot1["boxes"][1]], [
               '2-stage NN', 'MLP'], bbox_to_anchor=[0.91, 0.98], loc='upper right')
    # (0.85,0.9)
    # for ax in axes:
    #     ax.yaxis.grid(True)  # 在y轴上添加网格线
    # ax.set_xticks([y + 1 for y in range(len(all_data))])  # 指定x轴的轴刻度个数
    # # [y+1 for y in range(len(all_data))]运行结果是[1,2,3]
    # ax.set_xlabel('xlabel')  # 设置x轴名称
    # ax.set_ylabel('ylabel')  # 设置y轴名称

    # fig.suptitle('Comparsion of MAE on test set (20-fold Cross validation) over different sectors and months', fontsize = y_font_size)  # title and font size
    # plt.ylabel('MAE')

    # plt.show()
    plt.savefig("plot_result/MAE_boxplot.svg")
    # fig.savefig('boxplot_aug.eps', bbox_inches = 'tight', dpi=600, format='eps', pad_inches=0.0)


if __name__ == "__main__":
    month = ['Jan', 'Aug']
    sector = ['A', 'B', 'C']
    Loss3D = {f'{month[0]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []},
              f'{month[1]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []}}
    Loss2D = {f'{month[0]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []},
              f'{month[1]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []}}
    MAE_Mean3D = {f'{month[0]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []},
                  f'{month[1]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []}}
    MAE_Mean2D = {f'{month[0]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []},
                  f'{month[1]}': {f'{sector[0]}': [], f'{sector[1]}': [], f'{sector[2]}': []}}
    isplotting = True

    dim3MAE_A = np.load(
        "train_test_output/npy/jan/A/MAError_10flod_A_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2MAE_A = np.load(
        "train_test_output/npy/jan/A/MAError_var_10flod_A_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim3MAE_P_A = np.load(
        "train_test_output/npy/jan/A/MAError_P_10flod_A_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2MAE_P_A = np.load(
        "train_test_output/npy/jan/A/MAError_P_var_10flod_A_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim3Loss_A = np.load(
        "train_test_output/npy/jan/A/dim3Loss_all_10flod_A_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2Loss_A = np.load(
        "train_test_output/npy/jan/A/dim2Loss_all_10flod_A_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    ProcessData_A = np.load(
        "train_test_output/npy/jan/A/data_needed_10flod_A_v2_1403_jan_vae2_4f.npy", allow_pickle=True)

    Loss3D['Jan']['A'], Loss2D['Jan']['A'], MAE_Mean3D['Jan']['A'], MAE_Mean2D['Jan']['A'] = plot_results(
        dim3MAE_A.T, dim2MAE_A.T, dim3MAE_P_A.T, dim2MAE_P_A.T, dim3Loss_A.T, dim2Loss_A.T, ProcessData_A, isplotting, month[0], sector[0])

    dim3MAE_B = np.load(
        "train_test_output/npy/jan/B/MAError_10flod_B_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2MAE_B = np.load(
        "train_test_output/npy/jan/B/MAError_var_10flod_B_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim3MAE_P_B = np.load(
        "train_test_output/npy/jan/B/MAError_P_10flod_B_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2MAE_P_B = np.load(
        "train_test_output/npy/jan/B/MAError_P_var_10flod_B_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim3Loss_B = np.load(
        "train_test_output/npy/jan/B/dim3Loss_all_10flod_B_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2Loss_B = np.load(
        "train_test_output/npy/jan/B/dim2Loss_all_10flod_B_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    ProcessData_B = np.load(
        "train_test_output/npy/jan/B/data_needed_10flod_B_v2_1403_jan_vae2_4f.npy", allow_pickle=True)

    Loss3D['Jan']['B'], Loss2D['Jan']['B'], MAE_Mean3D['Jan']['B'], MAE_Mean2D['Jan']['B'] = plot_results(
        dim3MAE_B.T, dim2MAE_B.T, dim3MAE_P_B.T, dim2MAE_P_B.T, dim3Loss_B.T, dim2Loss_B.T, ProcessData_B, isplotting, month[0], sector[1])

    dim3MAE_C = np.load(
        "train_test_output/npy/jan/C/MAError_10flod_C_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2MAE_C = np.load(
        "train_test_output/npy/jan/C/MAError_var_10flod_C_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim3MAE_P_C = np.load(
        "train_test_output/npy/jan/C/MAError_P_10flod_C_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2MAE_P_C = np.load(
        "train_test_output/npy/jan/C/MAError_P_var_10flod_C_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim3Loss_C = np.load(
        "train_test_output/npy/jan/C/dim3Loss_all_10flod_C_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    dim2Loss_C = np.load(
        "train_test_output/npy/jan/C/dim2Loss_all_10flod_C_v2_1403_jan_vae2_4f.npy", allow_pickle=True)
    ProcessData_C = np.load(
        "train_test_output/npy/jan/C/data_needed_10flod_C_v2_1403_jan_vae2_4f.npy", allow_pickle=True)

    Loss3D['Jan']['C'], Loss2D['Jan']['C'], MAE_Mean3D['Jan']['C'], MAE_Mean2D['Jan']['C'] = plot_results(
        dim3MAE_C.T, dim2MAE_C.T, dim3MAE_P_C.T, dim2MAE_P_C.T, dim3Loss_C.T, dim2Loss_C.T, ProcessData_C, isplotting, month[0], sector[2])

    dim3MAE_A = np.load(
        "train_test_output/npy/aug/A/MAError_10flod_A_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2MAE_A = np.load(
        "train_test_output/npy/aug/A/MAError_var_10flod_A_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim3MAE_P_A = np.load(
        "train_test_output/npy/aug/A/MAError_P_10flod_A_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2MAE_P_A = np.load(
        "train_test_output/npy/aug/A/MAError_P_var_10flod_A_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim3Loss_A = np.load(
        "train_test_output/npy/aug/A/dim3Loss_all_10flod_A_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2Loss_A = np.load(
        "train_test_output/npy/aug/A/dim2Loss_all_10flod_A_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    ProcessData_A = np.load(
        "train_test_output/npy/aug/A/data_needed_10flod_A_v2_1403_aug_vae2_4f.npy", allow_pickle=True)

    Loss3D['Aug']['A'], Loss2D['Aug']['A'], MAE_Mean3D['Aug']['A'], MAE_Mean2D['Aug']['A'] = plot_results(
        dim3MAE_A.T, dim2MAE_A.T, dim3MAE_P_A.T, dim2MAE_P_A.T, dim3Loss_A.T, dim2Loss_A.T, ProcessData_A, isplotting, month[1], sector[0])

    dim3MAE_B = np.load(
        "train_test_output/npy/aug/B/MAError_10flod_B_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2MAE_B = np.load(
        "train_test_output/npy/aug/B/MAError_var_10flod_B_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim3MAE_P_B = np.load(
        "train_test_output/npy/aug/B/MAError_P_10flod_B_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2MAE_P_B = np.load(
        "train_test_output/npy/aug/B/MAError_P_var_10flod_B_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim3Loss_B = np.load(
        "train_test_output/npy/aug/B/dim3Loss_all_10flod_B_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2Loss_B = np.load(
        "train_test_output/npy/aug/B/dim2Loss_all_10flod_B_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    ProcessData_B = np.load(
        "train_test_output/npy/aug/B/data_needed_10flod_B_v2_1403_aug_vae2_4f.npy", allow_pickle=True)

    Loss3D['Aug']['B'], Loss2D['Aug']['B'], MAE_Mean3D['Aug']['B'], MAE_Mean2D['Aug']['B'] = plot_results(
        dim3MAE_B.T, dim2MAE_B.T, dim3MAE_P_B.T, dim2MAE_P_B.T, dim3Loss_B.T, dim2Loss_B.T, ProcessData_B, isplotting, month[1], sector[1])

    dim3MAE_C = np.load(
        "train_test_output/npy/aug/C/MAError_10flod_C_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2MAE_C = np.load(
        "train_test_output/npy/aug/C/MAError_var_10flod_C_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim3MAE_P_C = np.load(
        "train_test_output/npy/aug/C/MAError_P_10flod_C_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2MAE_P_C = np.load(
        "train_test_output/npy/aug/C/MAError_P_var_10flod_C_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim3Loss_C = np.load(
        "train_test_output/npy/aug/C/dim3Loss_all_10flod_C_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    dim2Loss_C = np.load(
        "train_test_output/npy/aug/C/dim2Loss_all_10flod_C_v2_1403_aug_vae2_4f.npy", allow_pickle=True)
    ProcessData_C = np.load(
        "train_test_output/npy/aug/C/data_needed_10flod_C_v2_1403_aug_vae2_4f.npy", allow_pickle=True)

    Loss3D['Aug']['C'], Loss2D['Aug']['C'], MAE_Mean3D['Aug']['C'], MAE_Mean2D['Aug']['C'] = plot_results(
        dim3MAE_C.T, dim2MAE_C.T, dim3MAE_P_C.T, dim2MAE_P_C.T, dim3Loss_C.T, dim2Loss_C.T, ProcessData_C, isplotting, month[1], sector[2])

    box_plot(Loss3D, Loss2D)
    bar_plot(MAE_Mean3D, MAE_Mean2D, sector)
