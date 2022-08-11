"""
This script aims to show the training results
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# Sector A, 10 fold cross-validation, 4D latest
def plot_results(dim3MAE, dim2MAE, dim3MAE_P, dim2MAE_P, dim3Loss, dim2Loss, ProcessData, isplotting):

    imp = (dim2MAE.mean()-dim3MAE.mean())/(dim2MAE.mean())
    imp_in_P = (dim2MAE_P.mean()-dim3MAE_P.mean())/(dim3MAE_P.mean())

    imp2 = (dim2Loss.mean()-dim3Loss.mean())/(dim3Loss.mean())


    MAE_Loss3D = np.zeros((len(ProcessData), 2))
    MAE_Loss2D = np.zeros((len(ProcessData), 2))

    t = 20

    for i in range(len(ProcessData)):
        tmp1 = np.sort(np.asarray(ProcessData[i, 2]))
        tmp2 = np.sort(np.asarray(ProcessData[i, 6]))
        MAE_Loss3D[i, 0], MAE_Loss3D[i, 1] = np.mean(tmp1[0:t]), np.var(tmp1[0:t])
        MAE_Loss2D[i, 0], MAE_Loss2D[i, 1] = np.mean(tmp2[0:t]), np.var(tmp2[0:t])
        # MAE_Loss2D = np.asarray(ProcessData[i, 5])

    x = np.arange(t)


    error = 0.1 + 0.2 * MAE_Loss3D[:,0]

    # plt.errorbar(x, y, e, linestyle='None', marker='^')
    if isplotting:
        plt.errorbar(x, MAE_Loss3D[:, 0], yerr=MAE_Loss3D[:, 1], fmt='-o')
        plt.errorbar(x, MAE_Loss2D[:, 0], yerr=MAE_Loss2D[:, 1], fmt='-*')

        # plt.plot(MAE_Loss3D[:,0], color='orange', label='2L')
        # plt.plot(MAE_Loss3D[:,0], color='red', label='3L')
        plt.xlabel('nums')
        plt.ylabel('Test MAE')
        plt.legend()
        plt.show()


        plt.figure(figsize=(10,5))#size of plot
        plt.title('Comparsion of MAE on test set (20-fold Cross validation)-Sector C',fontsize=15)# title and font size
        labels = '3 Features','2 Features'
        plt.boxplot([MAE_Loss3D[:, 0], MAE_Loss2D[:, 0]], labels = labels, vert=False,showmeans=True)#grid=False：without displaying the grid
        # data.boxplot()#another way to plot the box line, with less parameters and it only accepts dataframe, which is not commonly used.
        plt.xlabel('MAE')

        plt.show()#display the plot
        print(f'The improvement of MAE is {imp}')
        print(f'The improvement of MAE in P is {imp_in_P}')
        print(f'The of MAE of 2-tier NN is {dim3MAE.mean()}')
        print(f'The of MAE of MLP is {dim2MAE.mean()}')

    return MAE_Loss3D[:, 0], MAE_Loss2D[:, 0]



if __name__ == "__main__":
    isplotting = False

    dim3MAE_A = np.load("MAError_10flod_A_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2MAE_A = np.load("MAError_var_10flod_A_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim3MAE_P_A = np.load("MAError_P_10flod_A_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2MAE_P_A = np.load("MAError_P_var_10flod_A_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim3Loss_A = np.load("dim3Loss_all_10flod_A_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2Loss_A = np.load("dim2Loss_all_10flod_A_v2_1403_vae2_4f.npy",allow_pickle=True)
    ProcessData_A = np.load("data_needed_10flod_A_v2_1403_vae2_4f.npy",allow_pickle=True)

    Loss3D_A_Jan, Loss2D_A_Jan = plot_results(dim3MAE_A.T, dim2MAE_A.T, dim3MAE_P_A.T, dim2MAE_P_A.T, dim3Loss_A.T, dim2Loss_A.T, ProcessData_A, isplotting)


    dim3MAE_B = np.load("MAError_10flod_B_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2MAE_B = np.load("MAError_var_10flod_B_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim3MAE_P_B = np.load("MAError_P_10flod_B_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2MAE_P_B = np.load("MAError_P_var_10flod_B_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim3Loss_B = np.load("dim3Loss_all_10flod_B_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2Loss_B = np.load("dim2Loss_all_10flod_B_v2_1403_vae2_4f.npy",allow_pickle=True)
    ProcessData_B = np.load("data_needed_10flod_B_v2_1403_vae2_4f.npy",allow_pickle=True)

    Loss3D_B_Jan, Loss2D_B_Jan = plot_results(dim3MAE_B.T, dim2MAE_B.T, dim3MAE_P_B.T, dim2MAE_P_B.T, dim3Loss_B.T, dim2Loss_B.T, ProcessData_B, isplotting)

    dim3MAE_C = np.load("MAError_10flod_C_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2MAE_C = np.load("MAError_var_10flod_C_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim3MAE_P_C = np.load("MAError_P_10flod_C_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2MAE_P_C = np.load("MAError_P_var_10flod_C_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim3Loss_C = np.load("dim3Loss_all_10flod_C_v2_1403_vae2_4f.npy",allow_pickle=True)
    dim2Loss_C = np.load("dim2Loss_all_10flod_C_v2_1403_vae2_4f.npy",allow_pickle=True)
    ProcessData_C = np.load("data_needed_10flod_C_v2_1403_vae2_4f.npy",allow_pickle=True)

    Loss3D_C_Jan, Loss2D_C_Jan = plot_results(dim3MAE_C.T, dim2MAE_C.T, dim3MAE_P_C.T, dim2MAE_P_C.T, dim3Loss_C.T, dim2Loss_C.T, ProcessData_C, isplotting)



    dim3MAE_A = np.load("MAError_10flod_A_v2_0704_aug_4f.npy",allow_pickle=True)
    dim2MAE_A = np.load("MAError_var_10flod_A_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim3MAE_P_A = np.load("MAError_P_10flod_A_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim2MAE_P_A = np.load("MAError_P_var_10flod_A_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim3Loss_A = np.load("dim3Loss_all_10flod_A_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim2Loss_A = np.load("dim2Loss_all_10flod_A_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    ProcessData_A = np.load("data_needed_10flod_A_v2_0704_aug_vae2_4f.npy",allow_pickle=True)

    Loss3D_A, Loss2D_A = plot_results(dim3MAE_A.T, dim2MAE_A.T, dim3MAE_P_A.T, dim2MAE_P_A.T, dim3Loss_A.T, dim2Loss_A.T, ProcessData_A, isplotting)


    dim3MAE_B = np.load("MAError_10flod_B_v2_1403_0704_aug_4f.npy",allow_pickle=True)
    dim2MAE_B = np.load("MAError_var_10flod_B_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim3MAE_P_B = np.load("MAError_P_10flod_B_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim2MAE_P_B = np.load("MAError_P_var_10flod_B_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim3Loss_B = np.load("dim3Loss_all_10flod_B_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim2Loss_B = np.load("dim2Loss_all_10flod_B_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    ProcessData_B = np.load("data_needed_10flod_B_v2_0704_aug_vae2_4f.npy",allow_pickle=True)

    Loss3D_B, Loss2D_B = plot_results(dim3MAE_B.T, dim2MAE_B.T, dim3MAE_P_B.T, dim2MAE_P_B.T, dim3Loss_B.T, dim2Loss_B.T, ProcessData_B, isplotting)

    dim3MAE_C = np.load("MAError_10flod_C_v2_1403_0704_aug_4f.npy",allow_pickle=True)
    dim2MAE_C = np.load("MAError_var_10flod_C_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim3MAE_P_C = np.load("MAError_P_10flod_C_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim2MAE_P_C = np.load("MAError_P_var_10flod_C_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim3Loss_C = np.load("dim3Loss_all_10flod_C_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    dim2Loss_C = np.load("dim2Loss_all_10flod_C_v2_0704_aug_vae2_4f.npy",allow_pickle=True)
    ProcessData_C = np.load("data_needed_10flod_C_v2_0704_aug_vae2_4f.npy",allow_pickle=True)

    Loss3D_C, Loss2D_C = plot_results(dim3MAE_C.T, dim2MAE_C.T, dim3MAE_P_C.T, dim2MAE_P_C.T, dim3Loss_C.T, dim2Loss_C.T, ProcessData_C, isplotting)



    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 9))
    y_font_size = 14
    x_font_size = 13
    # plt.annotate('August', xy=(-2.7, 0.55), xycoords='axes fraction')

    bplot1 = axes[0][0].boxplot([Loss3D_A_Jan, Loss2D_A_Jan],
                             vert=True,
                             patch_artist=True,
                             showmeans=True,
                             labels = ['2-tier NN', 'MLP'])

    axes[0][0].set_title('Sector A', fontsize=x_font_size)
    axes[0][0].set_ylabel('January', fontsize=y_font_size)
    bplot2 = axes[0][1].boxplot([Loss3D_B_Jan, Loss2D_B_Jan],
                             vert=True,
                             patch_artist=True,
                             showmeans=True,
                             labels=['2-tier NN', 'MLP']
                             )
    axes[0][1].set_title('Sector B',fontsize=x_font_size)


    bplot3 = axes[0][2].boxplot([Loss3D_C_Jan, Loss2D_C_Jan],
                             vert=True,
                             patch_artist=True,
                             showmeans=True,
                             labels=['2-tier NN', 'MLP']
                             )
    axes[0][2].set_title('Sector C',fontsize=x_font_size)
    # axes[0][2].yaxis.set_label_position("right")
    # axes[0][2].yaxis.tick_right()


    bplot4 = axes[1][0].boxplot([Loss3D_A, Loss2D_A],
                             vert=True,
                             patch_artist=True,
                             showmeans=True,
                             labels = ['2-tier NN', 'MLP'])
    # axes[1][0].set_title('Aug-Sector A')
    axes[1][0].set_ylabel('August',fontsize=y_font_size)

    bplot5 = axes[1][1].boxplot([Loss3D_B, Loss2D_B],
                             vert=True,
                             patch_artist=True,
                             showmeans=True,
                             labels=['2-tier NN', 'MLP']
                             )
    # axes[1][1].set_title('Aug-Sector B')
    bplot6 = axes[1][2].boxplot([Loss3D_C, Loss2D_C],
                             vert=True,
                             patch_artist=True,
                             showmeans=True,
                             labels=['2-tier NN', 'MLP']
                             )
    # axes[1][2].set_title('Aug-Sector C')
    # 颜色填充
    colors = ['pink', 'lightblue']
    for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5, bplot6):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    fig.legend([bplot1["boxes"][0], bplot1["boxes"][1]], ['2-tier NN', 'MLP'], bbox_to_anchor=[0.91, 0.98], loc='upper right')
    # (0.85,0.9)
    # for ax in axes:
    #     ax.yaxis.grid(True)  # 在y轴上添加网格线
    # ax.set_xticks([y + 1 for y in range(len(all_data))])  # 指定x轴的轴刻度个数
    ## [y+1 for y in range(len(all_data))]运行结果是[1,2,3]
    # ax.set_xlabel('xlabel')  # 设置x轴名称
    # ax.set_ylabel('ylabel')  # 设置y轴名称

    # fig.suptitle('Comparsion of MAE on test set (20-fold Cross validation) over different sectors and months', fontsize = y_font_size)  # title and font size
    # plt.ylabel('MAE')

    # plt.show()
    plt.savefig("plot_result/boxplot.svg")
    # fig.savefig('boxplot_aug.eps', bbox_inches = 'tight', dpi=600, format='eps', pad_inches=0.0)