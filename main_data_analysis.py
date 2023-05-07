from data_analysis import *

##this is where we execute function related to data analysis

print(pred_PCA("dat files/Base_fits01.dat", "Arms"))
print(pred_LDA("dat files/Base_fits01.dat", "Arms"))
plot_df_PCA_reduction('Arms')
plot_df_LDA_reduction('Arms')