import pandas as pd
import matplotlib.pyplot as plt

p = 1
df = pd.read_csv(f'all_analysis_p{p}.csv')

max_diff_time = 5
from matplotlib.backends.backend_pdf import PdfPages

dfs_dict = {}
for noise in [0.05, 0.10, 0.15]:
    # delete timed-out simulation results in right noise profile
    aux_df = df[(df['noise_std'] == noise) & (df['t_diff'] < max_diff_time)]
    # select relative error columns
    dfs_dict[str(noise)] = aux_df.filter(regex='^img_index|^rel_error_synth')

noise_get = '0.1'
for step in range(0, len(dfs_dict.get(noise_get)), 21):
    img_indices = dfs_dict.get(noise_get)[dfs_dict.get(noise_get).columns[0]][step:(step + 21)]
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)
    i = 1
    for col_name in dfs_dict.get(noise_get).columns[1:]:
        ax.plot(img_indices, dfs_dict.get(noise_get)[col_name][step:(step + 21)],
                label=f'syn_{i}', marker=6 if i <= 10 else 7,
                linestyle='dashed', markersize=5,lw=0.5)
        i += 1
    ax.set_xticks(img_indices)

    plt.title(f'Relative error w.r.t. maximum PSNR value\nstd_noise = {noise_get}')
    plt.xlabel('image indices')
    plt.legend(loc=(1.04, 0))
    pp = PdfPages(f'{noise_get}-{step}-p{p}.pdf')
    pp.savefig(fig)
    pp.close()
    # plt.show()

# jugadita de pillar las cols q me interesan
# for col in err_005.columns[1:]:
#     print(col)
# jugadita de pillar la col como index
