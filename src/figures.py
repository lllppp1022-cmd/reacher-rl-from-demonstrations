from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def evaluation_plot(folder):
    a=dict(np.load(folder/'trajectories.npz'));fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(np.arange(1,251)*.02,a['distance_m']*100,alpha=.15);ax[0].axhline(1,color='black',ls='--');ax[0].set(xlabel='Time (s)',ylabel='XY error (cm)',title='Repeat verification: all episodes')
    ax[1].plot(np.arange(1,251)*.02,a['actions'].mean(axis=1));ax[1].set(xlabel='Time (s)',ylabel='Mean executed action',title='Both joints; episode mean')
    fig.tight_layout();fig.savefig(folder/'trajectories.png',dpi=150);plt.close(fig)
