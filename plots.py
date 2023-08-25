from pathlib import Path

import torch
import pandas as pd
import hydra
from omegaconf import OmegaConf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_stats(data_paths):
    # df = pd.concat(map(pd.DataFrame, map(torch.load, data_paths)))
    df = pd.DataFrame.from_records(map(torch.load, data_paths))
    return df


@hydra.main(version_base=None, config_path='configs', config_name='plots')
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    df = get_stats(sorted(cfg.data_paths))

    ax = sns.lineplot(
        data=df, x='num_neighbors', y='scaled_mse_over_all_batches', err_style='bars'
    )
    ax.get_figure().savefig('scaled_mse_over_all_batches.png')


if __name__ == '__main__':
    main()
