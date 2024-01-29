import hydra
from omegaconf import OmegaConf
import flatten_json
import wandb


def prefix_key(d, prefix):
    return {f'{prefix}.{k}': v for k, v in d.items()}


def query_runs(entity, project, global_filters, config, summary_metrics):
    filters = {
        **global_filters,
        **prefix_key(config, 'config'),
        **prefix_key(summary_metrics, 'summary_metrics'),
    }
    api = wandb.Api()
    runs = api.runs(path=f'{entity}/{project}', filters=filters)
    return runs


def filters_cfg_main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = flatten_json.flatten_json(cfg, separator='|')
    return {}, cfg, {}


def filters_job_type(job_type):
    return {'jobType': job_type}, {}, {}


@hydra.main(version_base=None, config_path='configs', config_name='runs')
def main(cfg):
    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    global_filters, config, summary_metrics = {}, {}, {}
    for name in cfg.filters:
        if name == 'cfg_main':
            a, b, c = filters_cfg_main(cfg.main)
        elif name == 'job_type':
            a, b, c = filters_job_type(cfg.job_type)
        global_filters.update(a)
        config.update(b)
        summary_metrics.update(c)
    runs = query_runs(cfg.wandb.entity, cfg.wandb.project, global_filters, config, summary_metrics)
    if cfg.verbose:
        for r in runs:
            print(r.name, r.id)
    else:
        print(','.join(r.id for r in runs))


if __name__ == '__main__':
    main()
