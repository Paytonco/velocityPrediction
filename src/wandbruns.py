import hydra
from omegaconf import OmegaConf
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


def flatten_dict_excluding_lists(d, separator='.'):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub_d = flatten_dict_excluding_lists(v)
            for sub_k, sub_v in sub_d.items():
                res[f'{k}{separator}{sub_k}'] = sub_v
        else:
            res[k] = v

    return res


def filters_cfg_main(cfg):
    if cfg is None:
        return {}, {}, {}
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = flatten_dict_excluding_lists(cfg)
    # list inclusion for all fields
    for k, v in cfg.items():
        cfg[k] = {'$in': v if isinstance(v, list) else [v]}
    return {}, cfg, {}


def filters_job_type(job_type):
    return {'jobType': job_type}, {}, {}


def filters_tags(tags):
    return {'tags': {'$in': tags}}, {}, {}


def filters_sweep(sweep):
    return {'sweep': {'$in': sweep}}, {}, {}


@hydra.main(version_base=None, config_path='../configs', config_name='runs')
def main(cfg):
    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    global_filters, config, summary_metrics = {}, {}, {}
    for name in cfg.filters:
        if name == 'cfg_main':
            a, b, c = filters_cfg_main(cfg.get('main'))
        elif name == 'job_type':
            a, b, c = filters_job_type(cfg.job_type)
        elif name == 'tags' and len(cfg.tags) > 0:
            a, b, c = filters_tags(OmegaConf.to_container(cfg.tags))
        elif name == 'sweep':
            a, b, c = filters_sweep(OmegaConf.to_container(cfg.sweep))
        global_filters.update(a)
        config.update(b)
        summary_metrics.update(c)
    runs = query_runs(cfg.wandb.entity, cfg.wandb.project, global_filters, config, summary_metrics)
    if cfg.verbose:
        for r in runs:
            print(r.name, r.id)
            print(OmegaConf.to_yaml(OmegaConf.create(r.config)))
    else:
        print(','.join(r.id for r in runs))


if __name__ == '__main__':
    main()
