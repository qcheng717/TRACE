"""Selector freezing boundaries for RoboTwin pretraining."""
import torch


def auto_freeze_step(config, stage):
    step = config['train'].get('stage1_auto_freeze_selector_at_step')
    if step is None:
        return None
    if (stage != 'stage1' or config['model']['benchmark_version'] not in ('robotwin_v4',)
            or not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < config['train']['max_steps']):
        raise ValueError('Automatic selector freezing requires an enabled Stage-I protocol and a valid boundary')
    if config['train'].get('stage1_freeze_selector_at_step') is not None:
        raise ValueError('Do not combine automatic and manual selector freezing')
    return step


def maybe_freeze_selector(model, ema, boundary, provenance):
    if boundary is None or model.step < boundary or model.selector_frozen:
        return False
    model.freeze_stage1_selector()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.startswith('predictor.selector.') and name in ema.shadow:
                ema.shadow[name].copy_(param)
    provenance.setdefault('selector_freeze', dict(step=boundary, mode='automatic',
        online_selector='frozen_raw', target_selector='frozen_raw_no_target_ema',
        inference_ema_selector='fixed_raw', optimizer='continued', lr_schedule='continued'))
    return True
