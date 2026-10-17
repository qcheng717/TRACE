"""Optimizer-update based warmup and optional cosine decay."""
import math


def lr_multiplier(step, config):
    warmup = config['warmup_steps']
    kind = config.get('lr_schedule', 'constant')
    floor = config.get('min_lr_ratio', 0.1)
    if kind not in ('constant', 'cosine') or not 0 <= floor <= 1:
        raise ValueError('Invalid learning-rate schedule')
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if kind == 'constant':
        return 1.0
    progress = min(1., max(0., (step-warmup) / max(1, config['max_steps']-warmup-1)))
    return floor + (1-floor) * (1+math.cos(math.pi*progress))/2
