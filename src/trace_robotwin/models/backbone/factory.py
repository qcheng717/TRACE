"""Construct the frozen DINOv2 backbone used by TRACE."""
from copy import deepcopy
from .dinov2 import DINOv2Backbone


def make_backbone(config):
    options = deepcopy(config)
    if options.pop('name', None) != 'dinov2':
        raise ValueError('This release supports the DINOv2 backbone')
    return DINOv2Backbone(**options)
