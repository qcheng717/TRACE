"""Strict YAML configuration with relative inheritance and explicit overrides."""
from copy import deepcopy
from pathlib import Path
import yaml


def merge(base: dict, patch: dict) -> dict:
    patch=deepcopy(patch)
    replace=patch.pop('_replace_',False)
    out={} if replace else deepcopy(base)
    for k,v in patch.items():
        out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None, _seen: tuple = ()) -> dict:
    path=Path(path).resolve()
    if path in _seen: raise ValueError('cyclic configuration inheritance')
    config=yaml.safe_load(path.read_text())
    if not isinstance(config,dict): raise ValueError('config must be a mapping')
    parents=config.pop('extends',[])
    if isinstance(parents,str): parents=[parents]
    result={}
    for parent in parents: result=merge(result,load_config(path.parent/parent,_seen=(*_seen,path)))
    result=merge(result,config)
    for override in overrides or []:
        key,sep,value=override.partition('=')
        if not sep: raise ValueError('override must be dotted.key=value')
        target=result; parts=key.split('.')
        for part in parts[:-1]: target=target[part]
        if parts[-1] not in target: raise KeyError(f'unknown override {key}')
        target[parts[-1]]=yaml.safe_load(value)
    return result
