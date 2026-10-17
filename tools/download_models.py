"""Fetch the exact frozen encoders used by the experiment."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import hashlib
root=Path(__file__).resolve().parents[1]
for name in ('dinov2_source','language_model_source'):
    spec=json.loads((root/'docs'/f'{name}.json').read_text())
    for item in spec['files']:
        p=hf_hub_download(spec['repo'],item['file'],revision=spec['revision'],local_dir=root/spec['path'])
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'Hash mismatch: {p}')
    print(spec['repo'],spec['revision'])
