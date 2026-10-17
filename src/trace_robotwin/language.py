import torch

from torch import Tensor


def encode_phrases(phrases: list[str], local_model_path: str) -> Tensor:
    """Offline preprocessing only; requires explicitly supplied local weights."""
    from pathlib import Path
    if not Path(local_model_path).is_dir(): raise ValueError('local sentence model directory required')
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError('Install trace-robotwin for sentence preprocessing') from exc
    encoder=SentenceTransformer(local_model_path,local_files_only=True,device='cpu')
    encoder.eval()
    with torch.no_grad(): return encoder.encode(phrases,convert_to_tensor=True).detach().cpu()

