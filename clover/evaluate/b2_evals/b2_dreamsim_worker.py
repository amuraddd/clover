"""DreamSim feature diversity worker, run in the project-local uv runtime.

Following section 4.2 of Liu et al. (ICLR 2025), measure feature variance
within a prompt. We explicitly choose trace(sample covariance), correction=1:
sum_j Var_i(e_ij), using the official normalized ensemble embeddings.
The paper does not specify the variance correction or dimension reduction.
For unit embeddings this equals half the mean squared distance, or the mean
cosine distance, over distinct pairs. Higher means more diversity.

Sources:
https://github.com/ssundaram21/dreamsim
https://proceedings.iclr.cc/paper_files/paper/2025/file/ec52962a55a83c3d860b5c9b7338422b-Paper-Conference.pdf
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def group_records(records):
    """Keep prompts, baselines and evaluation seeds independent."""
    groups = defaultdict(list)
    identities = set()
    for index, row in enumerate(records):
        identity = (row['baseline'], row['id'])
        if identity in identities:
            raise ValueError(f'Duplicate image ID: {identity}')
        identities.add(identity)
        key = (row['baseline'], row['seed'], row['template'], row['prompt_index'])
        groups[key].append(index)
    if not groups:
        raise ValueError('DreamSim requires nonempty image records')
    for key, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f'DreamSim diversity requires at least two images per prompt: {key}')
        if len({records[i]['prompt'] for i in indices}) != 1:
            raise ValueError(f'Inconsistent prompt text: {key}')
    return groups


def diversity_contributions(embeddings):
    """Return per-image contributions whose mean is trace(sample covariance)."""
    import torch
    import torch.nn.functional as F
    embeddings = embeddings.detach().to(device='cpu', dtype=torch.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or embeddings.shape[1] == 0:
        raise ValueError('Expected at least two nonempty embedding vectors')
    if not torch.isfinite(embeddings).all() or (embeddings.norm(dim=1) <= 1e-12).any():
        raise ValueError('DreamSim embeddings must be finite and nonzero')
    embeddings = F.normalize(embeddings, dim=1)
    n = embeddings.shape[0]
    return ((embeddings - embeddings.mean(dim=0)).square().sum(dim=1) * n / (n - 1)).tolist()


def score_records(records, network, preprocess, *, device='cpu', batch_size=8):
    """Embed each image once; GPU memory is bounded by the inference batch size."""
    import torch
    from PIL import Image
    from tqdm import tqdm
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    groups = group_records(records)
    scored = [dict(row) for row in records]
    network.eval().requires_grad_(False)
    with torch.inference_mode():
        for indices in tqdm(groups.values(), desc='DreamSim prompt diversity'):
            features = []
            for start in range(0, len(indices), batch_size):
                batch = []
                for index in indices[start:start + batch_size]:
                    with Image.open(records[index]['image_path']) as image:
                        batch.append(preprocess(image.convert('RGB')))
                embedding = network.embed(torch.cat(batch, dim=0).to(device))
                if embedding.ndim != 2 or embedding.shape[0] != len(batch):
                    raise ValueError('Unexpected DreamSim embedding shape')
                features.append(embedding.detach().cpu())
            contributions = diversity_contributions(torch.cat(features))
            diversity = sum(contributions) / len(indices)
            for index, contribution in zip(indices, contributions):
                scored[index].update(dreamsim_diversity=diversity,
                                     dreamsim_contribution=contribution,
                                     dreamsim_image_count=len(indices))
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda:0', choices=['cpu', 'cuda', 'cuda:0'])
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    group_records(request['records'])
    from dreamsim import dreamsim
    network, preprocess = dreamsim(pretrained=True, device=args.device,
                                  cache_dir=request['download_root'],
                                  dreamsim_type='ensemble', normalize_embeds=True,
                                  use_patch_model=False)
    scored = score_records(request['records'], network, preprocess, device=args.device,
                           batch_size=request['batch_size'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(scored, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
