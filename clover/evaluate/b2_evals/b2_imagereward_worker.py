"""ImageReward process entry point; invoked by the B2 notebook helper."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    import ImageReward as RM
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('request',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    network=RM.load('ImageReward-v1.0',device=args.device,download_root=request['download_root'])
    scores=[]
    with torch.inference_mode():
        for row in tqdm(request['records'],desc='ImageReward'):
            score=float(network.score(row['prompt'],row['image_path']))
            scores.append({**row,'imagereward':score})
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(scores,allow_nan=False))
    temporary.replace(args.output)


if __name__=='__main__':
    main()
