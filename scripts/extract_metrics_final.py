#!/usr/bin/env python3
import torch, json, re
from pathlib import Path

results = {}
for n in range(1, 13):
    ckpt_path = Path(f"outputs_ablation/N{n}_layer{13-n}_to_12/checkpoints/checkpoint_step98000.pt")
    if not ckpt_path.exists():
        continue
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Extraire poids si fusion pondérée
        state = ckpt.get('generator_state_dict', ckpt.get('model_state_dict', ckpt))
        weights = None
        if 'layer_weights' in state:
            w_raw = state['layer_weights'].cpu().numpy()
            import numpy as np
            weights = np.exp(w_raw) / np.exp(w_raw).sum()
        
        results[f'N{n}'] = {
            'step': ckpt.get('step', 98000),
            'weights': weights.tolist() if weights is not None else None,
            'layers': list(range(13-n, 13))
        }
        print(f"✅ N={n}: {ckpt_path.name} - Poids={weights}")
    except Exception as e:
        print(f"❌ N={n}: {e}")

with open('checkpoints_info.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Fichier: checkpoints_info.json")
