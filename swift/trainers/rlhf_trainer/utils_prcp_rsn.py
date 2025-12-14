import hashlib
from functools import partial
from typing import Dict, Any, TYPE_CHECKING

import numpy as np
from PIL import Image
from accelerate.utils import gather, gather_object

from .rollout_mixin import DataType
from ...llm.template.vision_utils import load_image
if TYPE_CHECKING:
    from .grpo_trainer import GRPOTrainer


def _stable_prompt_seed(prompt_id: str, salt: str = "corrupt_prompt_v1") -> int:
    s = f"{salt}:{prompt_id}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def get_corrupted_images_for_input(trainer: 'GRPOTrainer', inputs: DataType) -> DataType:
    corrupt_image: str = trainer.corrupt_image
    corrupt_image_kwargs: Dict[str, Any] = trainer.corrupt_image_kwargs
    corrupt_image_position: str = trainer.corrupt_image_position

    if corrupt_image == 'no_image':
        return inputs
    elif corrupt_image == 'random_patch':
        corrupt_func = partial(random_patch_blackening, patch_size=trainer.model.config.vision_config.patch_size, **corrupt_image_kwargs)
    else:
        raise NotImplementedError(f"Corrupt image method '{corrupt_image}' is not implemented.")

    if corrupt_image_position == 'completion':
        for inp in inputs:
            if 'images' in inp:
                corrupted_images = []
                for img in inp['images']:
                    if isinstance(img, dict):
                        img = (img.get('bytes') or img.get('path')) if 'bytes' in img else img['path']
                    img = load_image(img)
                    corrupted_images.append(corrupt_func(img))
                inp['corrupted_images'] = corrupted_images

    elif corrupt_image_position == 'prompt':
        prompt_cache: Dict[Any, Any] = {}
        for inp in inputs:
            if 'images' not in inp:
                continue
            if 'prompt_id' not in inp:
                raise KeyError("corrupt_image_position='prompt' requires each input to have 'prompt_id'.")

            pid = inp['prompt_id'] + f'_step:{trainer._step}'
            if pid in prompt_cache:
                inp['corrupted_images'] = prompt_cache[pid]
                continue

            seed_base = _stable_prompt_seed(pid)
            corrupted_images = []

            for img_idx, img in enumerate(inp['images']):
                if isinstance(img, dict):
                    img = (img.get('bytes') or img.get('path')) if 'bytes' in img else img['path']
                img = load_image(img)
                corrupted_images.append(corrupt_func(img, seed=seed_base+img_idx))

            prompt_cache[pid] = corrupted_images
            inp['corrupted_images'] = corrupted_images
    
    else:
        raise ValueError(f"'corrupt_image_position' can only be 'prompt' or 'completion', got '{corrupt_image_position}'.")
    
    return inputs


def random_patch_blackening(pil_img, patch_size=14, black_prob=0.6, seed=None):
    """Randomly blacken square patches in a PIL image (deterministic if seed provided)."""
    rng = np.random.default_rng(seed)

    img = np.array(pil_img)
    h, w = img.shape[:2]

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if rng.random() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                img[y:y_end, x:x_end, ...] = 0

    return Image.fromarray(img)
