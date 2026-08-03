from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@torch.no_grad()
def aesthetic_proxy_reward(images: list[Image.Image], prompts: list[str], device: torch.device | str | None = None) -> Tensor:
    """Small offline reward placeholder for preference-model based rewards."""
    rewards = []
    for image in images:
        arr = torch.from_numpy(np.asarray(image).astype("float32") / 255.0)
        mean_rgb = arr.mean(dim=(0, 1))
        brightness = arr.mean()
        saturation = (mean_rgb.max() - mean_rgb.min()).clamp_min(0.0)
        contrast = arr.std().clamp_max(0.35)
        rewards.append(0.45 * brightness + 0.35 * saturation + 0.20 * contrast)
    reward_tensor = torch.stack(rewards)
    return reward_tensor.to(device) if device is not None else reward_tensor


@torch.no_grad()
def clip_reward(images: list[Image.Image], prompts: list[str], device: torch.device | str | None = None) -> Tensor:
    """Compute CLIP scores for text-image alignment.

    Args:
        images: List of PIL images to evaluate
        prompts: List of text prompts (must match number of images)
        device: Device to run the model on (defaults to CUDA if available)

    Returns:
        Tensor of CLIP scores for each image-prompt pair
    """
    import open_clip

    if len(images) != len(prompts):
        raise ValueError(f"Expected one prompt per image, got {len(images)} images and {len(prompts)} prompts.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    # Preprocess images
    processed_images = torch.stack([
        preprocess(image.convert("RGB")) for image in images
    ]).to(device)

    # Tokenize text
    text = tokenizer(prompts).to(device)

    # Compute features
    autocast_device = "cuda" if device == "cuda" or str(device).startswith("cuda") else "cpu"
    with torch.autocast(autocast_device, enabled=autocast_device == "cuda"):
        image_features = model.encode_image(processed_images)
        text_features = model.encode_text(text)

    # Normalize and compute similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    scores = (image_features * text_features).sum(dim=-1)

    return scores.cpu()


@torch.no_grad()
def clip_prompt_cosine_similarity(
    prompt_a: str,
    prompt_b: str,
    device: torch.device | str | None = None,
) -> Tensor:
    """Measure semantic similarity between two prompts with CLIP text embeddings.

    Returns CLIP cosine similarity, which is equivalent to ``1 - cosine_distance``.
    """
    import open_clip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')

    text = tokenizer([prompt_a, prompt_b]).to(device)
    autocast_device = "cuda" if device == "cuda" or str(device).startswith("cuda") else "cpu"

    with torch.autocast(autocast_device, enabled=autocast_device == "cuda"):
        text_features = model.encode_text(text)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (text_features[0] * text_features[1]).sum()
    return similarity.cpu()


@torch.no_grad()
def clip_image_cosine_similarity(
    image_a: Image.Image,
    image_b: Image.Image,
    device: torch.device | str | None = None,
) -> float:
    """Measure semantic similarity between two images using CLIP embeddings.

    The returned value is the cosine similarity between the two normalized CLIP
    image embeddings. Higher values indicate greater semantic similarity.

    Args:
        image_a: First PIL image to compare.
        image_b: Second PIL image to compare.
        device: Device to run the model on (defaults to CUDA if available).

    Returns:
        CLIP cosine similarity as a Python float.
    """
    import open_clip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    model = model.to(device).eval()

    processed_images = torch.stack(
        [preprocess(image.convert("RGB")) for image in (image_a, image_b)]
    ).to(device)
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    with torch.autocast(autocast_device, enabled=autocast_device == "cuda"):
        image_features = model.encode_image(processed_images)

    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    similarity = torch.sum(image_features[0] * image_features[1])
    return float(similarity.cpu().item())


@torch.no_grad()
def bert_reward(
    images: list[Image.Image],
    prompts: list[str],
    device: torch.device | str | None = None,
    caption_model: str = "Salesforce/blip-image-captioning-base",
    bert_model_type: str = "microsoft/deberta-large-mnli",
    lang: str = "en",
    max_new_tokens: int = 50,
    batch_size: int = 4,
) -> Tensor:
    """Compute BERTScore F1 for text-image prompt alignment.

    This function generates captions for each image using BLIP, then computes
    BERTScore F1 between generated captions and ground-truth prompts.

    Args:
        images: List of PIL images to evaluate
        prompts: Ground-truth text prompts (must match number of images)
        device: Device to run the models on (defaults to CUDA if available)
        caption_model: Hugging Face BLIP caption model for generating prompts
        bert_model_type: Hugging Face model used by BERTScore
        lang: Language code used by BERTScore
        max_new_tokens: Maximum tokens generated by BLIP caption model
        batch_size: Number of images to process in a batch

    Returns:
        Tensor of BERTScore F1 scores for each image-prompt pair
    """
    from transformers import BlipForConditionalGeneration, BlipProcessor
    from bert_score import BERTScorer

    if len(images) != len(prompts):
        raise ValueError(f"Expected one prompt per image, got {len(images)} images and {len(prompts)} prompts.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    processor = BlipProcessor.from_pretrained(caption_model)
    captioner = BlipForConditionalGeneration.from_pretrained(caption_model).to(device).eval()
    scorer = BERTScorer(
        model_type=bert_model_type,
        lang=lang,
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )
    scorer._tokenizer.model_max_length = min(scorer._tokenizer.model_max_length, 512)
    
    scores = []
    
    # Process in batches
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch_prompts = prompts[start:start + batch_size]
        
        # Convert to RGB
        rgb_images = [img.convert("RGB") for img in batch_images]
        
        # Generate captions
        inputs = processor(images=rgb_images, return_tensors="pt", padding=True).to(device)
        generated_ids = captioner.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_prompts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_prompts = [prompt.strip() for prompt in generated_prompts]
        
        # Compute BERTScore
        _, _, f1 = scorer.score(generated_prompts, batch_prompts, batch_size=batch_size)
        scores.extend(f1.cpu().tolist())
    
    return torch.tensor(scores)


# Reward function registry
REWARD_REGISTRY: dict[str, Callable[[list[Image.Image], list[str], torch.device], Tensor]] = {
    "aesthetic": aesthetic_proxy_reward,
    "clip": clip_reward,
    "bert": bert_reward,
    # Future: "imagereward": imagereward_score,
}


def get_reward_fn(reward_type: str) -> Callable[[list[Image.Image], list[str], torch.device], Tensor]:
    """Get reward function by type name.

    Args:
        reward_type: Name of reward function ("aesthetic", "clip", "bert", "imagereward")

    Returns:
        Callable reward function matching signature (images, prompts, device) -> Tensor

    Raises:
        KeyError: If reward_type is not registered
    """
    if reward_type not in REWARD_REGISTRY:
        available = ", ".join(REWARD_REGISTRY.keys())
        raise KeyError(f"Unknown reward type '{reward_type}'. Available: {available}")
    return REWARD_REGISTRY[reward_type]
