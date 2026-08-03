"""Shared prompt definitions for baseline experiments.

This module provides default training and evaluation prompts for diffusion model
baselines. The separation between train and eval prompts enables better evaluation
methodology while maintaining thematic consistency.

Design rationale:
- Training prompts are used during RL rollout collection
- Evaluation prompts are held-out and used only for evaluation
- Eval prompts have thematic overlap with training but are distinct
- This prevents eval reward inflation from memorization

B2-DiffuRL Templates:
- Template 1 (activity): 135 prompts - animal-activity patterns, DDPO overlap
- Template 2 (attribute): 40 prompts - color-attribute patterns
- Template 3 (relation): 40 prompts - spatial-relational patterns
- Total: 215 prompts across three semantic families

Cross-baseline comparison:
Use B2_FULL_TRAIN_PROMPTS and B2_FULL_EVAL_PROMPTS for baselines requiring
complete semantic coverage. Individual template sets enable isolating performance
by semantic category (activity vs. attribute vs. spatial reasoning).

Usage Examples:
    # Example 1: Template 1 only (DDPO overlap focus)
    from clover.utils.prompts import TEMPLATE_1_TRAIN_PROMPTS, TEMPLATE_1_EVAL_PROMPTS
    
    config.train_prompts = TEMPLATE_1_TRAIN_PROMPTS  # 90 prompts
    config.eval_prompts = TEMPLATE_1_EVAL_PROMPTS    # 45 prompts
    
    # Example 2: Full B2 coverage
    from clover.utils.prompts import B2_FULL_TRAIN_PROMPTS, B2_FULL_EVAL_PROMPTS
    
    config.train_prompts = B2_FULL_TRAIN_PROMPTS  # 150 prompts
    config.eval_prompts = B2_FULL_EVAL_PROMPTS    # 65 prompts
    
    # Example 3: Custom template mix
    from clover.utils.prompts import (
        TEMPLATE_1_TRAIN_PROMPTS,
        TEMPLATE_3_TRAIN_PROMPTS,
        TEMPLATE_1_EVAL_PROMPTS,
        TEMPLATE_3_EVAL_PROMPTS,
    )
    
    # Activities + spatial relations (skip attributes)
    CUSTOM_TRAIN = (*TEMPLATE_1_TRAIN_PROMPTS, *TEMPLATE_3_TRAIN_PROMPTS)
    CUSTOM_EVAL = (*TEMPLATE_1_EVAL_PROMPTS, *TEMPLATE_3_EVAL_PROMPTS)

Migration path:
For manifest-based datasets, this module is superseded by data/manifests/.
Override config.train_prompts and config.eval_prompts in baseline configs when
custom prompt sets are needed.
"""
import random

# Default training prompts shared across baselines
DEFAULT_TRAIN_PROMPTS: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a close-up photo of a bright green clover leaf with dew",
    "a small robot holding a clover in a clean studio photo",
    "an impressionist painting of clovers under warm sunlight",
)

# Default evaluation prompts (separate from training for better evaluation methodology)
DEFAULT_EVAL_PROMPTS: tuple[str, ...] = (
    "a colorful clover field at sunrise, high detail",
    "a photorealistic clover close-up with water droplets",
    "a robot holding a four-leaf clover, studio lighting",
    "an oil painting of clovers in golden hour light",
)


# =============================================================================
# B2-DiffuRL Template 1: Animal-Activity Patterns
# =============================================================================
"""
B2-DiffuRL Template 1: Animal-Activity Patterns

This template implements the canonical DDPO/B2-DiffuRL overlap slice (ddpo_b2_shared).
It provides 135 prompts combining 45 animals with 3 activities, testing compositional
generation and action understanding.

Source: B2-DiffuRL CVPR 2025, Appendix H
Pattern: a(n) [animal] [activity]
Semantic coverage: activity, compositional
Animal count: 45 (matching DDPO baseline for reproducibility)
Activity count: 3 (riding a bike, playing chess, washing dishes)
Total prompts: 135 (45 animals × 3 activities)

Split strategy:
- Train: First 30 animals × 3 activities = 90 prompts (66.7%)
- Eval: Last 15 animals × 3 activities = 45 prompts (33.3%)
- Animal-level separation ensures no animal appears in both train and eval

This split preserves compositional generalization testing while maintaining
the semantic diversity of the original B2-DiffuRL experiments.

Example prompts:
- "a dog riding a bike"
- "a cat playing chess"
- "an elephant washing dishes"
"""

# Template 1 constants
TEMPLATE_1_ANIMALS: tuple[str, ...] = (
    "dog", "cat", "elephant", "rabbit", "fox", "lion", "tiger", "bear",
    "monkey", "giraffe", "zebra", "horse", "cow", "sheep", "pig", "goat",
    "chicken", "duck", "penguin", "owl", "eagle", "parrot", "flamingo", "swan",
    "turtle", "frog", "fish", "dolphin", "whale", "shark", "octopus",
    "snake", "lizard", "crocodile", "ant", "bee", "butterfly", "spider",
    "squirrel", "mouse", "rat", "hedgehog", "raccoon", "deer", "moose",
)  # 45 animals - matches DDPO baseline for reproducibility

TEMPLATE_1_ACTIVITIES: tuple[str, ...] = (
    "riding a bike",
    "playing chess",
    "washing dishes",
)  # 3 activities from B2-DiffuRL Appendix H


def generate_template_1_prompts() -> tuple[str, ...]:
    """Generate all 135 Template 1 prompts (45 animals × 3 activities).
    
    Returns Cartesian product in deterministic order with correct article usage.
    Articles are selected based on vowel-initial animals ("an" vs "a").
    
    Returns:
        Tuple of 135 prompts following pattern "a(n) [animal] [activity]"
    
    Example outputs:
        - "a dog riding a bike"
        - "an elephant playing chess"
        - "a cat washing dishes"
    """
    prompts = []
    for animal in TEMPLATE_1_ANIMALS:
        for activity in TEMPLATE_1_ACTIVITIES:
            article = "an" if animal[0] in "aeiou" else "a"
            prompts.append(f"{article} {animal} {activity}")
    return tuple(prompts)


# Generate all 135 Template 1 prompts
TEMPLATE_1_ALL_PROMPTS = generate_template_1_prompts()

# Template 1 splits (90 train / 45 eval based on B2-DiffuRL paper ratios)
# Train: first 30 animals × 3 activities = 90 prompts (66.7%)
# Eval: last 15 animals × 3 activities = 45 prompts (33.3%)
# This split preserves animal-level separation (no animal appears in both splits)
TEMPLATE_1_TRAIN_PROMPTS = TEMPLATE_1_ALL_PROMPTS[:90]  # First 30 animals
TEMPLATE_1_EVAL_PROMPTS = TEMPLATE_1_ALL_PROMPTS[90:]   # Last 15 animals


# =============================================================================
# B2-DiffuRL Template 2: Color-Attribute Patterns
# =============================================================================
"""
B2-DiffuRL Template 2: Color-Attribute Patterns

This template tests attribute binding with 40 prompts combining colors with
fruits/vegetables. Generated via GPT-4 assistance per B2-DiffuRL methodology.

Source: B2-DiffuRL CVPR 2025, Appendix H
Pattern: [color] [fruit/vegetable]
Semantic coverage: attribute, color
Total prompts: 40

Split strategy:
- Train: First 30 prompts = 30 prompts (75%)
- Eval: Last 10 prompts = 10 prompts (25%)
- Sequential split maintains distribution of color-fruit combinations

This template complements Template 1's action focus by testing attribute
binding and color-object associations.

Example prompts:
- "red apple"
- "yellow banana"
- "purple grape"
"""

# Template 2 constants - 40 color-attribute prompts from Appendix H
TEMPLATE_2_PROMPTS: tuple[str, ...] = (
    "red apple",
    "green apple",
    "yellow banana",
    "brown banana",
    "orange orange",
    "yellow orange",
    "red strawberry",
    "green strawberry",
    "purple grape",
    "green grape",
    "red watermelon",
    "green watermelon",
    "brown kiwi",
    "green kiwi",
    "orange mango",
    "yellow mango",
    "green pear",
    "yellow pear",
    "yellow pineapple",
    "brown pineapple",
    "orange peach",
    "yellow peach",
    "purple plum",
    "green plum",
    "blue blueberry",
    "purple blueberry",
    "red raspberry",
    "green raspberry",
    "yellow lemon",
    "green lemon",
    "green lime",
    "yellow lime",
    "green avocado",
    "brown avocado",
    "red cherry",
    "green cherry",
    "red pomegranate",
    "pink pomegranate",
    "pink grapefruit",
    "red grapefruit",
)  # 40 prompts - GPT-4 assisted generation per B2-DiffuRL

# Template 2 splits (30 train / 10 eval for ~75/25 split)
# Train: first 30 prompts (75%)
# Eval: last 10 prompts (25%)
TEMPLATE_2_TRAIN_PROMPTS = TEMPLATE_2_PROMPTS[:30]  # First 30 prompts
TEMPLATE_2_EVAL_PROMPTS = TEMPLATE_2_PROMPTS[30:]   # Last 10 prompts


# =============================================================================
# B2-DiffuRL Template 3: Spatial-Relational Patterns
# =============================================================================
"""
B2-DiffuRL Template 3: Spatial-Relational Patterns

This template tests spatial reasoning with 40 prompts combining objects with
spatial predicates. Based on Visual Relation Dataset per B2-DiffuRL methodology.

Source: B2-DiffuRL CVPR 2025, Appendix H
Pattern: [object_1] [predicate] [object_2]
Semantic coverage: relation, spatial
Total prompts: 40

Spatial predicates used:
- Vertical: "under", "on"
- Horizontal: "on the left of", "on the right of"

Split strategy:
- Train: First 30 prompts = 30 prompts (75%)
- Eval: Last 10 prompts = 10 prompts (25%)
- Sequential split maintains distribution of spatial predicates

This template complements Templates 1 and 2 by testing spatial relation
understanding and multi-object composition.

Example prompts:
- "chair under umbrella"
- "person on street"
- "dog on the right of vase"
"""

# Template 3 constants - 40 spatial-relational prompts from Appendix H
TEMPLATE_3_PROMPTS: tuple[str, ...] = (
    "chair under umbrella",
    "table under umbrella",
    "car on street",
    "wheel on train",
    "airplane on street",
    "bag on street",
    "tree under sky",
    "building under sky",
    "street under sky",
    "dog on boat",
    "tower under sky",
    "cup on shirt",
    "person on street",
    "laptop on table",
    "table under laptop",
    "person on sofa",
    "glasses on face",
    "sofa under person",
    "table under vase",
    "street under car",
    "dog on the right of vase",
    "building on the right of building",
    "suitcase on the left of person",
    "dog on the left of person",
    "kite on the right of kite",
    "person on the left of ball",
    "ball on the right of person",
    "road on the left of grass",
    "grass on the right of road",
    "person on the left of pillow",
    "bowl on the right of plate",
    "building on the right of truck",
    "person on the left of bottle",
    "bottle on the right of person",
    "box on the left of post",
    "building on the left of building",
    "car on the right of car",
    "truck on the right of car",
    "car on the left of car",
    "person on the left of person",
)  # 40 prompts - based on Visual Relation Dataset per B2-DiffuRL

# Template 3 splits (30 train / 10 eval for ~75/25 split)
# Train: first 30 prompts (75%)
# Eval: last 10 prompts (25%)
TEMPLATE_3_TRAIN_PROMPTS = TEMPLATE_3_PROMPTS[:30]  # First 30 prompts
TEMPLATE_3_EVAL_PROMPTS = TEMPLATE_3_PROMPTS[30:]   # Last 10 prompts


# =============================================================================
# Unified B2 Prompt Sets
# =============================================================================
"""
Unified B2 Prompt Sets

Combined prompt sets spanning all three B2-DiffuRL template families.
Use these for baselines requiring complete semantic coverage across
activities, attributes, and spatial relations.

Total: 215 prompts (135 + 40 + 40)
Train: 150 prompts (90 + 30 + 30)
Eval: 65 prompts (45 + 10 + 10)

Semantic coverage:
- Activity understanding (Template 1: animal-activity compositions)
- Attribute binding (Template 2: color-object associations)
- Spatial reasoning (Template 3: multi-object spatial relations)

Usage:
These unified sets provide a comprehensive benchmark for evaluating diffusion
models across multiple reasoning capabilities. Use B2_FULL_TRAIN_PROMPTS during
RL training and B2_FULL_EVAL_PROMPTS for held-out evaluation to ensure models
generalize across all three semantic families.

For targeted evaluation, use individual template train/eval splits to isolate
performance by semantic category.
"""
random.seed(123)

# Combined B2 template sets for baselines that want all templates
B2_FULL_TRAIN_PROMPTS = (
    *TEMPLATE_1_TRAIN_PROMPTS,
    *TEMPLATE_2_TRAIN_PROMPTS,
    *TEMPLATE_3_TRAIN_PROMPTS,
)  # 150 train prompts total (90 + 30 + 30)
B2_FULL_TRAIN_PROMPTS = tuple(random.sample(B2_FULL_TRAIN_PROMPTS, len(B2_FULL_TRAIN_PROMPTS)))  # Shuffle for training


B2_FULL_EVAL_PROMPTS = (
    *TEMPLATE_1_EVAL_PROMPTS,
    *TEMPLATE_2_EVAL_PROMPTS,
    *TEMPLATE_3_EVAL_PROMPTS,
)  # 65 eval prompts total (45 + 10 + 10)
B2_FULL_EVAL_PROMPTS = tuple(random.sample(B2_FULL_EVAL_PROMPTS, len(B2_FULL_EVAL_PROMPTS)))  # Shuffle for evaluation
