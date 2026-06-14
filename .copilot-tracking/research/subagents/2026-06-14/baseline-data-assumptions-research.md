# Baseline Data Assumptions Research

## Status
Complete

## Research Scope
Primary topic: dataset assumptions and common data strategy implications for the baseline set:
- Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards (CVPR 2025)
- Stable Diffusion / Latent Diffusion Models (CVPR 2022)
- DDPO
- DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models (NeurIPS 2023)

## Questions
- Which datasets or prompt/image sources does each baseline depend on for training, fine-tuning, reward modeling, or evaluation?
- Does each baseline fundamentally need images, prompts, prompt-image pairs, preference labels, rewards, or generated outputs?
- What is the common intersection across baselines for an initial shared subset?
- What practical dataset candidates exist for a first comparable experiment, and what are the trade-offs?

## Findings

### Baseline-by-Baseline Data Assumptions

#### 1) Stable Diffusion / Latent Diffusion Models (CVPR 2022)

- Core training assumption: text-to-image training requires large-scale image-caption pairs, not prompt-only data.
- Stable Diffusion v1 is explicitly trained on LAION-5B subsets. The official model card states that training data is "LAION-5B and subsets thereof," with checkpoints trained on `laion2B-en`, `laion-high-resolution`, and `laion-aesthetics v2 5+`.
- The original latent-diffusion repository shows broader LDM experiments across multiple datasets depending on task: ImageNet, LSUN, FFHQ, CelebA-HQ, Conceptual Captions, and OpenImages. For the text-to-image LDM line, the model zoo lists `Conceptual Captions` as a text-conditional dataset and notes that the released text-to-image checkpoint is "finetuned from LAION."
- Stable Diffusion evaluation uses prompt-only inference on random prompts from COCO 2017 validation for checkpoint comparison, but that is evaluation only; the model still fundamentally depends on paired image-text pretraining data.
- Minimal supervision/data objects needed:
	- required for pretraining: images + English captions or text prompts paired with images
	- not required for base training: preference labels or scalar rewards
	- used for evaluation/inference: prompts, generated outputs

#### 2) DDPO

- DDPO assumes a pretrained text-to-image diffusion model and does not require external real-image supervision during RL fine-tuning.
- The DDPO paper defines the RL objective over prompts `c` sampled from a chosen prompt distribution and generated images `x0`, with reward computed on `(x0, c)`. In practice, it fine-tunes Stable Diffusion v1.4 rather than retraining a diffusion model from raw image-caption pairs.
- For compressibility and incompressibility, prompts are sampled uniformly from all 398 ImageNet-1000 animal categories.
- For aesthetic quality, prompts are sampled uniformly from a smaller set of 45 common animals.
- For prompt-image alignment, prompts take the form `a(n) [animal] [activity]`, with the same 45 animals and 3 activities: `riding a bike`, `playing chess`, and `washing dishes`.
- DDPO reward inputs differ by task:
	- compressibility/incompressibility: generated image only
	- aesthetic quality: generated image scored by the LAION aesthetics predictor, which itself was trained on 176,000 human image ratings
	- prompt-image alignment: generated image + prompt, using LLaVA to describe the image and BERTScore against the original prompt
- Minimal supervision/data objects needed:
	- required for RL fine-tuning: prompts + generated outputs + reward function
	- optional/indirect dependency: a learned reward model or VLM that may itself have been trained on human annotations
	- not required at RL time: ground-truth prompt-image pairs, explicit preference labels collected for this experiment

#### 3) DPOK (NeurIPS 2023)

- DPOK also assumes a pretrained text-to-image diffusion model and fine-tunes it online with RL rather than retraining from paired image-text data.
- The DPOK abstract states that the method learns from human feedback through a reward function that captures what humans care about, then uses online RL with KL regularization to optimize the diffusion model against that learned reward.
- The open-source DPOK README exposes two prompt regimes:
	- single-prompt fine-tuning via `--single_prompt`
	- multi-prompt fine-tuning via `./dataset/drawbench/data_meta.json`
- The DPOK code path explicitly installs and uses ImageReward, so its practical reward model dependency is a feedback-trained reward model rather than raw pairwise preference labels at RL time.
- The DrawBench metadata file is prompt-only. It contains prompt strings categorized into groups such as `Colors`, `DALL-E`, `Descriptions`, `Positional`, `Reddit`, and `Text`. That means DPOK's released multi-prompt dataset surface is a prompt set, not a paired image dataset.
- DDPO's direct comparison appendix states that DPOK trains on four prompts for its headline comparison: `a green colored rabbit`, `four wolves in the park`, `a dog and a cat`, and `a dog on the moon`, using ImageReward as the reward function.
- Minimal supervision/data objects needed:
	- required for RL fine-tuning: prompts + generated outputs + scalar reward from a learned reward model
	- indirect upstream dependency: human preference data used to train the reward model (ImageReward)
	- not required at DPOK fine-tuning time: real training images paired to prompts, or fresh human labels for each rollout

#### 4) Towards Better Alignment / B2-DiffuRL (CVPR 2025)

- B2-DiffuRL is a framework layered on top of RL fine-tuning and assumes a pretrained Stable Diffusion v1.4 backbone with LoRA fine-tuning of the UNet.
- The paper samples prompts uniformly from a candidate prompt set and uses generated trajectories plus final prompt-image alignment scores as RL training data.
- Its experiments define three prompt-template training sets:
	- Template 1: `a(n) [animal] [activity]` using the same 45 common animals and 3 activities used by DDPO
	- Template 2: `[color] [fruit/vegetable]`, 40 prompts built with GPT-4 assistance
	- Template 3: `[object 1] [predicate] [object 2]`, 40 prompts based on Visual Relation Dataset annotations
- The training prompt lists are separate from held-out generalization lists, all provided in Appendix H.
- Reward functions are prompt-image alignment scores:
	- BERTScore route: generated image -> LLaVA v1.5 description -> DeBERTa/BERTScore against the prompt
	- CLIPScore route: direct text-image similarity via CLIP ViT-H/14
- B2-DiffuRL also creates a synthetic analysis dataset of 768 similar generated-image pairs to compare reward stability between BERTScore and CLIPScore, but that is for metric analysis rather than primary RL training.
- Minimal supervision/data objects needed:
	- required for RL fine-tuning: prompts + generated outputs + alignment reward function
	- not required at RL time: external real images paired with prompts
	- optional/indirect dependency: VLM/text encoder models used to compute rewards

### What Each Baseline Fundamentally Needs

| Baseline | Prompts | Images | Prompt-image pairs | Preference labels | Scalar rewards | Generated outputs |
| --- | --- | --- | --- | --- | --- | --- |
| Stable Diffusion / LDM pretraining | Yes | Yes | Yes | No | No | Used only at eval/inference |
| DDPO fine-tuning | Yes | No external real images required | No ground-truth pairs required at RL time | No direct labels required at RL time | Yes | Yes |
| DPOK fine-tuning | Yes | No external real images required | No ground-truth pairs required at RL time | Indirectly yes, upstream in reward model training | Yes | Yes |
| B2-DiffuRL fine-tuning | Yes | No external real images required | No ground-truth pairs required at RL time | No direct labels required at RL time | Yes | Yes |

### Common Intersection Across Baselines

- There are two different intersections depending on whether Stable Diffusion is treated as a pretraining baseline or as the shared starting checkpoint.

#### Strict intersection across all four papers

- The only fully safe common denominator across all four is `text/image pairing`: prompts or captions aligned to images.
- Reason: Stable Diffusion / LDM fundamentally trains on paired image-text data, while the RL papers can operate without external images once a pretrained backbone exists.
- Implication: if the project wants one dataset abstraction that can support both "train the base model" and "fine-tune with RL," the canonical raw unit should be a prompt-image pair, even if some baselines ignore the image side during RL.

#### Practical intersection for a first comparable experiment

- If the project uses an off-the-shelf Stable Diffusion checkpoint as the common starting model, then the practical intersection across DDPO, DPOK, and B2-DiffuRL is `prompt-only training lists + generated outputs + recomputable rewards`.
- This is the lowest-friction shared subset because all three RL baselines can be run from:
	- a prompt list
	- a generation procedure
	- a reward function or reward-model inference pass
- Implication: for a first comparable experiment, prompt records should be the first-class unit, with generated images and reward traces stored as derived artifacts.

### Practical Dataset Candidates For A First Comparable Experiment

#### Candidate A: DDPO/B2 Template-1 animal-activity prompts

- Definition: 45 common animals crossed with 3 activities, using the DDPO/B2 family prompts around `riding a bike`, `playing chess`, and `washing dishes`.
- Why it fits: this is the clearest overlap between DDPO and B2-DiffuRL, and it can also be represented in DPOK-style prompt-only RL.
- Advantages:
	- directly backed by DDPO and B2-DiffuRL
	- cheap to store because it is prompt-only
	- easy to regenerate outputs and recompute rewards
	- strong for studying prompt-image alignment and style collapse
- Trade-offs:
	- narrow semantic domain
	- encourages cartoon/style shortcuts already documented in DDPO and B2-DiffuRL
	- weak coverage of attributes, counting, or spatial relations

#### Candidate B: B2-DiffuRL three-template prompt suite

- Definition: Template 1 animal-activity prompts plus Template 2 color-fruit/vegetable prompts plus Template 3 spatial-relation prompts.
- Why it fits: broader prompt coverage while staying prompt-only and reward-recomputable.
- Advantages:
	- spans action, attribute, and relation alignment
	- already split into train and held-out generalization prompts in the paper
	- better coverage for cross-baseline generalization analysis than Template 1 alone
- Trade-offs:
	- departs from original DDPO's exact main setup
	- includes prompts curated through different processes, including GPT-4-assisted construction for Template 2
	- still not suitable for reproducing base-model pretraining

#### Candidate C: DPOK four-prompt benchmark

- Definition: the four prompts used in the DDPO vs DPOK comparison: color, count, composition, and location prompts.
- Why it fits: the smallest DPOK-faithful comparable setup.
- Advantages:
	- minimal cost
	- directly aligned to DPOK-style reward-model fine-tuning
	- useful as a smoke-test slice
- Trade-offs:
	- too small for meaningful shared-subset strategy by itself
	- high overfitting risk
	- poor coverage of prompt diversity and weak generalization analysis

#### Candidate D: DPOK DrawBench prompt subset

- Definition: a selected subset of prompts from `dpok/dataset/drawbench/data_meta.json` spanning categories such as Colors, Positional, Text, Descriptions, and Reddit.
- Why it fits: broader DPOK-compatible prompt-only corpus.
- Advantages:
	- prompt-only and easy to version
	- category labels already exist in the metadata
	- much broader prompt surface than the four-prompt setup
- Trade-offs:
	- no paired reference images in the released metadata
	- category distribution may not match DDPO/B2 prompt families
	- requires choosing a reward function that can score very heterogeneous prompts consistently

#### Candidate E: LAION-derived text-image pairs

- Definition: a small licensed subset of LAION-style text-image pairs or another public captioned image corpus.
- Why it fits: necessary only if the project wants one dataset object that can also support base-model or LDM-style supervised training.
- Advantages:
	- satisfies the strict all-baseline intersection
	- can later support supervised fine-tuning or reward-model training
	- preserves a path back to Stable Diffusion data assumptions
- Trade-offs:
	- largest licensing and storage burden
	- not necessary for first-pass RL fine-tuning comparisons
	- most expensive option operationally

### Recommended Initial Shared Subset

- Best first comparable experiment: start with a prompt-only subset centered on Candidate A or Candidate B, using a public pretrained Stable Diffusion checkpoint as the shared backbone.
- Most balanced choice: Candidate B, but stored so Template 1 can be isolated as a DDPO/B2-overlap slice and a DPOK-compatible reward-model experiment can run on the same prompt table.
- Recommended first-class record fields:
	- prompt_id
	- prompt_text
	- source_family (`ddpo_animals`, `b2_template2`, `b2_template3`, `dpok_drawbench`, etc.)
	- semantic_tag (`activity`, `attribute`, `relation`, `count`, `composition`, `text`)
	- split (`train`, `eval_generalization`)
	- optional source citation
- Recommended derived artifacts, not canonical raw data:
	- generated image path or URI
	- seed
	- checkpoint/backbone identifier
	- reward scores by scorer (`clipscore`, `bertscore`, `imagereward`, `aesthetic`)
	- VLM caption or intermediate text description when applicable

### Main Data Strategy Implications

- Do not force one monolithic dataset abstraction for both Stable Diffusion pretraining and RL fine-tuning unless the project truly plans to train or SFT a base diffusion model. Their minimal data units differ.
- For RL baselines, prompts are the stable raw asset; images and rewards are experiment outputs.
- DPOK is the main method in this set whose practical dependency is a learned reward model grounded in human preference data. That means a shared subset for DPOK-comparability should make reward recomputation easy and preserve prompt taxonomy.
- DDPO and B2-DiffuRL share the cleanest common prompt family through the animal/activity setup, making that the safest anchor slice.
- If later work needs to connect back to SD/LDM training assumptions, add a second tier of paired caption-image data rather than overloading the RL prompt subset.

## Sources

- Stable Diffusion v1 model card, `Training` and `Evaluation Results`: [https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md)
- Stable Diffusion README, `Stable Diffusion v1` weights and training-data notes: [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- Latent Diffusion repository, `Train your own LDMs` and `Model Zoo`: [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- DDPO paper HTML, especially Sections 4.1, 5.2, 5.3, 6.1, 6.2, Appendix C, Appendix D, Appendix F: [https://arxiv.org/html/2305.13301v4](https://arxiv.org/html/2305.13301v4)
- DDPO project site: [https://rl-diffusion.github.io/](https://rl-diffusion.github.io/)
- B2-DiffuRL paper HTML, especially Sections 3.1, 4.1, 4.6, Appendix F, Appendix H: [https://arxiv.org/html/2503.11240v2](https://arxiv.org/html/2503.11240v2)
- B2-DiffuRL arXiv abstract and metadata: [https://arxiv.org/abs/2503.11240](https://arxiv.org/abs/2503.11240)
- DPOK abstract page: [https://arxiv.org/abs/2305.16381](https://arxiv.org/abs/2305.16381)
- DPOK repository README: [https://github.com/google-research/google-research/tree/master/dpok](https://github.com/google-research/google-research/tree/master/dpok)
- DPOK DrawBench prompt metadata: [https://github.com/google-research/google-research/blob/master/dpok/dataset/drawbench/data_meta.json](https://github.com/google-research/google-research/blob/master/dpok/dataset/drawbench/data_meta.json)

## Open Questions

- The DPOK paper text itself was not available through the same HTML route as DDPO and B2-DiffuRL during this session, so the DPOK data assumptions here rely on the abstract page plus the official released code and prompt metadata.
- It remains unclear whether the project wants strict all-baseline comparability including base-model supervised training, or practical comparability across RL fine-tuning methods only. That decision changes whether prompt-image pairs are mandatory in the first shared subset.
- If a paired caption-image subset is required later, licensing and redistribution policy for any LAION-derived sample will need separate confirmation.
