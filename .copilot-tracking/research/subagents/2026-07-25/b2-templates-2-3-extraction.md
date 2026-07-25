# B2-DiffuRL Templates 2 & 3 Extraction Research

## Research Topics

1. Access the B2-DiffuRL paper at https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf
2. Navigate to Appendix H
3. Extract all 40 Template 2 prompts (color-attribute pattern: [color] [fruit/vegetable])
4. Extract all 40 Template 3 prompts (spatial-relational pattern: [object_1] [predicate] [object_2])
5. Document train/eval split assignments per template

## Research Status

Status: **BLOCKED** - PDF extraction failure prevents completion
Last Updated: 2026-07-25

**Blocker:** Unable to extract text content from B2-DiffuRL CVPR 2025 paper PDF via automated tools. PDF parsing fails on both main paper and supplementary materials from CVF Open Access repository.

**Progress:**
- ✅ Confirmed paper location and structure
- ✅ Documented Template 2 & 3 patterns and specifications
- ✅ Identified example prompts from existing workspace research
- ❌ Complete 40-prompt lists for Template 2 not extracted
- ❌ Complete 40-prompt lists for Template 3 not extracted  
- ❌ Train/eval split assignments not determined

## Findings

### Paper Access Status

**BLOCKED**: Unable to extract PDF content from CVF Open Access repository

Attempted Access Methods:
1. Direct PDF fetch from https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf
   - Result: Failed to extract meaningful content (PDF parsing issue)
2. Supplementary material fetch from https://openaccess.thecvf.com/content/CVPR2025/supplemental/Hu_Towards_Better_Alignment_CVPR_2025_supplemental.pdf
   - Result: Failed to extract meaningful content
3. arXiv search for "B2-DiffuRL Hu"
   - Result: No results found (paper may not be on arXiv)
4. GitHub repository search
   - Result: HTTP 429 rate limit
5. HuggingFace papers search
   - Result: HTTP 401 authentication error

**Alternative Sources Checked:**
- Workspace contains existing research that acknowledges Templates 2 & 3 require Appendix H extraction
- .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md documents the gap
- .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md confirms extraction dependency

### Existing Research Findings

The workspace contains comprehensive research on B2-DiffuRL templates but explicitly notes that complete Template 2 & 3 prompt lists require manual extraction:

**From b2-template-specs.md:**
> "Gaps in Specification:
> - Exact color list not detailed in available research documents
> - Exact fruit/vegetable list not detailed in available research documents  
> - Full prompt list requires reading B2-DiffuRL Appendix H from paper PDF"

### Template 2 Prompts (Color-Attribute Pattern)

**Pattern:** [color] [fruit/vegetable]
**Total Expected:** 40 prompts
**Generation Method:** GPT-4 assisted construction (per B2-DiffuRL paper)
**Semantic Tags:** attribute, color
**Source Family:** b2_template2

**Example Prompts (inferred from research, NOT complete list):**
```
red apple
green grape
yellow banana
purple eggplant
orange carrot
blue blueberry
```

**Train/Eval Split:** To be determined from Appendix H

**Status:** INCOMPLETE - Only pattern and examples documented, not full 40-prompt list

### Template 3 Prompts (Spatial-Relational Pattern)

**Pattern:** [object_1] [predicate] [object_2]
**Total Expected:** 40 prompts
**Source Dataset:** Visual Relation Dataset annotations
**Semantic Tags:** relation, spatial
**Source Family:** b2_template3

**Example Prompts (inferred from research, NOT complete list):**
```
a cat to the left of a couch
a dog on the moon
a book next to a lamp
```

**Predicates (inferred from Visual Relation Dataset):**
- to the left of
- to the right of
- on
- under
- next to
- behind
- in front of
- above
- below

**Train/Eval Split:** To be determined from Appendix H

**Status:** INCOMPLETE - Only pattern, examples, and predicate types documented, not full 40-prompt list

### Appendix H Location

**Paper Details:**
- Title: "Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards"
- Authors: Hu et al.
- Conference: CVPR 2025
- URL: https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf

**Required Content:**
- Appendix H should contain complete 40-prompt lists for Template 2 and Template 3
- Train/eval split assignments for each template
- Exact color vocabulary for Template 2
- Exact object and predicate vocabulary for Template 3

## References

- B2-DiffuRL CVPR 2025 Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Towards_Better_Alignment_Training_Diffusion_Models_with_Reinforcement_Learning_Against_CVPR_2025_paper.pdf
- Source Section: Appendix H (not yet accessed)
- Workspace Research: .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md
- Workspace Research: .copilot-tracking/research/2026-07-25/b2-templates-implementation-research.md
- Implementation Plan: .copilot-tracking/plans/2026-07-25/b2-templates-plan.instructions.md
- Visual Relation Dataset: https://visualgenome.org/static/paper/Visual_Genome.pdf (source for Template 3 predicates)

## Recommended Next Research

### High Priority

- [ ] **Manual PDF Extraction:** User manually opens B2-DiffuRL paper PDF and extracts Appendix H content
  - Navigate to Appendix H in local PDF reader
  - Copy Template 2 prompts table/list (40 prompts expected)
  - Copy Template 3 prompts table/list (40 prompts expected)
  - Document train/eval split assignments for each template
  - Paste extracted content into workspace file or provide to agent

- [ ] **GitHub Repository Search (manual):** User searches for B2-DiffuRL official code repository
  - Search GitHub for "B2-DiffuRL" or author names from paper
  - Check repository for prompts data files (JSON, txt, or Python constants)
  - If found, provide repository URL for automated extraction

- [ ] **Contact Paper Authors:** User reaches out to paper authors for prompt lists
  - Request supplementary code or data release
  - Ask for Appendix H content in machine-readable format

### Medium Priority

- [ ] **Alternative Paper Sources:** Check if paper is available on other platforms
  - Search Semantic Scholar for paper and supplementary materials
  - Check author personal/lab websites for preprint versions with appendices
  - Search Google Scholar for cached versions or author copies

- [ ] **GPT-4 Regeneration (Template 2 only):** Attempt to regenerate Template 2 prompts using known pattern
  - Risk: Generated prompts may differ from paper's exact vocabulary
  - Use as placeholder until authoritative list obtained
  - Mark clearly as "REGENERATED - NOT FROM PAPER"

### Low Priority

- [ ] **Visual Relation Dataset Mining (Template 3 only):** Extract spatial predicates and object pairs from Visual Relation Dataset
  - Download Visual Relation Dataset annotations
  - Filter to most common spatial predicates
  - Sample 40 diverse object-predicate-object combinations
  - Risk: Generated prompts may differ from paper's selection
  - Mark clearly as "INFERRED FROM VRD - NOT FROM PAPER"

## Follow-on Questions

1. **PDF Access:** Does the user have local access to the B2-DiffuRL CVPR 2025 paper PDF? If yes, can they:
   - Share the paper PDF file directly in the workspace
   - Manually navigate to Appendix H and extract the prompt lists
   - Take screenshots of Appendix H for OCR-based extraction

2. **Alternative Sources:** Is there a GitHub repository, supplementary code, or dataset release for B2-DiffuRL that includes the prompt lists?

3. **Prompt Generation:** Given the GPT-4 assisted generation method for Template 2, should we:
   - Attempt to regenerate similar prompts using the known pattern?
   - Wait for manual extraction from Appendix H?
   - Risk: Generated prompts may not match the exact vocabulary/combinations from the paper

4. **Partial Implementation:** Should we proceed with:
   - Template 1 implementation only (already complete in workspace)
   - Placeholder Template 2 & 3 constants with example prompts marked as incomplete
   - Wait until complete extraction is possible

5. **Train/Eval Split Ratios:** The existing research suggests ~30 train / ~10 eval for 40-prompt templates. Can the user confirm:
   - Is this ratio consistent with Template 1's 90 train / 45 eval (2:1 ratio)?
   - Are the exact split points documented in Appendix H?

## Key Discoveries

### Discovery 1: PDF Extraction Tools Cannot Parse CVPR PDF Format

**Evidence:** Multiple automated fetch attempts failed to extract text content from B2-DiffuRL paper PDFs
- CVF Open Access repository uses PDF format incompatible with standard web scraping tools
- Supplementary materials also inaccessible via automated extraction
- This is a known limitation of academic PDF repositories

**Implication:** Manual extraction or alternative access methods required for Appendix H content

### Discovery 2: Workspace Already Documents Template Patterns and Examples

**Evidence:** .copilot-tracking/research/subagents/2026-07-25/b2-template-specs.md contains:
- Complete Template 2 pattern specification: [color] [fruit/vegetable]
- Complete Template 3 pattern specification: [object_1] [predicate] [object_2]
- Example prompts for both templates
- Expected counts (40 prompts each)
- Generation methods (GPT-4 assisted for Template 2, Visual Relation Dataset for Template 3)

**Implication:** We have sufficient specification to understand the templates, but lack the exact 40-prompt vocabulary lists

### Discovery 3: Template 1 Implementation Is Already Complete

**Evidence:** 
- clover/utils/prompts.py contains full Template 1 implementation (lines confirmed via grep)
- TEMPLATE_1_ANIMALS: 45 animals
- TEMPLATE_1_ACTIVITIES: 3 activities  
- TEMPLATE_1_ALL_PROMPTS: 135 total prompts (45 × 3)
- TEMPLATE_1_TRAIN_PROMPTS: 90 prompts (first 30 animals)
- TEMPLATE_1_EVAL_PROMPTS: 45 prompts (last 15 animals)

**Validation Output:**
```
No overlap: True
Last train prompt: a moose riding a bike
First eval prompt: a snake playing chess
Train uses animals 0-29: ['dolphin', 'whale', 'shark']
Eval uses animals 30-44: ['snake', 'lizard', 'crocodile']
```

**Implication:** Only Templates 2 & 3 remain for complete implementation

### Discovery 4: Existing Research Acknowledges Extraction Dependency

**Evidence:** From b2-template-specs.md:
> "Gaps in Specification:
> - Exact color list not detailed in available research documents
> - Exact fruit/vegetable list not detailed in available research documents
> - Full prompt list requires reading B2-DiffuRL Appendix H from paper PDF"

And from b2-templates-implementation-research.md Phase 2 plan:
> "Step 2.1: Extract Template 2 and 3 prompts from B2-DiffuRL Appendix H
> * URL: https://openaccess.thecvf.com/content/CVPR2025/papers/...
> * Navigate to Appendix H
> * Extract all 40 Template 2 prompts (color-attribute pattern)
> * Extract all 40 Template 3 prompts (spatial-relational pattern)"

**Implication:** This is a known gap in the implementation plan, not a new discovery

### Discovery 5: Implementation Plan Phase Dependencies

**Evidence:** From b2-templates-plan.instructions.md:
- Phase 1: Template 1 Generator and Splits - **COMPLETE** ✅
- Phase 2: Templates 2 and 3 Constants - **BLOCKED** ❌ (requires Appendix H extraction)
- Phase 3: Unified B2 Prompt Sets - **BLOCKED** (depends on Phase 2)
- Phase 4: Baseline Integration and Testing - **BLOCKED** (depends on Phase 3)

**Implication:** Completing the full B2 template implementation requires resolving the PDF extraction blocker

### Discovery 6: Example Prompts Are Inferred, Not Authoritative

**Evidence:** All Template 2 & 3 examples in research docs are marked as "inferred" or "example":
- "Example Prompts (inferred from pattern)"
- "Example Prompts (inferred from research, NOT complete list)"
- Examples include generic combinations like "red apple", "green grape", "a cat to the left of a couch"

**Implication:** Cannot use inferred examples as authoritative source for implementation - need actual Appendix H content for reproducibility
