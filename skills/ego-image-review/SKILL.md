---
name: ego-image-review
description: Resolve listed garment-episode questions using attached source images, including evidence-supported boundary changes, episode splits or merges.
---

# Targeted image review

Read the [policy](../garment-review-policy.json), [contract](../review-contract.md)
and image packet. Inspect only the supplied actual images and listed questions.

- Resolve task boundaries, continuity, garment identity and completion from visible
  evidence. Correct actions or add visible omissions when supported; preserve short
  actions and explicit internal label gaps according to policy.
- An affected candidate can become several episodes; several affected candidates
  can form one. Each output lists its `source_episode_ids`. Preserve unrelated
  episodes; stay within the named parents and their supplied issue context.
- Return an outcome for every question. If evidence remains insufficient, retain
  the issue and mark UNRESOLVED. Phone interruptions remain REVIEW.
- For an unvalidated Qwen window, inspect its frames and adjacent context. Recover
  supported content or explicitly exclude non-task context; cite that window's images.
- Cite only images actually supplied and inspected. Evidence for actions must also
  lie inside their action intervals. Do not infer missing views or fetch more data.

Return contracted image JSON. The host records replacement lineage, assigns stable
IDs, validates each replacement group and saves failures separately. Do not rerun
Qwen, call another model, change policy, generate previews or publish results.
