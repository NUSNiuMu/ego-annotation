# Garment review contract

The shared `garment-review-policy.json` defines vocabulary and semantic acceptance.
Source clocks, packet identity and original model outputs are immutable.
Candidates with `requires_image` represent unvalidated Qwen windows. Their raw
text is not a valid action; the host attaches mandatory MISSING_CONTENT questions
to affected episodes, or a temporary gap parent if none covers the window.

## Output

Return JSON with `packet_id`, `review_type` (`text` or `image`), `episodes`,
`excluded`; image output also has `issue_results`. Copy the supplied packet ID.

Each episode has exactly:

```json
{
  "id": "candidate-local-id",
  "start_frame": 100,
  "end_frame_exclusive": 900,
  "disposition": "ACCEPT",
  "reason": "Source-supported task finding.",
  "boundary_notes": "Coarse boundary limitation, or empty.",
  "evidence_refs": ["w001:s00"],
  "actions": [{
    "code": "smooth_garment",
    "start_frame": 100,
    "end_frame_exclusive": 160,
    "description": "Smooth the garment.",
    "source_refs": ["w001:s00"],
    "evidence_frames": [100, 130],
    "edit": "normalize",
    "reason": "Evidence for the change."
  }],
  "unlabeled": [],
  "issues": []
}
```

Numbers are examples. Use real packet anchors and references. Starts must be
supplied samples; exclusive ends may be samples or sample+1. All intervals are
half-open and durations use actual source time. Actions and unlabeled spans stay
within their episode, ordered and non-overlapping. Episodes do not overlap.

`edit` is `retain`, `normalize`, `merge`, `split` or `supplement`. Each action cites
supplied evidence inside its interval and originating windows; image-reviewed
actions may additionally cite the actual attached images. Use only policy codes.

An `unlabeled` span has exactly `start_frame`, `end_frame_exclusive`, `reason`,
`source_refs`, `evidence_frames`. It preserves task content without guessing an
internal action. ACCEPT requires supported task bounds, continuity and completion,
at least one action or unlabeled span, and no unresolved issues. Short duration or
internal label gaps alone do not require REVIEW.

An issue has exactly `id`, `kind`, `question`, `start_frame`,
`end_frame_exclusive`, `source_refs`, `requested_frames`. Kinds are TASK_BOUNDARY,
GARMENT_IDENTITY, COMPLETION, ACTION_CONFLICT, MISSING_CONTENT, PHONE_INTERRUPTION.
Its bounded context must overlap its episode. Frame requests are hints: the host
restricts them to real samples inside that context and records any adjustment.
Phone interruptions stay REVIEW; observed unfinished tasks are INCOMPLETE.

`excluded` rows have exactly `source_ref`, `reason`. Account for every input
candidate through task spans, an issue or an exclusion. Exclusions mean non-task
context, not uncertain task content.

## Image replacement

Every image-output episode additionally requires `source_episode_ids`: the
nonempty list of affected input episode IDs it replaces. One parent may produce
multiple children and multiple named parents may merge. Children stay within
those parents and their supplied issue context. Local output IDs only link this
response; the host generates stable final IDs and records parent relationships.
For a temporary gap parent only, images may establish non-task context: exclude
its reference and resolve all its questions, with no replacement episode.

Each `issue_results` row has exactly `issue_id`, `resolution` (RESOLVED or
UNRESOLVED), `observation`, `evidence_frames`. Cite inspected attached frames.
Unresolved issues remain attached to a replacement episode. Cover all affected
parents and questions; do not modify unrelated episodes.

## Host responsibilities

The host validates shared identity and clocks first, then isolates episode errors
and conflicting ranges. Unaffected validated episodes continue. Invalid image
replacement groups retain their parent scope as unresolved execution evidence.
Service failures use bounded retries of failed calls; completed Qwen is reused.
Raw responses stay unchanged; mechanical corrections, lineage and errors are saved
separately. Acceptance is an automated semantic finding plus structural checks,
not independent visual truth, hand completion or Viewer publication. No previews.
