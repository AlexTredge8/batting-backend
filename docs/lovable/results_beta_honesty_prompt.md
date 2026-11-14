# Lovable prompt: show the score honestly (beta label, capture warnings, "not assessed" pillars)

Paste everything below the line into Lovable.

---

Update the analysis results page (`src/pages/AnalysisResults.tsx` and the analysis types in `src/types/analysis.ts`) to show three new fields the backend now returns. Do not change the upload flow, the API call or any other page.

**1. Types.** In `src/types/analysis.ts`, extend the types as follows. All fields are optional, so older results still render.
- `PillarResult` gets:
  - `assessed?: boolean`
  - `imputed?: boolean`
  - `raw_score?: number`
  - `basis?: "fault_found" | "no_fault_found" | "not_measured" | "suppressed"`
  - `rules?: string[]`
- The analysis response gets `warnings?: string[]` and `analysis_quality?: { scoring_status?: "beta"; scoring_note?: string; capture?: { batter_height_fraction?: number | null; duration_s?: number }; audio_available?: boolean; reliable?: boolean; detection_rate?: number }`.

**2. Beta label.** If `analysis_quality.scoring_status === "beta"`:
- Show a small pill reading **BETA** next to the overall BattingIQ score. Use a muted outline style, not a warning colour.
- Show `analysis_quality.scoring_note` in small muted text directly under the score.
- If the field is missing, use this text: "Beta: the score is being validated against qualified coaches. Treat it as a guide, and read it with the coaching points and storyboard."
- Do not hide or change the number.

**3. Filming tips (warnings).** If `warnings` has at least one item, show a card titled **"Tips to get a more accurate analysis"** above the pillar breakdown.
- List each warning as its own line, in the order given. Show the text exactly as sent; the backend writes it in plain English.
- Use a neutral information style (an info icon and a soft background). Do not use red or the word "error".
- If `warnings` is empty or missing, show nothing.

**4. Pillars that could not be measured.** For each pillar (access, tracking, stability, flow):
- If `assessed === false`:
  - Show the pillar's score as it is sent. When `imputed` is true, the backend has already replaced the old automatic 25 with the average of the other pillars.
  - Show a grey chip reading **"Estimated"** in place of the green/amber/red status colour.
  - Under it, show: "We could not measure this part of the shot in this clip, so this score is the average of your other areas." If `basis === "suppressed"`, add: "A key moment in the clip could not be pinned down."
  - Do not show the pillar's "positives" text when the pillar is not assessed.
- If `assessed` is true or missing, show the pillar exactly as today.

**5. Overall score when a pillar was estimated.** If one or more pillars have `assessed === false`, add one line under the overall score: "{n} of 4 areas measured; the rest estimated", where n counts the pillars that are not `assessed === false`.

**6. Order of the page.** Lead with the storyboard and the priority fix. Put the overall score after them, then the tips card, then the pillars. The number should read as a guide, not a verdict.

**Example to test with:** a result where `pillars.tracking = { score: 14, max: 25, status: "amber", faults: [], assessed: false, imputed: true, raw_score: 25, basis: "not_measured" }` and `warnings = ["The batter looks small in the picture, so body positions are measured less precisely. Move closer, so the batter fills at least a fifth of the picture's height."]`. This must show:
- Tracking showing 14 with an "Estimated" chip.
- The tips card with that one line.
- "3 of 4 areas measured; the rest estimated".
- The BETA pill.

Keep the current fonts, colours and layout everywhere else, and make sure the page still works on a phone.
