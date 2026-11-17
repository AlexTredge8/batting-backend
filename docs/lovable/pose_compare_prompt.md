# Lovable prompt: fault card ("Not this / This") under the priority fix

Paste everything below the line into Lovable. Backend details: `docs/phase6/pose_compare.md` (version 2, `pose_compare_v2`). Design spec: `docs/phase6/annotation_benchmark.md`, section 5. The backend sends this field only when `POSE_COMPARE=on`, so with it off the card never shows and nothing changes.

The cartoon pictures arrive **inside the API response** as SVG data URLs, the same way the old pictures did. Nothing needs to be uploaded or copied into the Lovable project. (The same files are in the backend repo at `assets/silhouettes/` if a static page ever needs them: copy that folder into `public/silhouettes/`.)

---

Add a **fault card** to the analysis results page (`src/pages/AnalysisResults.tsx`), directly **under the priority fix card**. It teaches the player's priority fault with two fixed cartoon pictures ("Not this" and "This"), one line on what to change, their own number against good drives, an optional look at their own video, and the drill. Do not change the upload flow, the API call, the priority fix card or any other part of the page. If an older "You vs best practice" card exists, replace it with this one.

**1. Types.** In `src/types/analysis.ts`, add an optional field to the analysis response, `pose_comparison?: PoseComparison`, with these types. Every field is optional except `available`, so older results still render.

```ts
export interface FaultSilhouette {
  fault?: string;                    // "F10"
  view?: "front_on" | "side_on" | "top_down" | "pitch_side";
  label_fault?: string | null;       // "Not this" (null for F00)
  label_correct?: string;            // "This"
  caption_fault?: string | null;     // "Head falls away"
  caption_correct?: string;          // "Head over the ball"
  caption_correct_finish?: string;   // F00 only: caption of the second "This" picture
  alt?: string;                      // alt text for the pair
  mirror?: boolean;                  // true for a left-hander: flip both pictures horizontally
  fault_svg_data_url?: string | null;
  correct_svg_data_url?: string | null;
  correct_finish_svg_data_url?: string | null;  // F00 only
  pair_svg_file?: string;            // not used by the card
}

export interface PoseComparisonMeasurement {
  id: string;                   // "M10"
  name: string;                 // "Head over the ball at contact"
  value: number | null;
  unit_text?: string;           // "% of your height", "degrees", "ms", "ratio"
  direction?: "higher_better" | "higher_worse";
  reference_range?: [number, number] | null;   // [low, high] for good drives
  status?: "outside" | "inside" | "unknown";
  position?: "below" | "within" | "above";
}

export interface PoseComparison {
  available: boolean;
  nothing_to_fix?: boolean;
  fault?: string | null;
  fault_text?: string;
  moment_label?: string;        // "Contact"
  what_to_change?: string;      // one line, e.g. "Keep your head over the ball and meet it close to your front pad."
  change_text?: string;         // the longer sentence with the player's number
  silhouette?: FaultSilhouette | null;
  measurement?: PoseComparisonMeasurement | null;
  images?: {
    overlay_label?: string;           // "On your video"
    overlay_data_url?: string | null;
  } | null;
  reason?: string;
}
```

**2. When to show the card.** Show it only when `result.pose_comparison?.silhouette?.correct_svg_data_url` is a non-empty string. In every other case render nothing: no empty card, no placeholder, no error text. Note that the card **can show when `available` is `false`**: the cartoon pictures teach the fault even when the player's own comparison could not be made; in that case simply leave out the parts whose data is missing (the bar and the video toggle).

**3. Card layout** (the app's normal card style, full width, a muted left accent border so the priority card stays the lead). Top to bottom:

1. **Heading:** `fault_text` shortened to its first clause (cut at the first comma or ", or"), e.g. "Head falls away at contact". Under it a small muted **moment chip** with `moment_label` (e.g. "Contact") when present.
2. **The cartoon pair**, two pictures side by side in a 2-column grid with a 12 px gap:
   - Left column: a row with a round **cross badge** (white ✕ on `#D55E00`, 22 px) and the word `label_fault` ("Not this") in bold `#D55E00`; under it the picture `fault_svg_data_url`; under that the caption `caption_fault` in bold, centred, 15 px, at most two lines.
   - Right column: the same with a round **tick badge** (white ✓ on `#0072B2`), `label_correct` ("This") in bold `#0072B2`, `correct_svg_data_url` and `caption_correct`.
   - Pictures: `<img>` at the column's full width, `height: auto`, no cropping. Give the pair one accessible description: `aria-label` / `alt` from `silhouette.alt` on the first image, and `alt=""` on the second.
   - **Left-handers:** if `silhouette.mirror === true`, apply `transform: scaleX(-1)` to both `<img>` elements only (never to the badges or the words). The pictures have no text in them, so this is safe.
   - **F00 (nothing to fix):** `fault_svg_data_url` is null. Show two "This" columns instead: `correct_svg_data_url` with `caption_correct`, and `correct_finish_svg_data_url` with `caption_correct_finish`, both with the tick badge.
   - **View tag:** if `view` is `"side_on"` or `"pitch_side"`, show under the pair, in small muted text with a small camera icon: "Seen from the side". If `view` is `"top_down"`: "Seen from above". Nothing for `"front_on"` (that matches the player's own video).
3. **What to change:** `what_to_change` in bold body text (17 px). Show it exactly as sent.
4. **Your number** (only when `measurement` is present, `measurement.value` is a number and `reference_range` has two numbers): a small heading "Your drive", then:
   - One line: `{measurement.name}: {formatted value} {unit_text}`. No decimals for "% of your height", degrees and ms; one decimal for "ratio". Show "%" for "% of your height" values (e.g. "9%").
   - The **range bar**: a thin horizontal track (8 px high, rounded). The scale runs from `min(low, value)` to `max(high, value)`, padded by 15% of that span on each side. The good range from `low` to `high` is a shaded band labelled "Good drives" under it. The player's value is a 14 px **diamond** marker labelled "You": `#D55E00` with a tiny ✕ when `status === "outside"`, `#0072B2` with a tiny ✓ otherwise (shape and words carry the meaning, not just colour).
   - Under the bar, in small muted text, the longer sentence `change_text`.
   - Hide this whole block when the measurement is missing or its value is null.
5. **See it on your video** (only when `images.overlay_data_url` is a non-empty string): a collapsed toggle row "See it on your video" with a chevron, **closed by default**, at least 44 px high. When opened it shows:
   - a small label `images.overlay_label` ("On your video") and the `moment_label`;
   - the image `overlay_data_url`, full width, `height: auto`, rounded corners, `object-fit: contain`. Alt text: "Your own frame at {moment_label}, face blurred, with one marker on the part that matters".
   - The player's own frame is never mirrored.
   Remember the open/closed state in component state only; do not store it.
6. **Drill link:** a text link "See the drill plan". It scrolls smoothly to the element with id `drill-plan` and opens it if collapsed. If no element with that id exists on the page, hide the link.
7. **Small print** at the bottom, muted: "Pictures show the idea, not your exact body."

**4. "Nothing major to fix".** If `nothing_to_fix === true`, keep the same card with the heading "Keep it grooved", the two "This" pictures (section 3, F00), `what_to_change`, the video toggle (when the image is present) and the drill link. No bar.

**5. Mobile first.** Design for a 360 px wide phone with 16 px side margins: the two pictures each take half the width (about 156 px), captions wrap to at most two lines without cutting words, the bar and its labels fit without horizontal scrolling, every tap target is at least 44 px high. On wider screens cap the pair at 480 px wide and centre it, and cap the video picture at 480 px.

**6. Colours, privacy and tone.**
- Use exactly `#D55E00` for "Not this" and `#0072B2` for "This"; never use colour alone (keep the badges and the words).
- The pictures carry their own light background, so they work in light and dark mode; do not add a filter or invert them in dark mode.
- The video picture is the player's own frame with the face already blurred by the backend. Do not add a download or share button to this card. Do not put the video picture in the share card.
- Never call the "This" picture "perfect" or name a player. Never say the advice prevents injury.

**Example to test with:** a result whose `pose_comparison` is

```json
{
  "available": true, "nothing_to_fix": false, "fault": "F10",
  "fault_text": "Head falls away at contact, or the ball is met away from the pad",
  "moment_label": "Contact",
  "what_to_change": "Keep your head over the ball and meet it close to your front pad.",
  "change_text": "At contact your head is 9% of your height to the off side of your hips; good drives are between 16% and 31% of your height to the off side of your hips. Keep your head over the ball and meet it close to your front pad.",
  "silhouette": {
    "fault": "F10", "view": "front_on", "label_fault": "Not this", "label_correct": "This",
    "caption_fault": "Head falls away", "caption_correct": "Head over the ball", "mirror": false,
    "alt": "Two cartoon batters at contact, seen from the bowler's end. Not this: the head falls away and the ball is met away from the pad. This: the head is over the ball, next to the front pad.",
    "fault_svg_data_url": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMDAgNDIwIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjQyMCIgcng9IjE4IiBmaWxsPSIjRjVGNEYwIiBzdHJva2U9IiNENTVFMDAiIHN0cm9rZS13aWR0aD0iNiIvPjwvc3ZnPg==",
    "correct_svg_data_url": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMDAgNDIwIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjQyMCIgcng9IjE4IiBmaWxsPSIjRjVGNEYwIiBzdHJva2U9IiMwMDcyQjIiIHN0cm9rZS13aWR0aD0iNiIvPjwvc3ZnPg=="
  },
  "measurement": {"id": "M10", "name": "Head over the ball at contact", "value": 9.42, "unit_text": "% of your height",
                  "direction": "higher_better", "reference_range": [16.2, 31.0], "status": "outside", "position": "below"},
  "images": {"overlay_label": "On your video",
             "overlay_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}
}
```

(The two test pictures are plain framed panels, vermilion and blue; real results carry the cartoon batters.) This must show:
- the card under the priority fix, headed "Head falls away at contact" with a "Contact" chip;
- "Not this" with a cross on the left and "This" with a tick on the right, each with its picture and caption; no "Seen from the side" tag;
- "Keep your head over the ball and meet it close to your front pad." in bold;
- "Head over the ball at contact: 9% of your height", a vermilion diamond left of the "Good drives" band, and the longer sentence under it;
- a closed "See it on your video" toggle that opens to the picture;
- the drill link (when the drill plan section is on the page) and the small print.

Then test three more cases:
- set `"mirror": true`: both pictures flip, the badges and words do not;
- set `"available": false`, remove `measurement` and `images`, and set `"view": "side_on"`: the card still shows the pair, the "Seen from the side" tag and the line on what to change, with no bar and no video toggle;
- remove `pose_comparison`, or set `"silhouette": null`: the card does not appear at all.

Keep the current fonts, colours and layout everywhere else.
