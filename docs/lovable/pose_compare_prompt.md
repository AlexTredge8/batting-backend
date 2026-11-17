# Lovable prompt: "You vs best practice" card under the priority fix

Paste everything below the line into Lovable. Backend details: `docs/phase6/pose_compare.md`. The backend sends this field only when `POSE_COMPARE=on`, so with it off the card never shows and nothing changes.

---

Add a **"You vs best practice"** card to the analysis results page (`src/pages/AnalysisResults.tsx`), directly **under the priority fix card**. It shows the player's own body position at the key moment of their priority fault next to a "best practice" stick figure. Do not change the upload flow, the API call, the priority fix card or any other part of the page.

**1. Types.** In `src/types/analysis.ts`, add an optional field to the analysis response, `pose_comparison?: PoseComparison`, with these types. Every field is optional except `available`, so older results still render.

```ts
export interface PoseComparisonJoint {
  name: string;                 // e.g. "nose", "front_ankle", "hands"
  label: string;                // plain words, e.g. "head", "front ankle"
  player_xy?: [number, number];     // fractions of the frame (0-1), not used for drawing
  reference_xy?: [number, number];
  difference_pct_height?: number;   // how far apart, % of the player's height
  highlighted?: boolean;            // ringed in the picture
}

export interface PoseComparisonMeasurement {
  id: string;                   // "M10"
  name: string;                 // "Head over the ball at contact"
  value: number | null;
  unit?: string;                // raw unit, e.g. "% BL, off side +"
  unit_text?: string;           // display unit, e.g. "% of your height", "degrees", "ms", "ratio"
  direction?: "higher_better" | "higher_worse";
  reference_range?: [number, number] | null;   // [low, high] for good drives
  reference_median?: number | null;
  status?: "outside" | "inside" | "unknown";
  position?: "below" | "within" | "above";
  blank_reason?: string;
}

export interface PoseComparison {
  available: boolean;
  nothing_to_fix?: boolean;
  fault?: string | null;        // "F10"
  fault_text?: string;
  moment?: string;              // "contact"
  moment_label?: string;        // "Contact"
  measurement?: PoseComparisonMeasurement | null;
  joints?: PoseComparisonJoint[];
  change_text?: string;
  reference?: { provisional?: boolean; n_clips?: number | null };
  images?: {
    overlay_data_url?: string | null;
    side_by_side_data_url?: string | null;
  } | null;
  reason?: string;
}
```

**2. When to show the card.** Show it only when `result.pose_comparison?.available === true` **and** at least one of `images.overlay_data_url` or `images.side_by_side_data_url` is a non-empty string. In every other case, render nothing: no empty card, no placeholder, no error text. This includes `pose_comparison` missing, `available: false`, or no images.

**3. Card layout** (the app's normal card style; full width; same left accent border as the priority fix card, but in a muted colour so the priority card stays the lead). From top to bottom:

- **Title:** "You vs best practice". Next to it, in small muted text: `moment_label` (e.g. "Contact").
- **View toggle:** a two-option segmented control, **"Overlay"** and **"Side by side"**, default "Overlay".
  - If only one image is present, hide the toggle and show that image.
  - Remember the choice for the session in component state; do not store it.
- **The image:** the chosen data URL in an `<img>`.
  - Full card width, `height: auto`, rounded corners, no cropping (`object-fit: contain`).
  - Alt text: "Your position at {moment_label} with the best-practice figure laid over it" for the overlay; "Your stick figure next to the best-practice stick figure at {moment_label}" for the side by side.
  - Under the image, a one-line key in small text: an orange dot "You", a grey/teal dot "Best practice", a yellow ring "Where you differ".
- **Measurement line** (only when `measurement` is present and `measurement.value` is a number and `reference_range` has two numbers):
  - One line: `{measurement.name}: {formatted value} {unit_text}`. Format the value with no decimals for "% of your height", degrees and ms, and one decimal for "ratio". Show "%" for "% of your height" values (e.g. "9%").
  - Under it, the **range bar**: a thin horizontal track (8 px high, rounded).
    - The scale runs from `min(low, value)` to `max(high, value)`, padded by 15% of that span on each side.
    - Draw the good range from `low` to `high` as a green band.
    - Draw the player's value as a 14 px round marker: amber when `status === "outside"`, green otherwise.
    - Under the track, small labels: the low and high numbers under the band ends, and "You" under the marker. If the labels would overlap, show only "You" and the text "good range" centred under the band.
  - If `status === "outside"`, add a small amber chip "Outside the good range"; if `"inside"`, a small green chip "In the good range".
  - If the measurement is missing or its value is null, show neither the line nor the bar.
- **What to change:** `change_text` in normal body text. It is one or two plain sentences from the backend; show it exactly as sent.
- **Joints:** if any joint has `highlighted === true`, a line in small text: "Look at: {labels}", with the highlighted joints' `label`s joined by ", " (e.g. "Look at: head"). If none are highlighted, leave the line out.
- **Drill link:** a text link **"See the drill plan"**. It scrolls smoothly to the element with id `drill-plan` (the drill plan section) and opens it if it is collapsed. If no element with that id exists on the page, hide the link.
- **Provisional note:** if `reference.provisional === true`, show in small muted text at the bottom: "Early version: the best-practice figure is still being built from more clips."

**4. "Nothing major to fix".** If `nothing_to_fix === true`, keep the same card. Use the title "You vs best practice" and show the image and the toggle. Hide the measurement line, the bar and the joints line. Show `change_text` (the backend sends the "nothing major to fix" sentence) and the drill link.

**5. Mobile first.** Design for a 375 px wide phone:
- The image fills the card width.
- The toggle is full width with two equal halves, each at least 44 px high.
- The bar and its labels fit without horizontal scrolling.
- Text wraps; nothing is cut off.

On wider screens, cap the card's image at 480 px wide and centre it.

**6. Privacy and tone.**
- The overlay is the player's own video frame with the face already blurred by the backend. Do not add a download or share button to this card. Do not put this image in the share card.
- Do not add any wording that calls the reference "perfect" or names a player; it is an average of good drives.
- Never say the comparison prevents injury.

**Example to test with:** a result whose `pose_comparison` is

```json
{
  "available": true, "nothing_to_fix": false, "fault": "F10", "moment": "contact", "moment_label": "Contact",
  "measurement": {"id": "M10", "name": "Head over the ball at contact", "value": 9.42, "unit_text": "% of your height",
                  "direction": "higher_better", "reference_range": [16.2, 31.0], "status": "outside", "position": "below"},
  "joints": [{"name": "nose", "label": "head", "highlighted": true}, {"name": "front_shoulder", "label": "front shoulder", "highlighted": false}],
  "change_text": "At contact your head is 9% of your height to the off side of your hips; good drives are between 16% and 31% of your height to the off side of your hips. Keep your head over the ball and meet it close to your front pad.",
  "reference": {"provisional": true, "n_clips": 1},
  "images": {"overlay_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
             "side_by_side_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}
}
```

This must show:
- the card under the priority fix, with the Overlay / Side by side toggle;
- "Head over the ball at contact: 9% of your height" with an amber marker left of the green band, and the "Outside the good range" chip;
- the change text;
- "Look at: head";
- the drill link (when the drill plan section is on the page);
- the provisional note.

With `pose_comparison` removed, or with `"available": false`, the card must not appear at all.

Keep the current fonts, colours and layout everywhere else.
