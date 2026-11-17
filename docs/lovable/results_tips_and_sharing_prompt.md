# Lovable prompt: one priority fix, progress, a share card and "Film it right"

Two passes. Paste **Pass 1** into Lovable, check it against its test examples, then paste **Pass 2**. Each pass is everything between its two lines. The note for the backend is at the very end and is not for Lovable.

Sources: the tips are the rubric v0.3 draft in `docs/phase6/fault_tips_v0_3.json` (Alex to sign off; it replaces `fault_tips_draft.json`, retires F14 into F07 and adds F17 to F19); the fault names, chain order and moments follow `docs/phase1/rubric_v0_3.json`. Each "Drill to try" is the **beginner** drill of that fault in `docs/phase6/drill_library.json` (drill id in the table), word for word, so this card and the drill plan (`docs/lovable/drill_plan_prompt.md`) always agree; Alex signs the drills off with `docs/phase6/drill_signoff.md`. If a tip or a beginner drill changes, change the same row in Pass 1's table. The rule-to-fault mapping (A5, T2, S4, F4 and the suspended rules) is ours and is the one thing Alex should check first. Legal fixes from `docs/legal/consumer_and_safeguarding_review.md` (Issues 4 and 6) and `docs/legal/ALEX_DECISIONS.md` are built in: the stronger drill safety lines, the tip caveat, and the junior share-card rules.

---

## Pass 1: the priority fix card and "Film it right"

Update the analysis results page (`src/pages/AnalysisResults.tsx`), the analysis types (`src/types/analysis.ts`) and the upload screen. Do not change the upload flow itself, the API call, the login, or any other page. Do not add any new call to the backend. Use UK spelling in all text.

**1. Types.** In `src/types/analysis.ts`, add these. Every field is optional, so older results still render.
- `type FaultId = "F00" | "F01" | "F02" | "F03" | "F04" | "F05" | "F06" | "F07" | "F08" | "F09" | "F10" | "F11" | "F12" | "F13" | "F14" | "F15" | "F16" | "F17" | "F18" | "F19"` (F14 is retired; see below)
- `type StoryboardKey = "setup" | "hands_start_up" | "front_foot_down" | "hands_peak" | "contact" | "follow_through"`
- `interface FaultTipPayload { id: FaultId; name?: string; tip?: string; frame?: StoryboardKey; drill?: { name?: string; text?: string }; source_rule_id?: string }`
- On the analysis response:
  - `priority_fault?: FaultTipPayload | null` (the backend will add this later)
  - `tips?: FaultTipPayload[]` (the backend will add this later)
  - `priority_fix?: { rule_id: string; fault: string; deduction: number; detail?: string; feedback: string } | null` (already sent today)
  - `storyboard_frames?: Partial<Record<StoryboardKey, { phase: string; label?: string; available?: boolean; image?: string | null; url?: string | null }>>` (already sent today; `image` is a `data:image/jpeg;base64,...` string)
- On `analysis_quality`, add `handedness_check?: { feet: "right" | "left" | null; used: "right" | "left"; source: string; disagrees: boolean; median_toe_dx?: number | null; samples?: number } | null`.
- Each fault in `pillars.<name>.faults` is `{ rule_id: string; fault: string; deduction: number; detail?: string; feedback: string }`.

**2. Coaching data.** Create `src/lib/coaching.ts` and hardcode these tables exactly. Do not reword them.

`FAULTS`: one entry per fault id, in this chain order (the order matters for ties and for the "Also spotted" list).

| Chain position | id | name (headline shown) | frame | tip | drill id | drill name | drill text |
|---|---|---|---|---|---|---|---|
| 1 | F01 | Head moves too early | setup | Keep your head still and your eyes level as the bowler runs in, so you get the clearest view of the ball. | F01-B | Still head, ball drop | Take your stance. Your partner stands 3 metres away, a little to your off side, and holds a tennis ball at shoulder height. Look at the ball in your partner's hand. Keep your head still and your eyes level. Your partner drops the ball when they choose. Keep your head still until the ball bounces. Play a soft forward defence, so the ball stops near your feet. 2 sets of 10 balls. |
| 2 | F02 | Unbalanced stance | setup | Stand with your weight even and your knees soft, so you can move forward or back with equal ease. | F02-B | Ready-position hops | Stand in an open space in your normal stance, with your feet about shoulder-width apart. Take your stance. Do two small hops on the spot. Land with your weight even on both feet and your knees soft. Hold still for 3 seconds. Check that you could step forward or back without a wobble. 2 sets of 10 reps. |
| 3 | F17 | Head, hands and feet not lined up as the bowler lets go | setup | As the bowler lets go, line up your head, hands and feet over off stump, so you start on the line of the ball. | F17-B | Line up on off stump | Put a stump or a cone on your off-stump line at the crease. Your partner stands 10 metres away, where a bowler lets go of the ball. Take your stance. Set your feet first, then your hands, then your head. Your partner pretends to bowl. As their arm comes over, line up your head, hands and feet over off stump. Your partner calls "yes" or "no" after each rep. 10 reps. |
| 4 | F13 | Hands late: no rhythm from the backlift | hands_start_up | Start lifting the bat a little earlier, as the bowler gets ready to let go, so the swing has time to flow. | F13-B | Lift on "now" | Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot. Take your stance. Your partner says "now" as their throwing arm starts to swing. Start to lift your hands on "now". Play a straight drive or a forward defence. Check that your bat was up before the ball bounced. 10 throws. |
| 5 | F03 | Hands go out away from the body | hands_start_up | Lift your hands up and back, close to your body, with your top hand in charge; it does not matter if the bat points towards the slips. | F03-B | Mirror backlift | Stand side-on to a mirror, about 2 metres away, so you can see yourself in your stance. Take your stance. Lift your hands slowly up and back, close to your body. Keep your top hand in charge. Check in the mirror that your hands stay near your back hip. Lower the bat and repeat. 2 sets of 10 reps. |
| 6 | F19 | Drove a ball that was too short | front_foot_down | Drive only when the ball is pitched up; if it is short, go back or leave it. | F19-B | Full or short? | Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot. They throw some full and some short, in a random order. Watch where the ball bounces. Say "full" or "short" out loud as it bounces. Drive the full balls. Go back to the short balls, or leave them. 12 throws. |
| 7 | F15 | Moving across, not out to the ball | front_foot_down | Move forward towards the ball rather than across the crease, so your head stays in line and you can hit straight. | F15-B | Stride along the line | Lay a spare bat or a stump on the ground from your front toe, pointing straight down the pitch. Take your stance. Stride forward alongside the bat on the ground, not across it. Hold the stride for 2 seconds. Do 10 strides. If you have a partner, add 10 soft drop feeds and drive each one. 2 sets of 10 strides. |
| 8 | F04 | Head and shoulder not leading | front_foot_down | Let your head and front shoulder lead you towards the ball, and your front foot will follow. | F04-B | Head first | Put a tennis ball on a cone or a tee, one long stride in front of your front foot. Take your stance and look at the ball on the cone. Move your head and front shoulder towards the ball first. Let your front foot follow. Stop in a shadow drive with your head over the ball. Do not hit the ball. Return to your stance. 10 slow reps. |
| 9 | F05 | Front foot away from the line of the ball | front_foot_down | Step towards the ball and land your front foot on its line, or just inside it, so the bat has room to come through. | F05-B | Step to the marker | Put a cone or a shoe where a full ball would bounce, just outside off stump. Take your stance. Stride so your front foot lands next to the cone, on its line or just inside it. Hold for 2 seconds and check the gap between your foot and the cone. Do 10 strides. If you have a partner, add 10 soft drop feeds that bounce near the cone. 2 sets of 10 strides. |
| 10 | F06 | Front leg stiff, weight stays back | front_foot_down | Bend your front knee as you land so that it takes your weight, instead of leaving the weight on your back foot. | F06-B | Bent-knee hold | Stand in an open space in your stance. Shadow a front-foot drive. Bend your front knee as you land, so it is over your toes. Let your back heel come off the ground. Hold the finish for 3 seconds. Return to your stance. 10 reps. |
| 11 | F18 | No push from the back foot | front_foot_down | Push off your back foot as you step forward, so your weight drives into the shot. | F18-B | Back-foot push | Stand in an open space in your stance. Shadow a front-foot drive. Push your back foot against the ground as you stride. Let your weight move forward. Finish with your back heel up and only the toe on the ground. Hold for 2 seconds. 10 reps. |
| 12 | F16 | Weight stays back | front_foot_down | Get your weight moving towards the ball, so your head and chest are over your front foot when you hit it. | F16-B | Nose over toes | Stand in an open space in your stance. Your partner stands side-on to you, 3 metres away, out of reach of your bat. Or prop a phone side-on and film. Shadow a front-foot drive. Freeze at the moment you would hit the ball. Your partner checks that your nose is over your front toes. Hold for 2 seconds. Return to your stance. 10 reps. |
| 13 | F07 | Front shoulder opens too early | contact | Keep your front shoulder pointing down the line of the shot until you have hit the ball, because an early open shoulder pulls the bat across. | F07-B | Side-on to contact | Put a tennis ball on a tee next to where your front foot lands. Face a net or open space. Your partner stands 4 metres behind you, like a wicketkeeper standing back. Or prop a phone there and film. Pick a target straight down the pitch. As you stride, point your front shoulder at the target. Drive the ball off the tee. Keep the shoulder pointing at the target until the ball has gone. Your partner calls "open" if your chest turned to the target before contact. 10 balls. |
| 14 | F08 | Bat comes across the line | contact | Bring the bat down straight along the line of the ball, and if it keeps coming across, check your stance first: head over your toes and eyes in line with off stump. | F08-B | Through the gate | Put two cones a little wider than a ball apart, one bat length in front of your front foot. Put a tennis ball on a tee between you and the gate. Check your stance: head over your toes, eyes in line with off stump. Take your stance. Drive the ball off the tee straight through the gate. Hold the finish for 2 seconds. Reset the ball and repeat. 10 balls. |
| 15 | F09 | Pause or hitch in the swing | hands_peak | Make it one smooth movement from backlift to contact; a short settle of the front foot is fine, but do not stop and start. | F09-B | One-two rhythm | Stand in an open space in your stance. Say "one" as your hands lift. Say "two" as you swing through. Leave no gap between "one" and "two". Do 10 shadow drives. If you have a partner, do 10 more with soft drop feeds. 2 sets of 10 reps. |
| 16 | F10 | Head falls away at contact | contact | Keep your head over the ball and meet it close to your front pad, so you stay in control and keep it on the ground. | F10-B | Eyes on the tee | Put a tennis ball on a tee next to your front pad. Face a net or open space. Take your stance. Drive the ball off the tee. Keep looking at the tee for one second after you hit it. Then look up to see where the ball went. Reset and repeat. 10 balls. |
| 17 | F11 | Off balance at the finish | follow_through | Hold your finish for a count of two, and if you cannot, keep your head over your front knee next time. | F11-B | Hold the pose | Put a tennis ball on a tee next to where your front foot lands, or ask a partner to drop feed. Face a net or open space. Drive the ball. Hold your finish for a count of two. Keep your head over your front knee. If you wobble, play the next ball at half pace. Reset and repeat. 10 balls. |
| 18 | F12 | Hands stop, or a wild follow-through | follow_through | Let your hands keep going towards your target after you hit the ball, and finish with the bat under control. | F12-B | Through and hold | Stand in an open space in your stance. Pick a target straight down the pitch. Shadow a drive. Let your hands keep going towards the target after the contact point. Finish with the bat under control. A short or a full finish are both fine. Hold the finish for 2 seconds. Return to your stance. 10 reps. |
| (none) | F00 | Nothing major to fix | contact | Nothing major to fix: this is a good, balanced drive, so keep repeating it. | F00-B | Keep it grooved | Put a tennis ball on a tee next to where your front foot lands, or ask a partner to drop feed. Face a net or open space. Drive 20 balls at three-quarter pace. Hold every finish for a count of two. Use the same set-up for every ball. At the end of the week, film one clip and analyse it. 20 balls. |

`F14` is retired (it is now part of F07). If the backend ever sends F14, show F07.

`DRILL_SAFETY` (shown in small muted text under every drill, for everyone): "Warm up first. Make sure there is space around you and nobody stands within reach of your swing. Use a tennis ball or soft ball for partner drills, with the feeder standing to the side, not in line with the shot. With a hard ball, always wear a helmet with a faceguard and proper protective kit. Stop if anything hurts."

`DRILL_SAFETY_JUNIOR` (shown under `DRILL_SAFETY` when the age band is `under_18` or `unknown`; see section 2b): "Players under 18: do these drills with an adult supervising, and always wear a helmet with a faceguard when batting against a hard ball, in nets and in matches."

`TIP_CAVEAT` (shown under the tip): "Tips are suggestions from a computer analysis of one clip. They can be wrong. If something doesn't feel right, check with a coach."

Never add wording that says the tips or drills prevent injury, or that the app is as good as a coach.

`RULE_TO_FAULT`: maps the backend's rule ids to fault ids. The first four are the rules that run today; the rest are switched off on the backend but are mapped in case they come back.

| rule_id | fault id | why |
|---|---|---|
| T2 | F01 | Head moves before the feet, from set-up into the backlift |
| A5 | F07 | Shoulders and hips out of line at contact: the top half has not closed, or opens early |
| S4 | F12 | The body keeps spinning after contact: an uncontrolled finish |
| F4 | F09 | Pause at the top of the backswing |
| T1 | F10 | Head outside the line at contact |
| A1 | F08 | Bat path goes around the body |
| A2 | F10 | Arms fully stretched at contact: the ball is met away from the pad |
| A3 | F08 | Short contact window: the bat is not travelling through the line |
| A4 | F10 | Torso leaning sideways at contact |
| S1 | F16 | Weight not moving to the front foot |
| S2 | F11 | Unsteady after contact |
| S3 | F11 | Hips drifting outside the base |
| F1 | F13 | Hands and front foot out of time |
| F2 | F09 | Jerky swing |
| F3 | F13 | Timing of the movement out of range |
| F5 | F09 | Hitch in the downswing |
| F6 | F12 | No natural follow-through |

`PILLAR_ORDER = ["stability", "tracking", "access", "flow"]` (the backend's own tie-break order).

**2b. Age band.** Create `src/lib/ageBand.ts` with `type AgeBand = "adult" | "under_18" | "unknown"` and `getAgeBand(profile, uploadRow?)`. Stop at the first that gives an answer:
1. `getProfileAge(profile)` from `src/lib/trainingConsent.ts` (if that file does not exist, use the same rule inline: a `date_of_birth`, `dob` or `birth_date` date, or an `age` number, on the profile, in whole years on today's date in Europe/London): under 18 gives `"under_18"`, 18 or more gives `"adult"`.
2. If the analysis row links to its upload row (an `upload_id` column or similar) and that row has `uploader_under_18`: `true` gives `"under_18"`, `false` gives `"adult"`.
3. Otherwise `"unknown"`.

Treat `"unknown"` the same as `"under_18"` everywhere (the safe default). Pass 2 and the drill plan prompt reuse this function.

`FRAME_LABELS`: setup "Setup", hands_start_up "Hands Up", front_foot_down "Front Foot Down", hands_peak "Hands Peak", contact "Contact", follow_through "Follow Through". Use the backend's `storyboard_frames[key].label` when present, else this table.

**3. Choosing the one priority fix.** Create `src/lib/priorityFix.ts` with `getPriorityFix(result)`. It returns either `null` (hide the card) or `{ state, id, name, tip, drillName, drillText, frameKey, frameLabel, source, caveat }`, where `state` is `"fault"`, `"clean"` (F00), `"unmapped"` or `"none"`. Work through these steps in order and stop at the first that gives an answer.

- **Step 0.** If `result.pillars` is missing or empty, return `null`.
- **Step A: the new backend field.** If `result.priority_fault` exists and its `id` is one of F00 to F19, use that fault (F14 becomes F07). Take `name`, `tip`, `frame`, `drill.name` and `drill.text` from the backend where they are non-empty strings, and fill any gaps from `FAULTS`. `source = "backend"`. `state = "clean"` for F00, else `"fault"`.
- **Step B: today's field.** Else, if `result.priority_fix` exists and has a `rule_id`:
  - if `RULE_TO_FAULT` has that rule id, use the mapped fault from `FAULTS`; `state = "fault"`, `source = "priority_fix"`.
  - if not, `state = "unmapped"`: name = `priority_fix.fault`, tip = `priority_fix.feedback`, no drill, `frameKey = "contact"`.
  - Note: `priority_fix` is `null` whenever the lowest-scoring pillar has no faults (for example an estimated pillar). A null `priority_fix` does **not** mean "nothing to fix"; go to step C.
- **Step C: work it out from the pillars.**
  1. Take the pillars in `PILLAR_ORDER` that exist, are not `assessed === false`, and have at least one fault.
  2. For each, add up the `deduction` of its faults (the points it lost).
  3. Pick the pillar that lost the most. On a tie, take the one that comes first in `PILLAR_ORDER`.
  4. In that pillar, pick the fault with the largest `deduction`; on a tie, the first one in the array.
  5. Map its `rule_id` with `RULE_TO_FAULT` (`state = "fault"`), or use the unmapped display from step B if the rule id is not in the table. `source = "derived"`.
- **Step D: no faults anywhere.** Count the measured pillars (those not `assessed === false`).
  - 0 measured: `state = "none"`.
  - 1 to 4 measured: F00, `state = "clean"`. If fewer than 4 were measured, set `caveat = "Some parts of this clip could not be measured, so there may be something we couldn't see."`
- **Handedness.** If `analysis_quality.handedness_check?.disagrees === true`, set `caveat` (for any state except `"none"`) to "The batting hand in this clip may not match your setting, so this fix could be wrong. See the filming tips below." This replaces the caveat from step D.

Keep this function pure (no network, no state), because Pass 2 reuses it on older analyses.

**4. The priority fix card.** Put it at the very top of the results page, above the storyboard. It replaces any priority-fix block the page already shows: never show two. Full width, the app's normal card style, an accent-coloured left border (4px).

For `state = "fault"` or `"unmapped"`:
- Small uppercase label: **YOUR PRIORITY FIX**
- Headline (largest text on the card, bold): the fault `name`.
- The `tip`, in normal body text, then `TIP_CAVEAT` in small muted text.
- A "moment" row: a 64×64 px rounded thumbnail of `storyboard_frames[frameKey].image` (object-fit: cover), then the text "In your shot: {frameLabel}" and a link **"See it in the storyboard"**. Tapping the thumbnail or the link scrolls smoothly to that storyboard frame and highlights it for 2 seconds (accent ring). Give every storyboard frame an id `storyboard-{key}` so this works; if the storyboard is a carousel or tabs, select that frame as well as scrolling to it. If the frame is missing or `available === false`, hide the thumbnail and the link and show "In your shot: {frameLabel}" only.
- A drill box (soft tinted background, rounded): **"Drill to try: {drillName}"**, then `drillText`, then `DRILL_SAFETY` in small muted text, then `DRILL_SAFETY_JUNIOR` for `under_18` and `unknown`. Hide the drill box when there is no drill (the unmapped case).
- If `caveat` is set: one line of small muted text at the bottom, with an info icon.
- Only one fix is ever shown here, even if there are several faults.

For `state = "clean"` (F00):
- Same label, **YOUR PRIORITY FIX**.
- Headline: **Nothing major to fix**, with a check icon in the success colour.
- Text: "Nice work: this looks like a good, balanced drive. The aim now is to keep it the same, ball after ball."
- The moment row uses the contact frame ("In your shot: Contact").
- Drill box: **"Drill to try: Keep it grooved"** and the F00 drill text, then `DRILL_SAFETY` (and `DRILL_SAFETY_JUNIOR` for `under_18` and `unknown`).
- The caveat line if set.

For `state = "none"`:
- Headline: **We couldn't pick out a fix from this clip**
- Text: "We couldn't measure enough of the shot to choose one thing to work on. Most of the time this is down to the filming."
- Link button: **"How to film it right"** (opens the sheet in section 6).

If `getPriorityFix` returns `null`, hide the card.

**"Also spotted" (only when the backend sends `tips`).** If `result.tips` has more than one item, add a collapsed row at the bottom of the card: **"Also spotted (fix these later)"**. When opened, list up to two other items from `tips` (skip the one with the same id as the priority fix), each as its name only, in the order sent. Fill names from `FAULTS` when missing. Do not show tips or drills for these. Never build this list from the pillars: when `tips` is missing, hide the row.

**5. Page order.** The results page now reads, top to bottom: priority fix card, storyboard, overall score (with the BETA pill and note), the "Tips to get a more accurate analysis" card, then the pillar breakdown. Keep everything else from the earlier beta prompt as it is.

**6. "Film it right".** Create one component, `FilmItRight`, used in three places. Its content, exactly:
- Title: **Film it right**
- Intro: "Five quick rules for the most accurate analysis."
- Numbered list, each with a small lucide icon:
  1. (Ruler) "Get close enough, or zoom in, so the batter fills about a third of the picture's height."
  2. (MapPin) "Stand behind the bowler's stumps and hold the phone steady at chest height. Resting it on something helps."
  3. (Camera) "Use the back camera, not the selfie camera, and keep the sound on."
  4. (Smartphone) "Landscape or portrait are both fine. Keep the phone still; don't follow the ball."
  5. (Timer) "Start before the batter's set-up and stop after the follow-through: 2 to 10 seconds is ideal."

Where it appears:
- **First-run card on the upload screen**, above the upload button, when the user has not dismissed it. Show it as a card with a **"Got it"** button. "Got it" saves `biq_film_it_right_seen = "1"` in `localStorage` and replaces the card with a small text link, **"How to film it right"**, in the same place. Wrap every `localStorage` read and write in try/catch; if storage fails, show the card (it is harmless to show it again).
- **Results page, tips card**: under the last warning, add the link **"How to film it right"**. It opens `FilmItRight` in a sheet (a bottom sheet on phones, a centred dialog on wider screens) with a **"Close"** button. The tips card is still hidden when there are no warnings.
- **Results page, priority card** in the `"none"` state (section 4), opening the same sheet.

**7. Mobile.** Design for a 375 px wide phone first. The priority card's headline wraps; nothing is cut off. The moment row stays on one line (thumbnail left, text right, the link under the text). The drill box is full width. The sheet scrolls if the content is taller than the screen. Tap targets are at least 44 px high.

**8. Test with this result** (shortened; real results have more fields):

```json
{
  "battingiq_score": 76,
  "score_band": "Good",
  "handedness": "right",
  "pillars": {
    "access":    { "score": 19, "max": 25, "status": "amber", "assessed": true, "basis": "fault_found",
                   "faults": [ { "rule_id": "A5", "fault": "Shoulders lagging behind hips at contact", "deduction": 6,
                                 "feedback": "Your shoulders are lagging behind your hips at impact." } ] },
    "tracking":  { "score": 19, "max": 25, "status": "amber", "assessed": false, "imputed": true, "raw_score": 25,
                   "basis": "not_measured", "faults": [] },
    "stability": { "score": 25, "max": 25, "status": "green", "assessed": true, "basis": "no_fault_found", "faults": [] },
    "flow":      { "score": 13, "max": 25, "status": "amber", "assessed": true, "basis": "fault_found",
                   "faults": [ { "rule_id": "F4", "fault": "Pause at the top of the backswing", "deduction": 12,
                                 "feedback": "Your hands are pausing at the top of the backswing." } ] }
  },
  "priority_fix": { "rule_id": "F4", "fault": "Pause at the top of the backswing", "deduction": 12,
                    "feedback": "Your hands are pausing at the top of the backswing." },
  "warnings": [ "The batter's feet look like a left-hander filmed from behind the bowler, but the analysis used right-handed. If the video is mirrored (front camera) or filmed from behind the batter, re-film from behind the bowler with the back camera; otherwise check the batting hand setting." ],
  "analysis_quality": {
    "scoring_status": "beta",
    "handedness_check": { "feet": "left", "used": "right", "source": "api", "disagrees": true, "median_toe_dx": 0.031, "samples": 40 }
  },
  "storyboard_frames": {
    "setup":      { "phase": "setup", "label": "Setup", "available": true, "image": "data:image/jpeg;base64,..." },
    "hands_peak": { "phase": "hands_peak", "label": "Hands Peak", "available": true, "image": "data:image/jpeg;base64,..." },
    "contact":    { "phase": "contact", "label": "Contact", "available": true, "image": "data:image/jpeg;base64,..." }
  }
}
```

This must show:
- A priority fix card at the top: **Pause or hitch in the swing**; the F09 tip; "In your shot: Hands Peak" with the hands-peak thumbnail, and "See it in the storyboard" scrolling to and highlighting that frame; "Drill to try: One-two rhythm" with its text; `DRILL_SAFETY` and, because no age is known, `DRILL_SAFETY_JUNIOR`; `TIP_CAVEAT` under the tip; and the handedness caveat line.
- With a profile aged 30: the same card without `DRILL_SAFETY_JUNIOR`.
- The tips card with the one warning and the "How to film it right" link, which opens the five rules in a sheet.
- No "Also spotted" row (there is no `tips` field).

And these variants:
- Same result with `"priority_fix": null`: step C picks flow (lost 12) over access (lost 6), so the card is the same, F09.
- `"priority_fix": null` and no fault in flow: step C picks access, A5, so the card shows **Front shoulder opens too early**, "In your shot: Contact", drill "Side-on to contact".
- All four pillars `assessed: true` with no faults, `priority_fix: null`, no handedness disagreement: **Nothing major to fix**, the positive text, "Keep it grooved", no caveat.
- All four pillars `assessed: false`: the "We couldn't pick out a fix from this clip" card with the "How to film it right" button.
- Add `"priority_fault": { "id": "F13", "tip": "Start the lift as the bowler's arm comes over." }`: the card shows **Hands late: no rhythm from the backlift** with the backend's tip (not the table's), "In your shot: Hands Up", and the table's drill, "Lift on 'now'".
- Add `"tips": [ {"id": "F13"}, {"id": "F09"}, {"id": "F11"} ]` as well: a collapsed "Also spotted (fix these later)" row listing "Pause or hitch in the swing" and "Off balance at the finish".
- On the upload screen of a new account: the Film it right card with "Got it"; after "Got it" and a reload, only the "How to film it right" link.

**Out of scope for this pass:** progress and sharing (Pass 2), any change to the backend call, editing the tips or drills in the app, and any new database table. Keep the current fonts, colours and layout everywhere else.

---

## Pass 2: "Your progress" and the share card

Pass 1 is in place: `src/lib/coaching.ts` and `getPriorityFix` in `src/lib/priorityFix.ts` exist. This pass changes the results page only. Do not add any new call to the backend (the Railway API), do not create a new database table, and do not change the upload flow. Use UK spelling.

**1. "Your progress" strip.**

Data: reuse the list of the user's past analyses that the app already loads (the same table and query the history or dashboard page uses, for example the Supabase `analyses` table, filtered to the signed-in user). Do not create a new table or a new backend call; one extra read from that existing table, with the existing query and filters, is fine. For each row you need the creation date, the overall score (`battingiq_score`), and, if the row stores it, the full result JSON. If you cannot find where past analyses are stored, or the read fails, hide the strip completely: no error message.

Logic:
- Keep only finished analyses with a numeric score, sorted by creation date, oldest first.
- Find the analysis being viewed in that list. "Earlier" means the analyses created before it (so viewing an old analysis compares it with the ones before it, not with later ones).
- No earlier analyses: show one muted line in place of the strip: "Analyse another clip of the same shot to see your progress here." Then stop.
- Otherwise show the strip, directly under the overall score:
  - Title: **Your progress**
  - A small line chart of up to the last 6 scores, ending with this one (oldest on the left). Use `recharts` if the project already has it, else a plain inline SVG polyline. No axes or grid; a dot for each analysis; this analysis's dot larger and in the accent colour; the score value printed above the first and last dots only. Height 64 px, full width.
  - Score line, with delta = this score minus the previous score:
    - delta > 0: "{score} this time: up {delta} on last time"
    - delta < 0: "{score} this time: down {abs(delta)} on last time"
    - delta = 0: "{score} this time: the same as last time"
  - Focus line: run `getPriorityFix` on the previous analysis's stored result JSON and compare with this one. Use the ids:
    - same id, not F00: "Same focus as last time: {name}. Keep going with the drill."
    - previous Fxx, now F00: "Last time's focus ({previous name}) didn't come up this time. Nice work."
    - previous Fxx, now a different Fyy: "New focus. Last time: {previous name}. This time: {name}."
    - previous F00, now Fxx: "New focus this time: {name}."
    - both F00: "Nothing major to fix, two clips running."
    - either is `null`, `"none"` or `"unmapped"`, or the previous row has no stored result JSON: leave the focus line out.
  - A muted footnote: "Scores are in beta and can move a few points between clips. Look at the trend, not one number."
- Never say a fault is "fixed" or "gone"; it only "didn't come up".

**2. The share card.**

Add a **"Share my result"** button (secondary style, share icon, full width on phones) under the progress strip. Disable it if there is no numeric score. Tapping it opens a sheet (bottom sheet on phones, dialog on wider screens) with: a live preview of the card (scaled to fit, at most 360 px wide), the face-blur control (a **"Blur face"** switch for adults; a locked note for everyone else, see below), the "check the photo" warning, and the buttons **"Share"** and **"Download image"**. Work out the age band with `getAgeBand` (Pass 1, section 2b) each time the sheet opens.

Draw the card on a `<canvas>` of 1080 × 1350 px (4:5). Do not use an HTML-to-image library. Wait for `document.fonts.ready` before drawing, and use the app's own font and colour variables (background, foreground, muted text, accent). Layout, top to bottom:
- **Header (y 0 to 110):** "BattingIQ" wordmark, bold 48 px, left at x 60; "Front-foot drive", 32 px muted, right-aligned at x 1020.
- **Photo (x 60, y 120, box 960 × 740, 32 px rounded corners, darker panel behind it):** the contact frame, `storyboard_frames.contact.image` (a data URI). Fit the whole image inside the box (contain, centred), never crop it, so the batter's head and feet stay in. If the contact image is missing, use the priority fix's frame; if that is missing too, leave the panel plain with "BattingIQ" centred in muted text, and hide the face-blur control and the "check the photo" warning. For `under_18` and `unknown`, also use the plain panel when `JUNIOR_PHOTO_ON_CARD` is `false` (a named constant, `true` for now; Alex sets it to `false` if the blur test in the backend note fails).
- **Name and score (y 890 to 1070):** a 36 px muted line "{First name}'s BattingIQ"; under it the score in bold 150 px, then "/100" at 48 px, then a **BETA** pill (outline, 28 px text) to the right.
  - **Adults only:** the first name is the player name stored on the analysis if there is one, else the profile's first name; take the first word only. If there is no name, the line reads "My BattingIQ".
  - **Under-18s and unknown age:** the line always reads "My BattingIQ". Never show a name.
  - Never show a surname, email, club or date of birth, for any age.
- **No referral code or offer** on the card. If one is added later, it goes only on adult cards, with an "Offer terms" link, never on `under_18` or `unknown` cards.
- **Priority fix (y 1100 to 1240):** a 30 px muted label "Working on" with the fault name under it, bold 46 px, at most 2 lines (ellipsis after that). For F00 the label is "This clip" and the line is "Nothing major to fix". For `"none"`, leave this block out.
- **Footer (y 1290):** the app's address, `window.location.host`, at 30 px on the left; "Beta score: a guide, not a verdict" at 26 px muted on the right. Link to the app's home page only, never to the analysis page (analyses are private).

Face blur. Keep `FACE_BLUR_FRACTION = 0.25` as a named constant. When the blur is on, blur the top 25% of the drawn image (the image's own height, not the box's). Do it in a way that works in iPhone Safari (do not rely on `ctx.filter`): copy that strip onto a small offscreen canvas at 1/24 of its size with image smoothing on, then draw it back at full size with smoothing on, and repeat once more. If the contact frame ever comes with a `face_box` (`{ x, y, w, h }`, fractions of the image), blur that box made 30% bigger instead.
- **Under-18s and unknown age: blur locked on.** No switch. In its place, a muted line with a lock icon: "Faces are always blurred for players under 18, or when we don't know your age." Under it: "Ask a parent or guardian before sharing a photo."
- **Adults:** a **"Blur face"** switch, which starts **on** every time the sheet opens (do not save the choice). Helper text under it: "Blurs the top quarter of the photo." The preview redraws straight away when the switch changes.
- **Check the photo (everyone):** above the buttons, a warning line with an alert icon: "Check nobody else can be recognised in the photo, such as other players in the nets. If someone can, don't share it."

Buttons:
- **Share** shows only when `navigator.canShare?.({ files: [file] })` is true for a PNG `File` named `battingiq-{yyyy-mm-dd}.png` (the analysis date). It calls `navigator.share({ files: [file], title: "My BattingIQ", text: "My front-foot drive scored {score} on BattingIQ (beta). {window.location.origin}" })`. If the user cancels (`AbortError`), do nothing. Any other error: a toast, "Couldn't open sharing. Download the image instead."
- **Download image** is always shown. It saves the canvas as a PNG with that file name. Under it, on iPhone and iPad only, a muted line: "If the picture opens in a new tab, press and hold it to save it."
- Close the sheet only when the user closes it, not after sharing.

**3. Mobile.** The progress strip and the share button are full width on a 375 px screen. The share sheet scrolls if needed; the preview stays at most 360 px wide and keeps the 4:5 shape.

**4. Test with the Pass 1 example result**, viewed by "Sam Taylor" (age unknown), with two earlier analyses scoring 64 and 70, where the 70 had `priority_fix.rule_id = "F4"`. This must show:
- Under the score, **Your progress**: a line through 64, 70 and 76, the 76 dot highlighted; "76 this time: up 6 on last time"; "Same focus as last time: Pause or hitch in the swing. Keep going with the drill."; the beta footnote.
- "Share my result" opens a sheet with a card reading "BattingIQ / Front-foot drive", the contact frame with its top quarter blurred, "My BattingIQ" (no name, because the age is unknown), "76/100" with BETA, "Working on: Pause or hitch in the swing", the app's address, and "Beta score: a guide, not a verdict". There is no "Blur face" switch; the locked line "Faces are always blurred for players under 18, or when we don't know your age." and the parent-or-guardian line show instead, then the "Check nobody else can be recognised" warning.
- No referral code or offer appears anywhere on the card.
- On a desktop browser without file sharing, only "Download image" shows, and it saves `battingiq-{date}.png`.

And these variants:
- The user's only analysis: no chart, just "Analyse another clip of the same shot to see your progress here."
- The previous analysis has no stored result JSON: the chart and score line show, the focus line does not.
- A profile with a date of birth 15 years ago: the same as age unknown: "My BattingIQ", blur locked on with no switch, the locked line and the parent-or-guardian line.
- A profile aged 30 named "Sam Taylor": "Sam's BattingIQ" and a "Blur face" switch that starts on; turning it off un-blurs the preview straight away; the "Check nobody else can be recognised" warning still shows. Close and reopen the sheet: the switch is on again.
- Age unknown, but the upload row has `uploader_under_18 = false`: treated as an adult (name and switch shown).
- Set `JUNIOR_PHOTO_ON_CARD = false`: under-18 and unknown-age cards show the plain "BattingIQ" panel and no blur line; adult cards are unchanged.
- A result whose contact frame is missing: the card falls back to the priority fix frame (hands peak here); with no frames at all, a plain panel, no face-blur control and no "check the photo" warning.

**Out of scope for this pass:** public share pages, links to a single analysis, saving cards to the database, new backend calls, and comparing different shots. Keep the current fonts, colours and layout everywhere else.

---

## Note for the backend (not for Lovable)

Add two fields to the analysis response; the app already prefers them over its own fallback:
- `priority_fault`: `{ "id": "F09", "name": "Pause or hitch in the swing", "tip": "...", "frame": "hands_peak", "drill": { "name": "...", "text": "..." }, "source_rule_id": "F4" }`, or `{ "id": "F00", ... }` when nothing is found in a measured clip, or `null` when nothing could be measured. Only `id` is required; the app fills the rest from its own table.
- `tips`: every fault found in this clip, in rubric v0.3 chain order (F01, F02, F17, F13, F03, F19, F15, F04, F05, F06, F18, F16, F07, F08, F09, F10, F11, F12), each in the same shape as `priority_fault`; the priority fault is included.
- Choose `priority_fault` by the rubric's rule (the fix that would help most, usually the earliest in the chain; F19 when the ball was too short to drive), not only by the lowest pillar, and never from an estimated (not measured) pillar.
- Keep the tip text in one place (`docs/phase6/fault_tips_v0_3.json`, once Alex signs it off) and the rule-to-fault mapping in config, matching the table in Pass 1 (T2 to F01, A5 to F07, S4 to F12, F4 to F09).
- **Before release, test the share-card blur** on all 18 clips, for the contact frame: check that the top-quarter blur (or `face_box`, once sent) covers the face every time. If it misses on any clip, set `JUNIOR_PHOTO_ON_CARD = false` until `face_box` is in place and passes the same test (`consumer_and_safeguarding_review.md` Issue 6c).
- Optional, for the share card: `storyboard_frames.<key>.face_box` = `{ x, y, w, h }` as fractions of the image, from the pose head points, so the blur covers the face rather than a fixed top quarter.
