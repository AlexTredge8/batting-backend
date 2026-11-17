# Drill plan rules v0.1 (draft)

DRAFT, 17 Nov 2026. This is the full logic and copy for the "Your drill plan" section of the results page. The app prompt that builds it is `docs/lovable/drill_plan_prompt.md`; if you change a rule or a line of copy here, change it there too. The drills are in `drill_library.json`. Coaching video links are in `drill_videos.json` (read only here). Alex (a qualified coach) must sign off every drill before the plan goes live (`drill_signoff.md`).

The plan is three weeks long, three sessions a week, 15 to 20 minutes a session. It is built from one analysis and is saved with that analysis.

---

## 1. Inputs

| Input | Where it comes from, in order (stop at the first that gives an answer) | Values |
|---|---|---|
| **Priority fault (P)** | `getPriorityFix(result)` from `src/lib/priorityFix.ts` (built by `results_tips_and_sharing_prompt.md` Pass 1). F14 becomes F07. | `state` = `fault`, `clean`, `unmapped` or `none`; `id` = F00 to F19 |
| **Second fault (S)** | The first item in `result.tips` (in the order the backend sends it) whose `id` is F01 to F19 (F14 becomes F07), is not F00 and is not the same as P. If `result.tips` is missing or has no such item, S = none. Never work S out from the pillars: this is the same rule as the "Also spotted" row, because the four live rules are too few to name a second fault safely. | F01 to F19, or none |
| **Player level (L1)** | 1. The level the player picked in this plan (saved in the plan state). 2. A profile field `batting_level`, `playing_level` or `level` whose value is `beginner`, `club` or `advanced` (any letter case). 3. `result.score_band`: Fundamentals or Work Needed = `beginner`; Developing, Good or Excellent = `club`. 4. `result.battingiq_score`: under 55 = `beginner`, 55 or more = `club`. 5. `beginner`. | `beginner`, `club`, `advanced` |
| **Age band** | 1. `getProfileAge(profile)` from `src/lib/trainingConsent.ts`: under 18 = `under_18`, 18 or more = `adult`. 2. If the analysis links to its upload row: `uploader_under_18` true = `under_18`, false = `adult`. 3. `unknown`. | `adult`, `under_18`, `unknown` |
| **Release mode** | `FEATURES.drillPlan` in `src/config/features.ts` | `off`, `preview`, `live` |

**Why the score never gives "advanced".** The score is in beta. In the 14 November baseline, several beginner clips scored Good or Excellent. So the score can only suggest beginner or club; only the player can choose advanced. The player can change the level at any time (section 6).

**Why unknown age is treated as under 18.** It is the safe default, and it matches the share card rule.

---

## 2. When the plan shows

| `getPriorityFix` result | What shows |
|---|---|
| `null` | Nothing (the whole section is hidden). |
| `state = "none"` | The "no plan" box: copy `noPlanNone`, with the "How to film it right" link. |
| `state = "unmapped"` | The "no plan" box: copy `noPlanUnmapped`, with the "How to film it right" link. |
| `state = "fault"` | The plan for P (section 4). |
| `state = "clean"` (F00) | The "keep it grooved" plan (section 4, F00 rules). |

Release mode:
- `off`: never show the section. The priority card keeps its own "Drill to try" box.
- `preview`: show the section only to admins (a row in `user_roles` with `role = 'admin'` for the signed-in user) or when the page address has `?drills=preview`. Everyone else sees the page as in `off`. Drills that are not signed off show a "Draft" badge.
- `live`: show the section to everyone. Use only drills with `signoff = "ok"`.

If P has no usable drill at any level (for example, in `live` mode before Alex has signed off that fault), show copy `noPlanChecking` instead of the plan.

---

## 3. Picking a drill: `pickDrill(fault, level)`

1. Usable drills: in `live` mode, drills of that fault with `signoff = "ok"`; in `preview` mode, drills of that fault with any `signoff` except `"removed"`.
2. Try the levels in this order and return the first usable drill:
   - `beginner`: beginner, club, advanced
   - `club`: club, beginner, advanced
   - `advanced`: advanced, club, beginner
3. If none is usable, return none.

`nextLevel`: beginner goes to club, club goes to advanced, advanced stays advanced.

---

## 4. Building the plan

L1 = player level. L2 = `nextLevel(L1)`. Each week has a **main** drill and, from week 2, a **second** drill.

### For a fault (P = F01 to F19)

| Week | Sessions | Main drill | Second drill |
|---|---|---|---|
| 1 | 1, 2, 3 | `pickDrill(P, L1)` | none |
| 2 | 1, 2, 3 | `pickDrill(P, L2)` | if S: `pickDrill(S, L1)`; else `pickDrill(P, L1)` (a short recap of week 1) |
| 3 | 1, 2 | `pickDrill(P, L2)` | if S: `pickDrill(S, L2)`; else `pickDrill("F00", L1)` |
| 3 | 3 (check-in) | `pickDrill(P, L2)`, one set only | none; the "film and re-analyse" block takes its place |

### For F00 (nothing major to fix)

| Week | Sessions | Main drill | Second drill |
|---|---|---|---|
| 1 | 1, 2, 3 | `pickDrill("F00", L1)` | none |
| 2 | 1, 2, 3 | `pickDrill("F00", L2)` | the F00 drill one level below the main drill's level (club → beginner, advanced → club); if that is not usable, none |
| 3 | 1, 2 | `pickDrill("F00", L2)` | none |
| 3 | 3 (check-in) | `pickDrill("F00", L2)`, one set only | none; the "film and re-analyse" block takes its place |

### Clean-up rules, applied in this order

1. If S has no usable drill at any level, treat S as none and rebuild.
2. If a week's second drill is the same drill as its main drill: for a fault plan, use `pickDrill("F00", L1)`; if that is also the same, or P is F00, use none.
3. If a main drill is none (P has no drill at that level and no fallback), copy the main drill of the week before.
4. **Repeat marker.** If a week's main drill was also the main drill of an earlier week, show copy `makeItHarder` with that drill's `progression` under its name for that week. This happens in week 3, and in week 2 when L1 is advanced (L2 is then the same level). Second drills never get the marker: a recap is meant to be easy.

### Worked examples

- **F09, club (score band Good), no second fault, age unknown.** Week 1: F09-C Metronome drives. Week 2: F09-A Smooth at speed, then F09-C as a recap. Week 3: F09-A with "Make it harder" and its progression, then F00-C Groove and vary. Check-in: one set of F09-A, then film. The junior safety line shows, because the age is unknown.
- **F07 with F10 second, beginner, adult.** Week 1: F07-B Side-on to contact. Week 2: F07-C Chin to shoulder, then F10-B Eyes on the tee. Week 3: F07-C with "Make it harder", then F10-C Ball under the eyes. Check-in: one set of F07-C, then film.
- **F00, advanced (picked by the player), age 15.** Week 1: F00-A Pressure grooving. Week 2: F00-A with "Make it harder", then F00-C Groove and vary. Week 3: F00-A (still marked "Make it harder"), no second drill. Check-in: one set of F00-A, then film. The junior line shows on every card.
- **F05, advanced, no second fault, adult.** Week 1: F05-A Read the line. Week 2: F05-A with "Make it harder"; the recap would also be F05-A, so clean-up rule 2 gives F00-A Pressure grooving. Week 3: F05-A with "Make it harder", then F00-A Pressure grooving.

---

## 5. Sessions

Three sessions a week. Leave at least one day between sessions. Each session is made of blocks:

| Block | Minutes | Content |
|---|---|---|
| Warm-up | 3 | copy `warmUp` |
| Main drill | 10 (5 in the check-in) | the main drill card; instruction `mainInstruction` (or `checkInMainInstruction`) |
| Second drill | 5 | weeks 2 and 3 only; the second drill card; instruction `secondInstruction` |
| Film and re-analyse | 10 | check-in session only (week 3, session 3); copy `filmBlock` |
| Finish | 2 | copy `finish` (not in the check-in) |

Totals: week 1 sessions are 15 minutes; weeks 2 and 3 are 20 minutes; the check-in is 18 minutes. Show the total on each session row.

Session notes (one muted line under each session in weeks 1 to 3, sessions 1 to 3): `sessionNote1`, `sessionNote2`, `sessionNote3`. The check-in has `checkInNote` instead.

Session keys for the checklist: `w1s1`, `w1s2`, `w1s3`, `w2s1`, `w2s2`, `w2s3`, `w3s1`, `w3s2`, `w3s3`. There are always 9.

---

## 6. Level picker

At the top of the plan, one line: copy `levelLine` with the level name, and a **Change** link. Change opens three options (radio buttons), each with its description:

| Value | Label | Description |
|---|---|---|
| beginner | Beginner | New to cricket, or new to batting with a proper ball. |
| club | Club | I play for a school or club team. |
| advanced | Advanced | I play first-team adult cricket, or county or academy age-group cricket. |

When the level came from the score (sources 3 to 5 in section 1), show copy `levelGuessed` under the line. After the player picks a level, save it in the plan state (`levelSource = "self"`), rebuild the plan straight away, and keep the ticks already made.

---

## 7. Safety and under-18s

Every drill card shows, in this order, in small muted text:
1. The drill's own `safety` line.
2. `DRILL_SAFETY` (all players): "Warm up first. Make sure there is space around you and nobody stands within reach of your swing. Use a tennis ball or soft ball for partner drills, with the feeder standing to the side, not in line with the shot. With a hard ball, always wear a helmet with a faceguard and proper protective kit. Stop if anything hurts."
3. For age band `under_18` or `unknown` only, `DRILL_SAFETY_JUNIOR`: "Players under 18: do these drills with an adult supervising, and always wear a helmet with a faceguard when batting against a hard ball, in nets and in matches."

The plan header also shows `DRILL_SAFETY_JUNIOR` once, in a highlighted note, for `under_18` and `unknown`.

Other rules:
- **No drill needs a hard ball.** Every partner drill uses a tennis ball, and solo drills use a tennis ball or no ball.
- **Nets note**, once at the end of week 2 and week 3: `netsAdult` for adults; `netsJunior` for `under_18` and `unknown`.
- **Solo badge.** If `soloOk` is true, show the badge `soloBadge` for adults, or `soloBadgeJunior` for `under_18` and `unknown`. If false, show `partnerBadge`.
- **No personal details** anywhere in the plan, the print view or the shared text: no name, no score, no photo, no link to the analysis, no referral code. This is the same for every age, so there is nothing to switch off for juniors.
- **Claims.** Never say the drills prevent injury, fix a fault for certain, or match a coach. The footer copy `footer` says what they are. Show `checkedBadge` only on drills whose `signoff` is `"ok"`.

---

## 8. Checklist and saving

Each of the 9 sessions has a checkbox. Ticking or unticking saves at once. Show `progressLine` above the weeks, with a thin progress bar.

The current week is the first week that has an unticked session; if all 9 are ticked, it is week 3. Open the current week when the page loads.

**Plan state** (one per analysis):

```json
{ "version": 1, "analysisId": "<analysis id>", "level": "club", "levelSource": "self",
  "done": ["w1s1", "w1s2"], "createdAt": "2026-11-17T18:00:00Z", "updatedAt": "2026-11-18T19:05:00Z" }
```

- `levelSource` is `self` when the player picked the level; otherwise leave `level` and `levelSource` out, so the level is worked out again from the inputs.
- `createdAt` is set on the first save and never changed.

**Where it is saved:**
1. Always in `localStorage`, key `biq_drill_plan_v1_<analysisId>` (wrap every read and write in try/catch; if storage fails, the plan still works for this visit).
2. Also with the analysis in Supabase, if the analyses table exists and belongs to the user: a new nullable `jsonb` column `drill_plan_state` on that table, written only through the function `save_drill_plan_state(analysis_id, state)`, which updates the row only if it belongs to the signed-in user and the state is 4 KB or less.
3. On load, read both. Use the one with the later `updatedAt`. If they differ, write the winner to the other place.
4. If the server save fails, keep the local copy and show nothing (no error), but try again on the next tick.

---

## 9. Re-analyse to see progress

- The check-in session (week 3, session 3) has the "Film and re-analyse" block: copy `filmBlock`, a **"How to film it right"** link (opens the existing sheet) and a button **"Analyse a new clip"** that goes to the app's upload page.
- After all 9 sessions are ticked, or when the plan was created 21 or more days ago, show the **re-analyse card** at the top of the plan: `reanalyseTitle`, `reanalyseBody`, the same button.
- If the user has an analysis created after this one, show `newerAnalysis` at the top of the plan instead, with a link to the newest analysis's results page. That analysis has its own plan. Its "Your progress" strip already compares it with this one.
- Never say the fault is fixed or gone. Use the words in the progress strip ("didn't come up").

---

## 9b. Coaching videos for this fix

Source: `drill_videos.json` (2 to 4 public YouTube videos per fault, by other coaches). The app keeps a copy as `COACHING_VIDEOS`.

- **Which entries:** only `verified: true`. In `live` mode, also only `content_reviewed: true` (Alex has watched the video and says it fits). In `preview`, unreviewed videos show with the badge `videoNotReviewedBadge`.
- **Which faults:** P in every week; S as well in weeks 2 and 3, under a sub-heading with S's name.
- **How many:** at most 2 per fault. First the videos whose `level` is the player's level or `all`, then the others, each group in the file's order.
- **How they play:** YouTube's own player only, from `youtube-nocookie.com`, with `rel=0` and `start` when `start_seconds` is set. Nothing loads from YouTube until the player taps `videoPlayButton` (UK cookie rules; most players are juniors). Show `videoCookieNote` under the button. Never autoplay, loop or cover the controls.
- **Credit:** `videoCredit` with the channel name, linking to the original video. Never suggest the channel endorses Batting IQ.
- **Not in** the print view or the shared text.
- If no video is usable, leave the block out.

These videos are different from each drill's own demo (`video_slot`), which shows "Demo video coming soon" until a demo is filmed.

## 10. Link from the priority fix card

When the plan section shows, the drill box on the priority fix card changes to:
- **"Drill to try: {week 1 main drill name}"**, then that drill's `purpose`, then a link `seePlanLink` that scrolls to the plan section.
- The safety lines stay in the drill box (section 7 lines 2 and 3).

When the plan section is hidden (`off`, or `preview` for other users), the card keeps the beginner drill from its own table. The beginner drill of every fault in `drill_library.json` is the same as that table, so both views agree.

---

## 11. Print and share

- **"Print or save as PDF"** opens a clean plan view (all three weeks open, every drill card in full, no video areas, the safety lines, and the footer) and calls the browser's print. On phones the browser offers "Save as PDF" from the print screen.
- **"Share plan"** (only where the phone can share text) shares the plan as plain text (template `shareText`). **"Copy plan"** copies the same text and shows the toast `copiedToast`.
- The shared text has no name, score, photo, analysis link or referral code (section 7).

---

## 12. Copy (all text, exactly)

`{fault}`, `{second}`, `{level}`, `{done}` and the other braces are filled in by the app. UK spelling throughout.

| Key | Text |
|---|---|
| sectionTitle | Your drill plan |
| sectionIntroFault | Three weeks of short sessions to work on your priority fix: {fault}. |
| sectionIntroClean | Nothing major to fix, so this plan helps you keep your drive the same, ball after ball. |
| levelLine | Level: {level} |
| levelGuessed | We've guessed your level from your score. Pick the one that fits you best. |
| levelChange | Change |
| progressLine | {done} of 9 sessions done |
| week1Title | Week 1: learn the fix |
| week1Aim | Learn one change for {fault}. Go slowly and get it right. |
| week1AimClean | Groove your drive with one simple drill. |
| week2Title | Week 2: build on it |
| week2AimSecond | Make the fix harder, and start work on {second}. |
| week2AimNoSecond | Make the fix harder, and keep last week's drill going as a short recap. |
| week2AimClean | Make the drill harder, and keep last week's drill going. |
| week3Title | Week 3: put it together |
| week3Aim | Join the drills up, then film a new clip to see how it's going. |
| sessionLabel | Session {n} · {minutes} minutes |
| sessionNote1 | Go slowly. Get every step right before you add speed. |
| sessionNote2 | Normal pace. Do the check at the end of the drill card. |
| sessionNote3 | Normal pace. Do the check again: is it easier than in session 1? |
| checkInLabel | Check-in · 18 minutes |
| checkInNote | One set of your main drill, then film a new clip. |
| warmUp | Warm-up (3 minutes): jog or skip on the spot for 1 minute. Swing your arms in big circles, 10 each way. Do 5 slow shadow drives, then 5 at normal pace. |
| mainInstruction | Main drill (10 minutes): do {reps}. If you finish early, do the last set again. |
| checkInMainInstruction | Main drill (5 minutes): do one set of {repsOneSet}. |
| secondInstruction | Second drill (5 minutes): do one set of {repsOneSet}. |
| finish | Finish (2 minutes): play 5 slow shadow drives with today's cue. Hold each finish for a count of two. |
| filmBlock | Film and re-analyse (10 minutes): film 3 drives, following the five "Film it right" rules. Upload the best one. Your new result shows your progress against this one. |
| makeItHarder | Make it harder: {progression} |
| soloBadge | Can do alone |
| soloBadgeJunior | No partner needed (an adult still supervises) |
| partnerBadge | Needs a partner |
| draftBadge | Draft: waiting for a coach to check |
| checkedBadge | Checked by a qualified coach |
| youNeed | You need: {equipment} |
| setupLabel | Set-up |
| stepsLabel | Steps |
| repsLabel | Reps |
| checkLabel | Check |
| nextStepLabel | Next step |
| videoComingSoon | Demo video coming soon |
| netsAdult | If you have a net session this week, take your cue into it. With a hard ball, always wear a helmet with a faceguard and proper protective kit. |
| netsJunior | If you have a net session this week, take your cue into it, with an adult supervising. Always wear a helmet with a faceguard against a hard ball. |
| reanalyseTitle | Ready to check your progress? |
| reanalyseBody | Film a new clip of the same shot and analyse it. We'll show your new result next to this one. |
| reanalyseButton | Analyse a new clip |
| newerAnalysis | You've analysed a newer clip since this plan. Your latest plan is on that result. |
| newerAnalysisLink | See my latest result |
| seePlanLink | See your 3-week drill plan |
| noPlanNone | We need a clearer clip to build your drill plan. |
| noPlanUnmapped | We can't match this fix to our drills yet, so there is no plan for this clip. Try another clip. |
| noPlanChecking | Drills for this fix are being checked by our coach. Check back soon. |
| printButton | Print or save as PDF |
| shareButton | Share plan |
| copyButton | Copy plan |
| copiedToast | Plan copied |
| printTitle | My 3-week drill plan |
| videosTitle | Coaching videos for this fix |
| videosIntro | Videos from other coaches on YouTube that show the same fix. Batting IQ did not make them. |
| videoPlayButton | Play video (YouTube) |
| videoCookieNote | YouTube may set cookies when the video plays. |
| videoCredit | Video: {channel} on YouTube |
| videoNotReviewedBadge | Not yet watched by our coach |
| footer | Drills are suggestions based on a computer analysis of one clip. They can be wrong, and they don't know about your body or any injury. If something doesn't feel right, stop and check with a coach. |

**Share text template** (`shareText`), plain text, one item a line:

```
My 3-week Batting IQ drill plan
Working on: {fault}
Week 1: {week 1 drill names, joined with " + "}
Week 2: {week 2 drill names, joined with " + "}
Week 3: {week 3 drill names, joined with " + "}, then film a new clip
3 sessions a week, 15 to 20 minutes each. Warm up first.
{DRILL_SAFETY}
{DRILL_SAFETY_JUNIOR, only for under_18 or unknown}
{footer}
{window.location.origin}
```

For F00, the second line is "Working on: keeping my drive the same".

---

## 13. Going live

1. Alex signs off the drills (`drill_signoff.md`). Set each drill's `signoff` in `drill_library.json` and in the app's data to `ok`, or make Alex's change, or set `removed`.
2. Run the tests in the prompt with `FEATURES.drillPlan = "preview"`.
3. Alex watches the coaching videos (sign-off sheet part 2) and `content_reviewed` is set in `drill_videos.json` and in the app's copy.
4. Tell Alex. Then set `FEATURES.drillPlan = "live"`, and run prompt Pass 0b (the pricing line reads "Personalised drill plan"; add "with coaching videos" only when every fault has at least one watched video).
5. Add demo videos as they are filmed (an adult batter only; who films them is not decided yet, and Alex does not film): set `videoUrl` on the drill (the prompt has a one-line template).
