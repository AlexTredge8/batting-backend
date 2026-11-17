# Showing a fault: benchmark and design spec (fault cards)

Written 17 Nov 2026 for Alex. Status: **draft for Alex to review**. Research, a critique of our current "You vs best practice" pictures (`pose_compare.py`, `docs/phase6/pose_compare.md`), and a design spec for a new **fault card** built on fixed, original cartoon silhouettes.

In one line: **stop drawing on the player. Show the principle with two fixed, exaggerated cartoon figures ("not this" and "this"), give one short instruction, show the player's own number on a simple bar, and keep their video as an optional extra with one small marker.**

**Top findings**
1. **GolfFix leads with one main fault, then cause, effect, fix and a drill** (sources S4, S6). Its pictures teach the principle; the user's own video, with lines, is a separate view. The exact artwork could not be seen from here (sites blocked), so the "fixed cartoon silhouettes" point rests on Alex's own use of the app.
2. **Our overlay is a coach's tool, not a player's explanation.** Two near-identical skeletons on a busy net photo cannot be read in 2 seconds; the real difference is often a few pixels, and pose-model wobbles look like faults (section 4).
3. **Draw sideways faults front-on and forward or timing faults side-on.** Ten faults read well from the bowler's end (our camera); six (F06, F16, F18, F12, F09, F13) only make sense side-on; F15 is clearest from above and F19 is a pitch diagram, not a body (section 6).
4. **Keep the numbers, change how they are shown.** The measurement, the good range and the fault-to-moment table stay; they become a simple "You vs good drives" bar and one marker on the player's own frame (section 5).
5. **Colour must never carry the meaning alone.** Use vermilion and blue (colour-blind safe) plus a cross and a tick and the words "Not this" / "This"; take all text, "DEMO" and "provisional" stamps out of the pictures (sections 4 and 5).

Contents:
1. Sources
2. How GolfFix shows a fault
3. Other apps
4. Flaws in our current pictures
5. Design spec
6. Per-fault view table (F01 to F19, and F00)
7. Open questions for Alex

---

## 1. Sources

**A note on access.** From this session, the GolfFix website (golffix.io, notice.golffix.io), the App Store, Google Play and most review sites were blocked by the network, so no page could be opened directly and **no screenshots could be viewed**. Everything below comes from web-search summaries of those pages (and the search results' own quotes). Each claim names its source. Where a point rests only on Alex's own description of GolfFix, it says so. Nothing here was copied from GolfFix's artwork; we have not seen it.

| # | Source | What it was used for |
|---|---|---|
| S1 | GolfFix, App Store listing: https://apps.apple.com/gb/app/id1586120680 (also /us/, /au/ versions) | Feature list; "45+ swing issues, each with an explanation, a recommended solution and a visual example"; update notes (skeleton overlay synced with video; named faults; One-Point Lesson) |
| S2 | GolfFix, Google Play listing: https://play.google.com/store/apps/details?id=kr.co.moais.golffix | Developer MOAIS, Inc.; about 840,000 downloads; user reviews (60 fps limit, left-handers, generous scores, recommends exercise videos) |
| S3 | GolfFix official site: https://www.golffix.io/en/golffix | Diagnosis in about 10 seconds from a phone video; Focus Drill; Shot Fix Lesson; one-to-one comparison with pros (user quote) |
| S4 | GolfFix user guide, analysis: https://notice.golffix.io/en/guide/analysis | Report contents: posture issues, their causes, the effect on the swing, solutions, Focus Drill suggestion; slow-motion playback; filming at the advised angle and distance |
| S5 | GolfFix user guide, Focus Drill: https://notice.golffix.io/en/guide/focus-drill | Posture Drill reached from the bottom of the report |
| S6 | GolfFix APEX release notes 0.14.0, 0.16.0, 0.16.1, 0.20: https://notice.golffix.io/2716f302-8c3c-804a-8c19-d70b7998db00, https://notice.golffix.io/2b06f302-8c3c-802e-a53a-f20b1c2ad876, https://notice.golffix.io/2b76f302-8c3c-8083-8831-ca8bd9b0ba8b, https://notice.golffix.io/3336f302-8c3c-80a8-bc55-e86642c3bded | "Key Problem" and "Root Cause" screens with a problem score; a screen for other issues; "Main Issue" and "Causal Issues"; Focus Drill link at the bottom of the issue content; shot troubles (hook, slice, pull, push, skying, topping, shank) |
| S7 | GolfFix APEX page: https://www.golffix.io/en/apex and https://notice.golffix.io/en/apex | "Most critical swing fault" and its root causes; two cameras, front and side |
| S8 | GolfFix iOS 1.18 and Android 2.44, 2.71.0 release notes: https://notice.golffix.io/abbeb91f-81b0-4be9-b13a-7a622787f7c7, https://notice.golffix.io/04b6c7c4-3f28-4440-8b21-eceaece91cf4, https://notice.golffix.io/1306f302-8c3c-8172-b2bd-e675289eac6a | Focus Drill: practise one problem at a time; fault names such as "Top Overswing" |
| S9 | GolfFix notice "GolfFix is Evolving!": https://notice.golffix.io/1e56f302-8c3c-8087-b20f-fb77354e3a6d | One-Point Lesson flow: choose your problem, record, get the analysis and suggestions |
| S10 | Golf Training Daily, "Best AI Golf Swing Analyzer Apps: 7 Tested": https://www.golftrainingdaily.com/blog/ai-golf-swing-analyzer-apps | GolfFix "most polished UX"; draws a silhouette outline round the body and scores positions at key points; grades generously |
| S11 | DeepSwing blog, "GolfFix vs DeepSwing": https://deepswing.io/blog/golffix-vs-deepswing-detailed-comparison-of-the-ai-golf-swing-apps?lang=en | GolfFix breaks the swing into 8 steps and shows pelvis and shoulder turn, spine tilt and body axis lines; DeepSwing's own KeyChecks, top tips, 3D ghost overlay |
| S12 | GFLO Pro review of GolfFix: https://gflopro.com/2025/11/23/golffix-app-review-ai-powered-swing-analysis-thats-transforming-my-golf-swing-teaching-by-moais/ | Visual overlays comparing the user with pros (reads as promotional) |
| S13 | Golfspan, "7 Best Golf Swing Analyzer Apps": https://www.golfspan.com/best-golf-swing-analysis-app | Free tier limits; general features |
| S14 | Sportsbox AI: https://www.sportsbox.ai/technology, https://help.sportsbox.ai/tour-ranges-for, https://www.globalgolfpost.com/biz/sportsbox-ai-the-swing-from-all-angles/ | 3D avatar from one phone video, viewed from six angles; "trackers" compared with tour ranges from 100+ tour players; custom goal ranges; avatar comparison |
| S15 | OnForm, App Store: https://apps.apple.com/app/id1490334045; Tech & Learning: https://www.techlearning.com/how-to/onform-how-to-use-it-for-school-athletics | Side-by-side or overlaid video; skeleton tracking wireframe; markup and voice-over by a coach |
| S16 | Hudl Technique summaries: https://sportsgadgetreview-com.pages.dev/reviews/hudl-technique/, https://rotaryswing.com/goaty-vs-hudl-technique | Slow motion, drawing tools (lines, circles, angles), side by side with a model |
| S17 | V1 Golf help, drawing: https://help.v1sports.com/drawing-in-v1-golf-with-ios | Coach drawing tools on the user's video |
| S18 | CricVision, App Store: https://apps.apple.com/us/app/-/id6744972352 | Cricket: auto-clips shots; feedback on timing, footwork, body position, shot selection |
| S19 | Gritts, App Store: https://apps.apple.com/ca/app/gritts/id6760328668 | Cricket: AI flags footwork, head position, follow-through |
| S20 | StanceBeam, App Store: https://apps.apple.com/lc/app/stancebeam/id1435907313 | Cricket: coach feedback by video markup, audio and text; assigned video drills |
| S21 | Fulltrack AI: https://www.fulltrack.ai/ and https://apps.apple.com/us/app/fulltrack-ai/id1568524044 | Cricket: session review, ball tracking |
| S22 | PitchVision, "What camera angles are best for filming cricketers?": https://cdn0.pitchvision.com/video-angles; CricMotion guide: https://cricmotion.pages.dev/learn/how-to-film-bowling-action-smartphone; Cricket Wales, "Analysing batters": https://cricketwales.org.uk/documents/analysing-batters-2041.pdf | Cricket coaching convention: side-on is the main batting view (backlift height, base width, contact point, weight); front-on (behind the bowler) shows head position, weight transfer sideways and bat path |
| S23 | Okabe-Ito colour-blind-safe palette: https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference, https://siegal.bio.nyu.edu/color-palette/ | Colour choices |
| S24 | WCAG 2.1, 1.4.1 (use of colour), 1.4.3 and 1.4.11 (contrast): https://www.w3.org/TR/WCAG21/ | Accessibility rules |
| S25 | Our own repo: `docs/phase6/pose_compare.md`, `pose_compare.py`, `docs/phase1/rubric_v0_3.json`, `docs/phase6/fault_tips_v0_3.json`, `docs/phase6/drill_library.json`, the six example pictures | What we have today |

**Could not confirm** (blocked or not found): the exact look of GolfFix's fault illustrations (cartoon or not, colours, arrows, whether they are fixed drawings or the user's own frame), the exact order of its screens, and how CricVision, Gritts, StanceBeam and Fulltrack draw a fault. No YouTube review of GolfFix turned up in search, so no video titles could be checked through the oEmbed route. Alex has the app; a five-minute screen recording from him would settle these points (see Open questions).

---

## 2. How GolfFix shows a fault

What we could confirm, then what rests on Alex's description.

**Confirmed from sources**
- **Fast, phone-only diagnosis.** One phone video, recorded in the app or imported, analysed in about 10 seconds (S3). It asks for a set camera angle and distance, and warns when the golfer is too tilted to analyse (S4, S8).
- **One main problem first, then its causes.** The newer product (APEX) has a "Key Problem" screen and a "Root Cause" screen, each with a problem score (S6, 0.14.0); later called "Main Issue" and "Causal Issues" (S6, 0.20), plus a separate screen for the other issues found (S6, 0.16.0). The APEX page sells it as finding "your most critical swing fault" and its root causes (S7).
- **Short fault names: body part + phase + what goes wrong.** Examples from the update notes: "Downswing Over the Top", "Lateral Left Arm Pull", "Top Overswing", "Impact Shoulder Shrug", "Impact Hip Slide", "Impact Weight Shift", "Lack of unity in take-back", "Insufficient rotation" (S1, S6, S8). Faults are split by **front view** and **side view** (S1), and APEX films both views at once (S7).
- **The text pattern is problem, cause, effect, fix.** The report gives the posture issue, its causes, its effect on the swing, a solution, and a Focus Drill suggestion (S4). The store listing says each of the 45+ issues comes with "an explanation, a recommended solution and a visual example" (S1).
- **Drill at the bottom of the issue.** The Focus Drill shortcut sits at the bottom of the issue content (S6, 0.20; S5). A Focus Drill practises one problem at a time, with the camera checking it (S8). Users say it recommends exercise videos for the problem (S2).
- **Shot-trouble route.** The One-Point Lesson starts from the ball flight the golfer suffers (slice, hook, pull, push, skying, topping, shank) and works back to the swing issues behind it (S6, S9).
- **Their own video is still there, with lines.** A skeleton overlay is synced with the video in the report (S1 update note). A reviewer describes an outline drawn round the golfer and scores at key positions (S10). Another describes 8 steps from address to finish with pelvis and shoulder turn, spine tilt and body axis lines (S11). Slow-motion playback is offered (S4).
- **Comparison with a pro** side by side is the feature users praise most (S3, S10, S12).
- **Criticisms worth learning from:** the same problem comes up every time with no sense of priority (S2 review); scores are generous to poor shots (S10); it favours a "modern" swing (review in S1 search results); left-handed support was lacking in drills (S2).

**From Alex's description (not confirmed here)**
- The fault itself is illustrated with **fixed, exaggerated, cartoon-like silhouettes** showing the principle (wrong against right), **not** drawn on the user's own body. This fits "a visual example" per issue (S1) but we could not see the artwork, its colours, arrows or layout.

**What we take from it (principles, not the look)**
1. One main fault, named in a few plain words, then the cause, the effect and the fix.
2. Teach the principle with a clear, generic picture; the user's own video is a separate, secondary view.
3. A drill, with a video, at the bottom of the fault, so the flow ends in action.
4. A score or number per fault so progress can be tracked.
5. Faults are tied to the camera view that shows them.

What we must not do: copy their character, colours, names, icons, layout or screen order literally. Our design below is cricket-specific and drawn from our own rubric.

---

## 3. Other apps

| App | How it shows a fault | Lesson for us |
|---|---|---|
| **Sportsbox AI** (golf) (S14) | Turns one phone video into a 3D avatar that can be turned to six angles. "Trackers" (chest turn, pelvis turn, sway and so on) are compared with tour ranges from 100+ tour players; coaches can set goal ranges; two avatars can be overlaid. | **A number against a good range is a strong, honest pattern.** Keep our measurement bar. Their avatar is a 3D model of the user; we cannot do 3D from one front-on view, which is another reason to use fixed drawings for the principle. |
| **OnForm** (S15) | Coach tool: two videos side by side or overlaid, a skeleton wireframe, markup and voice-over. | Overlay on the athlete's own video is a **coach's tool** for someone who already knows what to look for, not an explanation for a player. That is what our current picture is. |
| **Hudl Technique** (S16), **V1 Golf** (S17) | Slow motion, lines, circles and angles drawn by a coach; side by side with a model swing. | Same: good for coaches, cluttered for players. A single line (for example a vertical line through off stump) is the useful part. |
| **DeepSwing** (golf) (S11) | "KeyChecks" and "top tips", a 3D ghost overlay and swing plane; drills matched to the top faults. | Top tip plus a matched drill again. The ghost overlay is the approach we are moving away from. |
| **CricVision, Gritts** (cricket) (S18, S19) | AI feedback on footwork, head position, body position, follow-through, shot selection. How it is drawn could not be confirmed. | Cricket apps name the same areas we do; none we found sells a clear "principle picture". A gap we can fill. |
| **StanceBeam, Fulltrack AI** (cricket) (S20, S21) | StanceBeam: bat sensor data, plus coach video markup and assigned drills. Fulltrack: ball tracking and session review. | Not fault pictures. StanceBeam's "assign a drill" step matches our drill library. |

**Cricket camera convention (S22).** Coaching material treats **side-on** (from point or square leg) as the main batting view: backlift height, base width, contact point and weight going forward. **Front-on** (behind the bowler, our camera) shows head position, sideways weight transfer and bat path through the line. This matches our rubric's "visible" notes (`rubric_v0_3.json`), and decides the view per fault in section 6.

---

## 4. Flaws in our current pictures

From the six examples (`demo_F07_overlay`, `demo_F10_overlay`, `demo_F05_overlay`, `demo_F05_side_by_side`, `real_F17_overlay`, `live_F07_overlay`, plus `real_F10_overlay` and `live_F07_side_by_side` for comparison).

**A. Can a player tell in 2 seconds what to change? No.**
1. **"Spot the difference".** On real clips the two skeletons sit almost on top of each other (`real_F17`, `real_F10`, `live_F07`). The difference that matters is a few pixels, so the eye cannot find it. On `real_F10` the ghost and the player are identical (the provisional reference is built from that same clip), yet the card still names a fault.
2. **No instruction on the picture.** The only "what to change" is a long sentence elsewhere with percentages ("9% of your height to the off side of your hips; good drives are between 16% and 31%..."). A player cannot turn "% of your height" into a movement.
3. **Pose-estimation errors look like faults.** Drawing the exact skeleton shows its wobbles: the zigzag arms in `demo_F07`, the arm crossing the body in `live_F07`, the hands drawn as a single wrist point. The player sees "mistakes" that are the model's, which costs trust.
4. **F00 shows a ghost with nothing marked.** It says nothing.

**B. Clutter and "which is which"**
5. **Two full skeletons on top of a busy photo** (nets, curtains, pads, gloves, stumps, a red ball): too many lines, all the same thickness.
6. **Which figure is "you" relies on a small colour legend** at the top (12 px text). There is no label on the figures themselves.
7. **Colours are not consistent across versions:** the ghost is cyan in the overlays, dark teal in the side-by-side panel, grey/white in `live_F07`. The legend swatch in the side-by-side does not match the overlay.
8. **Rings change colour:** yellow on the overlay, red on the side-by-side.
9. **Front and back limbs look the same.** From front-on the legs and arms overlap; nothing tells front leg from back leg.

**C. Arrows and markers**
10. **The arrows' meaning is unclear.** An arrow from the player's joint to the ghost's joint means "your joint should be here", but it reads as "your joint moved this way". In `demo_F07` two arrows cross each other (head to shoulder, shoulder to head) and the message is lost.
11. **Unexplained marks:** a small green dot at the arrow tip, yellow rings, red rings. No key explains them.
12. **Yellow on cream nets and white shirts has poor contrast** (rings and arrows in `demo_F10`, `demo_F07`). The cyan ghost needed a dark outline to show on white nets at all.

**D. The face blur, cropping and view**
13. **The blur hides the very body part many faults are about.** F01, F04, F10 and F17 are head faults, and a big pale oval covers the head. The oval also reads as a halo or a second ghost.
14. **Blur placement is fragile.** It is placed from the pose; if the head points are wrong the oval can sit off the face (it sits high and left of the head in `demo_F07`; check `real_F10`, where the dark pixel block and the head do not quite line up). One bad frame can show a face.
15. **Cropping does not zoom on the fault.** Every picture is the whole body in a tall 480 x 674 frame, mostly net. For a foot fault (F05) the marker sits at the bottom edge next to the "Reference: provisional" text; for a head fault the head is small.
16. **The view hides depth.** Our camera is front-on; faults about weight going forward, front knee bend, back-foot push and follow-through (F06, F16, F18, F12) cannot be seen in it, and the overlay cannot show them either.

**E. The bat and ball**
17. **The bat is not drawn.** It only appears if it happens to be in the photo, and is never part of the "best practice" figure. F03 and F08 are about the bat and hands, and the bat is the most recognisable cricket cue.
18. **The ball is never shown.** F05 ("line of the ball") and F10 ("ball met away from the pad") cannot be understood without it.

**F. Stamps and wording**
19. **"DEMO: ... moved by hand"** in red sits over the picture and, in the side-by-side, over the "Best practice" label. It is a test note and must never reach a player.
20. **"Reference: provisional"** in tiny type on every picture. A player reads it as "this might be wrong". It is a data-quality flag and belongs in the JSON and the coach view, not in the picture.
21. **Text is baked into the JPEG**: it cannot be read by a screen reader, translated, resized or fixed without re-running the analysis.

**G. Colour blindness and accessibility**
22. **Meaning is carried by colour alone** (orange = you, cyan = best practice, ring = differs). WCAG 1.4.1 (S24) asks for a second cue (shape, label, pattern).
23. **Red ring on an orange figure** (side-by-side) is close to invisible for red-green colour blindness; yellow ring next to a green dot is also weak.
24. **No alt text** and no text equivalent for the picture.

**H. Privacy**
25. **The "best practice" ghost is one real batter today** (one clip, `anonymised: false`). Showing a real person's body shape as "best practice" is a privacy and an honesty issue until the average is rebuilt.
26. **The player's frame carries more than the face**: the club's nets, other people in the background, kit logos and body shape. It is fine for the player alone, but the picture will travel if they share their results, and the base64 JPEGs sit inside the report.
27. Fixed silhouettes remove the need to show any real person in the "principle" picture at all.

**I. Phone and product**
28. **The moment label** ("Just before contact") is small grey text top right; there is no sense of where in the shot this is.
29. **Left-handers**: arrows and "off side" wording flip; the picture is drawn as filmed, so a left-hander's arrow points the other way from a right-hander's for the same fault. Fine when drawn on their frame, but confusing when compared with any help text.
30. **The side-by-side stick figures are thin (about 4 px), have no head direction and no bat**, and the "You" figure is anonymous, so the player cannot link it to themselves.

---

## 5. Design spec

### 5.1 The idea

For each fault there is **one fixed, original cartoon picture pair**, drawn once and reused for every player:
- **"Not this"**: the fault, exaggerated so it is obvious at a glance.
- **"This"**: the same figure, same pose, with only the one thing changed.

The player's **own number** goes on a simple bar against the good range (kept from today's build). Their **own video frame** becomes an optional extra with one small marker, not the main picture.

### 5.2 What we keep and what we drop

**Keep (from today's build)**
- The fault choice (priority fix, stopgap rule-to-fault mapping, F00 when nothing is found).
- The **fault-to-moment table** in `pose_compare.md` (moment, main joint, alignment point, measurement). The moment names the card's moment chip; the main joint places the single marker in "See it on your video".
- The **measurement and its good range**: Phase 5 measurement (v8 code), 10th to 90th percentile of good drives, "only the worse side counts", `status` and `position`. This drives the bar.
- Normalisation (mirror left-handers, camera roll, body length) for the measurement and the marker.
- The privacy rules: face blurred before anything is drawn; picture shown only to the player; nothing stored beyond the report.
- The `POSE_COMPARE` switch (default off), the "not available" reasons, and the rule that it never changes a score.
- The tip text (`fault_tips_v0_3.json`) as the source for "what to change", shortened.
- The drill library (`drill_library.json`) for the drill at the bottom.

**Drop**
- The full **ghost skeleton laid over the player's frame**, and all rings and arrows on it.
- The **side-by-side stick-figure panel**.
- The orange / cyan / teal / grey / yellow / red colour mix.
- All text baked into pictures: "DEMO", "Reference: provisional", the legend, the moment label. These become app text (or, for DEMO and provisional, internal flags only).
- The long "x% of your height" sentence as the main message (the numbers move to the bar).
- The per-player reference figure as something shown to players. `pose_reference_v1.json` is still needed for the **ranges**; its joint coordinates are no longer drawn.

### 5.3 The card on a phone (top to bottom)

Designed for a 360 dp wide screen, single column, 16 dp side margins.

1. **Fault name** (heading, at most 6 words) and a small **moment chip** under it, for example `Just before contact`. A tiny 5-step shot strip (set-up, backlift, stride, contact, finish) with the moment highlighted is optional.
2. **The silhouette pair**, side by side, each panel about 156 x 220 dp:
   - left: label **"Not this"** with a cross badge;
   - right: label **"This"** with a tick badge.
   - Same figure, same scale, same ground line, same camera view. One cue per panel (section 5.5).
   - Below the pair, a small view tag when the pair is not drawn front-on: `Seen from the side` (with a little camera icon), so the player knows it is not the same view as their video.
3. **What to change**: one line, imperative, at most 12 words, in bold. Example: "Keep your front shoulder pointing down the pitch until you hit it."
4. **Why** (one line, normal weight, optional): the effect. Example: "An open shoulder drags the bat across the ball."
5. **Your drive**: a horizontal bar (section 5.7). Hidden for pose-only faults and when the value is unknown.
6. **See it on your video** (collapsed by default, tap to open): the player's own frame at the key moment, face blurred, cropped to the fault area, with **one** marker on the main joint and, where it helps, the same cue line as the silhouette (for example the vertical off-stump line). No skeleton, no ghost, no arrows.
7. **Drill**: name, one line, level (beginner, club, advanced), and a "Watch" link to the drill video. This is the last thing on the card, so the flow ends with something to do.

Small print under the card (app text, not in the picture): "Pictures show the idea, not your exact body." When the reference ranges are provisional, the coach view and logs say so; the player card does not.

### 5.4 Figure style rules

- **Original figure.** A simple, friendly, gender-neutral cricketer in solid shapes: rounded capsule limbs, an oval body, a round helmeted head with a visor line to show which way it faces (no face features, no skin tone, no hair). Pads on both legs, gloves, a plain bat, a plain ball. No logos, no club colours, no name, no mascot personality. It must not resemble any existing app's character.
- **Flat, filled silhouettes**, not stick figures. No gradients, no shading except one tone to separate near limbs from far limbs.
- **Near and far limbs**: near limbs in the main body tone, far limbs one step lighter, so the front leg and back leg can be told apart at a glance.
- **The bat is always drawn** (blade as a rounded rectangle, handle as a thin bar). The **ball** is drawn when the fault is about the line or contact point (F05, F08, F10, F15, F19) or as the timing cue (F13, F09).
- **Ground and orientation**: a ground line and, in front-on pictures, three stumps behind the batter (with off stump picked out) so the player can orient themselves. In side-on pictures, a short crease line and the bowler's direction shown by an arrowhead at the edge ("to bowler").
- **Right-handed by default; mirrored for left-handers**, including the cue lines, so the picture matches the player's own video.
- **One figure per panel**, centred on the same ground point in both panels so the eye compares like with like.

### 5.5 One key cue per picture

Each pair teaches one thing. Exactly one of these cues per panel:
- **Reference line**: a dashed line that carries the principle, for example a vertical line through off stump (line faults), a vertical plumb line from the head (balance and weight faults), the shoulder line (F07), the ball's line on the ground (F05).
- **Highlight**: the body part that matters is filled in the panel's accent colour; the rest of the figure stays neutral.
- **Arrow (correct panel only)**: one thick arrow showing the movement to make. Arrows never appear on the "Not this" panel, so an arrow always means "do this". Where a wrong movement must be shown (F15, F08), use a dotted trail on the "Not this" panel, never an arrow.
- **Timing cues** (F13, F09) use a two-frame mini strip inside the panel, or a stopwatch-style pause mark, rather than an arrow.

No angle numbers, no degree arcs, no percentages on the pictures.

### 5.6 Exaggeration level

- Make the fault **about twice** as large as a typical real fault, so it reads at thumbnail size (for example a head clearly outside the off-stump line by a full head width; shoulders opened to about 45 degrees; the front foot a full foot-length off the ball's line).
- Stay physically possible: no bent-back joints, no floating feet. A coach should look at it and say "yes, that's the fault, a bit overdone".
- The "This" figure is a clean, textbook version, not an average of real players.

### 5.7 Colours, weights and type (colour-blind safe)

Okabe-Ito based (S23), with shape and words always carrying the meaning too (S24):

| Role | Colour | Notes |
|---|---|---|
| Card background | `#F5F4F0` (warm off-white) | Not pure white, so white pads still show against it |
| Figure body | `#3D4450` (slate) | Far limbs `#6B7380` |
| Pads, gloves | `#FFFFFF` with a 2 dp slate outline | |
| Bat | `#C8A165` (willow) with slate outline | |
| Ball | `#7A1F1F` (dark cherry) with a white seam | A real ball colour, dark enough to read on the light ground, and distinct from the vermilion accent |
| **"Not this" accent** | `#D55E00` (vermilion) | Highlighted body part, cross badge, panel label underline |
| **"This" accent** | `#0072B2` (blue) | Highlighted body part, tick badge, the arrow, panel label underline |
| Cue lines | `#1A1A1A` dashed (6 on, 4 off) | Same in both panels |
| Ground line, stumps | `#9AA0A6` | |

- Vermilion against blue is safe for the common red-green colour blindness and for blue-yellow; the **cross and tick badges and the words "Not this" / "This"** mean the pair still works in greyscale. Test every pair in a greyscale and deuteranopia simulation before sign-off.
- Line weights at 1x (panel about 156 dp wide): figure outline 2 dp; cue line 2 dp dashed; arrow shaft 5 dp with a head at least 14 dp wide; badges 24 dp.
- Text is app text, not in the image: heading 20 sp, instruction 17 sp bold, body 15 sp, minimum 13 sp. Contrast at least 4.5:1 for text and 3:1 for graphics (S24).
- Pictures are delivered as **SVG** (sharp at any size, small, recolourable); PNG at 3x as a fallback.

### 5.8 The bar ("Your drive")

- A single horizontal track. The **good range** is a shaded band labelled "Good drives". The player's value is a **diamond marker** labelled "You". Each end of the track carries a word, not a number, from the fault's direction (for example "Head to the leg side" ... "Head to the off side").
- When the player is outside the range, the marker is vermilion with a cross; inside, blue with a tick. Shape and words again carry the meaning.
- Numbers (value, range, unit) go in a small line beneath for those who want them, using today's `unit_text`.
- Use today's fields unchanged: `measurement.value`, `reference_range`, `status`, `position`, `direction`.
- Hidden for F15, F18, F19, F00 and when `status` is `unknown`.

### 5.9 "See it on your video" (optional)

- Off by default; the player taps to open.
- The frame at the fault's moment, face blurred first (today's method; widen the oval and fall back to blurring the whole top fifth of the person box when head points are missing).
- Crop to the fault area (head and shoulders for head and shoulder faults; feet and stumps for foot faults; whole body for balance), at least 2x larger than today's crop for the key area.
- **One marker** on the main joint from the moment table: a 3 dp blue ring with a white halo (to show on nets), and, if the silhouette uses a reference line, the same dashed line on the frame (for example the off-stump line). No skeleton, no ghost, no arrow, no text.
- Optional later: a 1-second loop of the clip around the moment, with the same marker.

### 5.10 Animation (optional, later)

A gentle loop that morphs "Not this" into "This" over about 1 second, then holds, can replace the static pair on a tap. Respect the phone's reduce-motion setting. Static pictures are the default and must work on their own.

### 5.11 Text pattern

| Slot | Pattern | Example (F07) |
|---|---|---|
| Fault name | [Body part] [what goes wrong] [when], at most 6 words | Front shoulder opens too early |
| Moment chip | From the moment table | Just before contact |
| What to change | [Verb] your [body part] [direction or target] [until/when], at most 12 words | Keep your front shoulder pointing down the pitch until you hit it. |
| Why | [Fault] [causes] [effect on the shot], one line | An open shoulder drags the bat across the ball. |
| Bar ends | Plain words for each end of the scale | Shoulder closed ... Shoulder open |
| Drill | Name; one line; level; "Watch" | From `drill_library.json` |
| Alt text | "Two cartoon batters. Not this: [fault]. This: [fix]." | "Two cartoon batters seen from the bowler's end. Not this: chest turned to face the bowler before contact. This: front shoulder still pointing down the pitch." |

UK spelling, plain English, no percentages in the main lines. "Off side" and "leg side" are fine (players know them). The tips in `fault_tips_v0_3.json` are the source; the card uses the first clause as "What to change" and the "so that..." clause as "Why".

### 5.12 Data and build notes for the silhouette builder

- Asset names: `fault_cards/{FAULT}_not_this.svg`, `fault_cards/{FAULT}_this.svg`, and `F00_this.svg` only. Mirroring for left-handers is done in the app (horizontal flip) unless a picture contains text (it should not).
- Add to `report.pose_comparison` (when the card version ships): `card_version`, `view` (`front_on`, `side_on`, `top_down`, `pitch_side`), `asset_ids`, `what_to_change`, `why`, `bar_end_labels`, `alt_text`, `drill_id`. Keep the existing fields so the old card can be compared during testing. The overlay and side-by-side JPEGs can be dropped from the default payload (saves 50 to 75 KB per report) and replaced by one smaller cropped frame with the single marker, only when requested.
- Alex signs off every pair before it goes live (it is coaching content, like the drills).

---

## 6. Per-fault view table (F01 to F19, and F00)

Names from `docs/phase1/rubric_v0_3.json`. "Front-on" means drawn as our camera sees it (from behind the bowler). "Side-on" means drawn as if from point, the classic coaching view. Rule of thumb: **sideways faults are drawn front-on**, so the player can find the same thing on their own video; **forward-and-back or timing faults are drawn side-on (or from above)**, because front-on cannot show them, and the card says so with the view tag.

| Fault | Rubric name | View | Why this view | Moment | "Not this" (exaggerated) | "This" | The one cue | Ball? |
|---|---|---|---|---|---|---|---|---|
| F01 | Head moves early or is not still at set-up | Front-on | Head tilt and sideways sway are what our camera sees (C01 visible: yes) | Set-up | Head tilted, eyes on a slant, with two faint earlier head positions behind it (motion trail) | Head upright, eyes level | Dashed horizontal eye-level line | No |
| F02 | Unbalanced stance | Front-on | Leaning to the off or leg side shows front-on; stance width lies along the pitch and is not judged | Set-up | Body leaning well to one side, head outside the feet, knees locked | Weight even, head between the feet, knees soft | Vertical plumb line from the head to the ground | No |
| F17 | Head, hands and feet not lined up over off stump as the bowler lets go | Front-on | It is about line, which is exactly what the bowler's end sees (C17) | Set-up (bowler lets go) | Head, hands and feet stacked outside off stump (eyes beyond it) | Head, hands and feet stacked over off stump | Vertical dashed line up from off stump; stumps drawn | No |
| F13 | Hands late or static: no rhythm from the backlift | Side-on, two-frame strip | It is timing; side-on shows backlift height clearly; front-on foreshortens a bat lifting towards the slips | Set-up to top of backlift | Bowler icon at release (small inset) while the bat is still grounded | Bowler icon at release while the bat is already rising | The small bowler-at-release inset, same in both | Yes, as the timing cue (in the bowler's hand) |
| F03 | Hands go out away from the body in the backlift | Front-on | The sideways gap between hands and body is what M03 measures from our camera | Top of backlift | Hands far out from the back hip, arms straight | Hands up and back, close to the body, top hand in charge (bat may point to slips) | Highlight the hands; short dashed line from hands to back hip | No |
| F19 | Drives a ball that is too short to drive | Pitch diagram, side-on (not a body picture) | Length lies along the pitch, invisible front-on; it is shot choice, not body position | None | Ball pitching in the short zone with a batter-icon driving | Ball pitching in the full zone with a batter-icon driving; short zone marked "go back or leave" | Two shaded zones on a side-on pitch strip: "full: drive", "short: don't" | Yes |
| F15 | Moves across the crease, not out towards the ball | **Top-down** (footprints), with a small front-on figure | Front-on shows "across" but not "out"; from above both are obvious | Front foot down | Front footprint stepping sideways across the crease, dotted trail | Front footprint stepping out towards the ball, along its line | Ball's line drawn on the pitch; crease line | Yes (line only) |
| F04 | Head and shoulder do not lead to the line | Front-on | The sideways lead of head and front shoulder to the ball's line is what our camera sees (C04: line yes) | Front foot down | Front foot stepping out while head and shoulder stay back, upright | Head and front shoulder leaning towards the ball's line, foot following | Arrow (correct panel) from head towards the line; ball's line dashed | Line only |
| F05 | Front foot well away from the line of the ball | Front-on | The foot's sideways distance from the ball's line is seen and measured front-on (M05) | Front foot down | Front foot landing well to the leg side of the ball's line, body open | Front foot on the line or just inside it | Ball's line dashed on the ground from the ball to the foot | Yes |
| F06 | Front leg stiff; weight stays back | **Side-on** | Knee bend is foreshortened front-on (C06: partly); side-on shows it plainly | Contact | Front leg straight, body leaning back, weight on back foot | Front knee bent over the front foot, body over it | Highlight the front knee | No |
| F18 | Back foot stays put: no push to drive the weight forward | **Side-on** | The back leg hides behind the front leg front-on (C18: partly) | Contact | Back heel flat, back knee bent, weight sitting back | Back heel up, back leg extended, pushing forward | Arrow (correct panel) along the back leg towards the bowler | No |
| F16 | Centre of mass stays back: does not move out over the ball | **Side-on** | The coaching point is forward ("head and chest over the front foot"), which front-on cannot see; our sideways measurement M16 still feeds the bar | Contact | Head and chest behind the back hip, a big gap to the front foot | Head and chest over the front foot | Vertical plumb line from the head to the ground | No |
| F07 | Front shoulder opens early, or the top half never closes | Front-on | From the bowler's end a closed batter shows a narrow shoulder line; opening early shows the chest (C07: yes) | 100 ms before contact | Chest square to the camera, both shoulders wide, bat path angled across | Front shoulder pointing at the camera, narrow shoulder line | Shoulder line drawn as a dashed bar; small top-down shoulder icon optional | No |
| F08 | Bat comes across the line | Front-on | Bat path across the ball's line shows best from the bowler's end | Contact | Bat swinging from off to leg across the ball, dotted swing trail | Bat coming straight down the ball's line | Ball's line dashed; bat's path as a dotted trail (not an arrow) on "Not this", one arrow along the line on "This" | Yes |
| F09 | Pause or hitch in the swing | Side-on | The swing arc from backlift to contact is clearest side-on | Top of backlift | Swing arc drawn as a dotted trail with a break and a pause mark at the top | One smooth, unbroken arc from backlift to contact | The swing arc trail | No |
| F10 | Head falls away at contact, or the ball is met away from the pad | Front-on | Sideways head position and the gap between bat and pad are what our camera sees (C10: sideways yes) | Contact | Head tipped to the off side, ball met a bat-length from the front pad | Head over the ball, ball met next to the front pad | Vertical dashed line from the head down to the ball | Yes |
| F11 | Off balance at the finish | Front-on | Sideways falls are what we judge (C11) | Finish | Body toppling to one side, back foot swinging round, head outside the base | Balanced finish held, head over the front knee | Vertical plumb line from the head | No |
| F12 | Hands stop at contact, or the follow-through is wild | **Side-on** | The follow-through goes towards the target, which is along the pitch, foreshortened front-on | Finish | Hands stopped at waist height, bat jammed; or (variant) a wild wrap-around | Hands carried through high towards the target, bat under control | Arrow (correct panel) from contact to the finish, towards the target | No |
| F00 | Nothing major to fix | Front-on | Matches their video | Contact | (none) | The clean "This" figure at contact with a tick | None; caption "Keep repeating this" | Yes |
| F14 | Retired, now part of F07 | Use F07 | | | | | | |

Notes:
- **Side-on pictures and the player's video.** For F06, F16, F18, F12, F09 and F13 the silhouette is side-on but the player's video is front-on. The view tag ("Seen from the side") says so, and "See it on your video" still marks the main joint on their front-on frame. If Alex prefers everything front-on so it always matches the video, these six would be the weakest pictures; see Open questions.
- **Bar availability** follows today's table: no bar for F15, F18 (pose only), F19 (shot choice) and F00.

---

## 7. Open questions for Alex

1. **GolfFix check.** Could you send a short screen recording of one GolfFix fault from the result screen to the drill? We could not open their site or store screenshots from here. We only need it to confirm the principles; we will not copy the look.
2. **Side-on pictures.** Are you happy for six faults (F06, F16, F18, F12, F09, F13) to be drawn side-on, with a "Seen from the side" tag, even though the player's video is front-on? Or should every picture be front-on so it matches their video?
3. **Top-down for F15.** Is a footprints-from-above picture clear to club players, or would you rather a front-on picture?
4. **Exaggeration.** About twice a typical fault: right, too much, or too little?
5. **The figure.** Helmet on, plain kit, no face, right-handed by default and mirrored for left-handers. Any objection? Should juniors see a smaller figure?
6. **Wording.** "Not this" / "This", or "Avoid" / "Aim for", or something else you use with players?
7. **Their own video.** Collapsed by default, with one marker, as specced; or open by default?
8. **Keep the ghost for coaches?** Should the old overlay (ghost on the player's frame) stay available in the coach view only, or be removed entirely?
9. **Priority.** Which 5 faults should be drawn first? We suggest the four live rules' faults (F01, F07, F12, F09) plus F10.
10. **Sign-off.** Each picture pair is coaching content. Will you sign them off the same way as the drills (`docs/phase6/drill_signoff.md`)?
