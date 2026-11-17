# Drill videos per fault (draft, 17 Nov 2026)

Data file: `docs/phase6/drill_videos.json`. It lists, for each fault in `faults_in_chain_order` of `docs/phase1/rubric_v0_3.json` plus F00 ("keep it grooved"), two to four public YouTube coaching videos that explain the fix or show a drill for the front-foot drive. F14 has been merged into F07, so it has no entry of its own.

## Status: links checked 17 Nov 2026; content still to watch

- **Links: all checked.** On 17 Nov 2026 YouTube was allowed in this environment, and every one of the 37 videos was looked up through YouTube's oEmbed service (a public look-up of a video's title and channel; no video is downloaded). All 37 came back: each video exists, is public, and allows embedding (oEmbed refuses videos whose owner has turned embedding off). Titles and channels in this file and in the JSON are now YouTube's own, so the "Probably ..." and "Not confirmed" channels are settled. `verified` is true on every entry and `link_check` records the date.
- **Content: not yet watched.** `content_reviewed` is false on every entry. `what_it_shows` is still our reading of the title and the search summary. Before a video goes into the app, Alex (a qualified coach) watches it and checks three things: the content matches `what_it_shows`; the advice agrees with the rubric (for example, F03 says a bat that lifts towards the slips is fine); and, for a long video, the `start_seconds` of the useful part. Then set `content_reviewed` to true.
- **Not known yet:** video lengths. The watch pages stay blocked from this session, so `duration` is mostly empty; Alex sees the length when he opens each link.
- **Two channels to note:** "Toby Radford on the 60/40 Batting Weight Rule" is on The Cerebral Cricketer, and "How to play the off drive" (also said to feature Radford) is on Omnia Cricket Coaching, not Radford's own channel. Watch both to confirm that Radford is the coach shown. Thrive cricket academy is still flagged: its coaches' qualifications are not known.

## How the videos were chosen

- **Sources, in order of preference:** governing bodies, county academies, named coaches and former professionals; then established coaching channels.
- **Preferred:** short videos, drills a junior or club player can do with a partner, a tee or a tennis ball, and advice that matches the rubric's checkpoint text.
- **Left out:** highlight reels, gambling-sponsored channels, baseball and golf videos that turned up in searches, and channels whose maker could not be identified at all. One channel whose coaching credentials were not checked (Thrive Cricket) is included and flagged.
- **Governing bodies are thin.** Searches found Cricket Australia material (on cricket.com.au) and one NZC "Coaching Quick Fix". They found no ECB, CSA or ICC YouTube coaching video on the drive. The ECB, CSA and ICC manuals are still the evidence behind the rubric (see `docs/phase1/coaching_sources.md`). The videos only show the fix.
- **Same video, several faults.** Some videos appear under more than one fault, because one drive breakdown covers several checkpoints. The app should link to the part of the video that matters once start times are known.
- **Fit** gives a rough sense of how directly the videos address the fault: good (aimed at it), fair (covers it as part of a wider drive lesson), weak (only touches it). It is a judgement made from titles and descriptions, not from watching the videos.

## Videos by fault (chain order, then F00)

### F01: Head moves early or is not still at set-up

Fit: **good**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [This can COMPLETELY DESTROY your BATTING! - Fixing Incorrect head positions when batting](https://www.youtube.com/watch?v=wjEqqElR8ek) | CoachCricXI - Online Cricket Coaching | Common head-position faults (head moving or falling away) and drills to keep it still and over the ball. | club | coach | link yes, content no |
| 2 | [How To Bat In Cricket With The CORRECT Grip, Backlift & Set-Up / Technical Foundations Of Batting](https://www.youtube.com/watch?v=UxX4IQL03UU) | CRICKET LIFE STORIES | Toby Radford (former West Indies batting coach) works through grip, set-up and backlift: still head, hands close to the body. | club | coach | link yes, content no |
| 3 | [DRILLS to help you WATCH THE BALL better in Cricket](https://www.youtube.com/watch?v=0t08DPxOSqg) | CoachCricXI - Online Cricket Coaching | Five hand-eye drill sets with a partner or a wall that train a still head and eyes on the ball. | all | coach | link yes, content no |

### F02: Unbalanced stance

Fit: **good**. 4 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [The BEST cricket batting stance / Mark Ramprakash Batting Masterclass](https://www.youtube.com/watch?v=oTiREL-x6Gg) | Wisden Cricket | Former England batter and England batting coach Mark Ramprakash sets up a relaxed, balanced stance with the head still and eyes level. | all | professional | link yes, content no |
| 2 | [Toby Radford on the 60/40 Batting Weight Rule](https://www.youtube.com/watch?v=Nuzm6PRO1-0) | The Cerebral Cricketer | Radford's balance cue: keep the head stacked over the feet, as when standing and talking; weight split for moving either way. | all | coach | link yes, content no |
| 3 | [The SECRET to a PERFECT Cricket Batting STANCE / Everything you NEED to know](https://www.youtube.com/watch?v=xebUS0CCg6k) | CoachCricXI - Online Cricket Coaching | Stance set-up: feet about shoulder-width apart, weight even, knees soft, head still. | beginner | coach | link yes, content no |
| 4 | [Correct Batting Stance Facing Pace & Spin In Cricket - Gary Palmer Open Stance Coaching Masterclass](https://www.youtube.com/watch?v=7CgCjJRjvgc) | CRICKET LIFE STORIES | Gary Palmer (coach to Alastair Cook) on a stance that keeps the batter balanced and able to move either way. | advanced | coach | link yes, content no |

### F17: Head, hands and feet not lined up over off stump as the bowler lets go

Fit: **good**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [How to use a trigger movement when batting / Mark Ramprakash Batting Masterclass](https://www.youtube.com/watch?v=katRDmaYF1E) | Wisden Cricket | Ramprakash on a small trigger movement that gets the head, hands and feet set on the line as the bowler lets go. | club | professional | link yes, content no |
| 2 | [How To Bat In Cricket With The Correct Trigger Movement - Gary Palmer Coaching Tips](https://www.youtube.com/watch?v=2KJLpsFVwTg) | CRICKET LIFE STORIES | Palmer's masterclass on a trigger that sets the head, hands and feet in line over off stump at release. | advanced | coach | link yes, content no |
| 3 | [Are You Making This Trigger Movement Mistake? Fix It Now! 🚨🏏 /Batting Tips/Thrive Cricket](https://www.youtube.com/watch?v=iUa9qXZxjJc) | Thrive cricket academy | A trigger that opens the front shoulder and spoils alignment, and the foot and toe position that keeps the batter side-on and on off stump. | club | coach | link yes, content no |

### F13: Hands late or static: no rhythm from the backlift

Fit: **good**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [How To Perfect Your Backlift: Mark Ramprakash](https://www.youtube.com/watch?v=3svWej4YDCg) | Wisden Cricket | Ramprakash demonstrates the orthodox backlift: hands up and back, close to the body, top hand in control. | all | professional | link yes, content no |
| 2 | [Improve your BATTING TIMING in 4 minutes](https://www.youtube.com/watch?v=QCwSaXEJI2w) | CoachCricXI - Online Cricket Coaching | A short drill set for timing: let the backlift flow into the downswing in one rhythm. | all | coach | link yes, content no |
| 3 | [Batting Pick Up Tips / Toby Radford Coaching](https://www.youtube.com/watch?v=RYGaXnxF4yQ) | CRICKET LIFE STORIES | Short clip from a net session with Toby Radford on the pick-up (backlift) and getting it started in time. | club | coach | link yes, content no |

### F03: Hands go out away from the body in the backlift

Fit: **good**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [How To Perfect Your Backlift: Mark Ramprakash](https://www.youtube.com/watch?v=3svWej4YDCg) | Wisden Cricket | Ramprakash demonstrates the orthodox backlift: hands up and back, close to the body, top hand in control. | all | professional | link yes, content no |
| 2 | [How To Bat In Cricket With The CORRECT Grip, Backlift & Set-Up / Technical Foundations Of Batting](https://www.youtube.com/watch?v=UxX4IQL03UU) | CRICKET LIFE STORIES | Toby Radford (former West Indies batting coach) works through grip, set-up and backlift: still head, hands close to the body. | club | coach | link yes, content no |
| 3 | [Gary Palmer - Batting drill to develop high hands good alignment and balance.](https://www.youtube.com/watch?v=CaAeVsqGD4s) | Gary Palmer | A drill for high hands, a straight swing through the hitting zone and a balanced, aligned finish to the drive. | club | coach | link yes, content no |

### F19: Drives a ball that is too short to drive

Fit: **good**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [Batting Drill: Picking Length](https://www.youtube.com/watch?v=deGKL_sOwUY) | PitchVision Sports | Monty Desai (former West Indies batting coach) runs a drill that helps a young batter watch the ball and pick length early. | club | academy | link yes, content no |
| 2 | [How to JUDGE LENGTH - Cricket Batting Drills and Tips](https://www.youtube.com/watch?v=wH6fiC-P9Bo) | CoachCricXI - Online Cricket Coaching | Drills for judging length early so the batter drives only the full ball and goes back to the short one. | all | coach | link yes, content no |
| 3 | [Improve your SHOT SELECTION while judging LINE & LENGTH](https://www.youtube.com/watch?v=y6fVgUSkKa8) | CoachCricXI - Online Cricket Coaching | A partner drill for shot selection: read line and length, then choose drive, back-foot shot or leave. | club | coach | link yes, content no |

### F15: Moves across the crease, not out towards the ball

Fit: **fair**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [How to IMPROVE your BATTING footwork / Cricket Batting Drills](https://www.youtube.com/watch?v=ZGPXm8yAKVg) | CoachCricXI - Online Cricket Coaching | Four footwork drills, beginner to advanced, for moving out towards the ball instead of across the crease. | all | coach | link yes, content no |
| 2 | [FIX these OFF SIDE batting MISTAKES - Cricket Batting Drills](https://www.youtube.com/watch?v=UN6Ff31MQEo) | CoachCricXI - Online Cricket Coaching | Common off-side faults (falling across, opening up, reaching) with drills to fix each. | club | coach | link yes, content no |
| 3 | [Coaching Quick Fix / Accessing The Off-Side](https://www.youtube.com/watch?v=I8YV8hzydqA) | New Zealand Cricket | NZC quick fix for helping a batter get access to the off side: stay closed and on the line instead of falling across or opening up. | club | governing body | link yes, content no |

### F04: Head and shoulder do not lead to the line

Fit: **good**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [How to play the PERFECT defence / Mark Ramprakash Batting Masterclass](https://www.youtube.com/watch?v=5TcZlYt4AUQ) | Wisden Cricket | Front-foot defence: the head and front shoulder lead towards the line and the front foot follows; the same lead as the drive, at low speed. | all | professional | link yes, content no |
| 2 | [Use THESE 5 BATTING DRILLS to help you score more runs TODAY!](https://www.youtube.com/watch?v=j2cHkeoqNGk) | CoachCricXI - Online Cricket Coaching | Five drills: head position, getting into the line of the ball, and transferring the weight forward into the shot. | club | coach | link yes, content no |
| 3 | [Cricket batting/ How to play the off drive.](https://www.youtube.com/watch?v=sgZEt5Ag4Lo) | Omnia Cricket Coaching | Radford on the off drive: head and front shoulder lead the front foot to the line of the ball. | club | coach | link yes, content no |

### F05: Front foot well away from the line of the ball

Fit: **good**. 4 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [How to Play the PERFECT Cover Drive / Serious Cricket](https://www.youtube.com/watch?v=GlI27mVGzo4) | Serious Cricket | ECB Level 3 coach and former Hampshire player Richard Hindley walks through the cover drive: front foot to the line, head over the ball, controlled finish. | club | coach | link yes, content no |
| 2 | [Steve Snell #1 - The Cover Drive](https://www.youtube.com/watch?v=4sflN54fFho) | SomersetCricket | Somerset academy director Steve Snell's coaching tips on the cover drive: get to the line, keep the shoulder closed, play under the eyes. | club | academy | link yes, content no |
| 3 | [Cricket Batting Drills for Beginners- Front-Foot Practice with Drop Feed](https://www.youtube.com/watch?v=uzk8axtFVgc) | CRICDOJO | Drop-feed drill for under-7s to under-11s: step to the line and meet the ball close to the front pad, under the eyes. | beginner | academy | link yes, content no |
| 4 | [Meg Lanning teaches the cover drive / Coaching Masterclass](https://www.youtube.com/watch?v=ONFlBAU88y0) | Cricket Australia | Former Australia captain Meg Lanning's masterclass on the cover drive: front shoulder towards the ball, foot to the line, full face, controlled finish. | all | professional | link yes, content no |

### F06: Front leg stiff; weight stays back

Fit: **fair**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [Cricket Batting Drills for Juniors - Front-Foot Drive Power Drill](https://www.youtube.com/watch?v=ye16yb27sS4) | CRICDOJO | Junior drill for driving the weight forward: front knee bends, back foot pushes and the body moves out over the ball. | beginner | academy | link yes, content no |
| 2 | [Use THESE 5 BATTING DRILLS to help you score more runs TODAY!](https://www.youtube.com/watch?v=j2cHkeoqNGk) | CoachCricXI - Online Cricket Coaching | Five drills: head position, getting into the line of the ball, and transferring the weight forward into the shot. | club | coach | link yes, content no |
| 3 | [DRIVE Like a Pro - Full Cricket Drive Technique Breakdown + Drills](https://www.youtube.com/watch?v=yeImrfgNJoM) | CoachCricXI - Online Cricket Coaching | Full breakdown of the cover and straight drive, with drills: straight swing through the line and hands through to the target. | club | coach | link yes, content no |

### F18: Back foot stays put: no push to drive the weight forward

Fit: **weak**. 2 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [Cricket Batting Drills for Juniors - Front-Foot Drive Power Drill](https://www.youtube.com/watch?v=ye16yb27sS4) | CRICDOJO | Junior drill for driving the weight forward: front knee bends, back foot pushes and the body moves out over the ball. | beginner | academy | link yes, content no |
| 2 | [Use THESE 5 BATTING DRILLS to help you score more runs TODAY!](https://www.youtube.com/watch?v=j2cHkeoqNGk) | CoachCricXI - Online Cricket Coaching | Five drills: head position, getting into the line of the ball, and transferring the weight forward into the shot. | club | coach | link yes, content no |

### F16: Centre of mass stays back: does not move out over the ball

Fit: **weak**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [Toby Radford on the 60/40 Batting Weight Rule](https://www.youtube.com/watch?v=Nuzm6PRO1-0) | The Cerebral Cricketer | Radford's balance cue: keep the head stacked over the feet, as when standing and talking; weight split for moving either way. | all | coach | link yes, content no |
| 2 | [Use THESE 5 BATTING DRILLS to help you score more runs TODAY!](https://www.youtube.com/watch?v=j2cHkeoqNGk) | CoachCricXI - Online Cricket Coaching | Five drills: head position, getting into the line of the ball, and transferring the weight forward into the shot. | club | coach | link yes, content no |
| 3 | [Cricket Batting Drills for Juniors - Front-Foot Drive Power Drill](https://www.youtube.com/watch?v=ye16yb27sS4) | CRICDOJO | Junior drill for driving the weight forward: front knee bends, back foot pushes and the body moves out over the ball. | beginner | academy | link yes, content no |

### F07: Front shoulder opens early, or the top half never closes (includes F14, top half stays open)

Fit: **good**. 4 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [Meg Lanning teaches the cover drive / Coaching Masterclass](https://www.youtube.com/watch?v=ONFlBAU88y0) | Cricket Australia | Former Australia captain Meg Lanning's masterclass on the cover drive: front shoulder towards the ball, foot to the line, full face, controlled finish. | all | professional | link yes, content no |
| 2 | [Coaching Quick Fix / Accessing The Off-Side](https://www.youtube.com/watch?v=I8YV8hzydqA) | New Zealand Cricket | NZC quick fix for helping a batter get access to the off side: stay closed and on the line instead of falling across or opening up. | club | governing body | link yes, content no |
| 3 | [Steve Snell #1 - The Cover Drive](https://www.youtube.com/watch?v=4sflN54fFho) | SomersetCricket | Somerset academy director Steve Snell's coaching tips on the cover drive: get to the line, keep the shoulder closed, play under the eyes. | club | academy | link yes, content no |
| 4 | [FIX these OFF SIDE batting MISTAKES - Cricket Batting Drills](https://www.youtube.com/watch?v=UN6Ff31MQEo) | CoachCricXI - Online Cricket Coaching | Common off-side faults (falling across, opening up, reaching) with drills to fix each. | club | coach | link yes, content no |

### F08: Bat comes across the line

Fit: **good**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [DRIVE Like a Pro - Full Cricket Drive Technique Breakdown + Drills](https://www.youtube.com/watch?v=yeImrfgNJoM) | CoachCricXI - Online Cricket Coaching | Full breakdown of the cover and straight drive, with drills: straight swing through the line and hands through to the target. | club | coach | link yes, content no |
| 2 | [The #1 Straight Drive Mistake Batters Make](https://www.youtube.com/watch?v=PQajjXvzkuQ) | CoachCricXI - Online Cricket Coaching | The most common straight-drive mistake and how to swing straight down the line instead of across it. | club | coach | link yes, content no |
| 3 | [Gary Palmer - Batting drill to develop high hands good alignment and balance.](https://www.youtube.com/watch?v=CaAeVsqGD4s) | Gary Palmer | A drill for high hands, a straight swing through the hitting zone and a balanced, aligned finish to the drive. | club | coach | link yes, content no |

### F09: Pause or hitch in the swing

Fit: **fair**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [Improve your BATTING TIMING in 4 minutes](https://www.youtube.com/watch?v=QCwSaXEJI2w) | CoachCricXI - Online Cricket Coaching | A short drill set for timing: let the backlift flow into the downswing in one rhythm. | all | coach | link yes, content no |
| 2 | [How To Use A Trigger Movement When Batting: Mark Ramprakash](https://www.youtube.com/watch?v=7KvrXWOJJfI) | Wisden Cricket | Ramprakash on timing the trigger so the movement flows into the shot rather than stopping and starting. | club | professional | link yes, content no |
| 3 | [How to TIME THE BALL while BATTING / Cricket Batting Drills](https://www.youtube.com/watch?v=S_IrRtW0aC8) | CoachCricXI - Online Cricket Coaching | Favourite drills for timing the ball: smooth swing, late contact, no forcing. | club | coach | link yes, content no |

### F10: Head falls away at contact, or the ball is met away from the pad

Fit: **good**. 4 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [This can COMPLETELY DESTROY your BATTING! - Fixing Incorrect head positions when batting](https://www.youtube.com/watch?v=wjEqqElR8ek) | CoachCricXI - Online Cricket Coaching | Common head-position faults (head moving or falling away) and drills to keep it still and over the ball. | club | coach | link yes, content no |
| 2 | [DRILLS to help you WATCH THE BALL better in Cricket](https://www.youtube.com/watch?v=0t08DPxOSqg) | CoachCricXI - Online Cricket Coaching | Five hand-eye drill sets with a partner or a wall that train a still head and eyes on the ball. | all | coach | link yes, content no |
| 3 | [How to Play the PERFECT Cover Drive / Serious Cricket](https://www.youtube.com/watch?v=GlI27mVGzo4) | Serious Cricket | ECB Level 3 coach and former Hampshire player Richard Hindley walks through the cover drive: front foot to the line, head over the ball, controlled finish. | club | coach | link yes, content no |
| 4 | [Cricket Batting Drills for Beginners- Front-Foot Practice with Drop Feed](https://www.youtube.com/watch?v=uzk8axtFVgc) | CRICDOJO | Drop-feed drill for under-7s to under-11s: step to the line and meet the ball close to the front pad, under the eyes. | beginner | academy | link yes, content no |

### F11: Off balance at the finish

Fit: **weak**. 2 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [4 Batting Drills To Improve YOUR Game Against Pace & Spin / Learn How To Bat In Cricket From A Pro](https://www.youtube.com/watch?v=RjDGEpWtfmQ) | CRICKET LIFE STORIES | Former county batter Hassan Azad's top-hand, bottom-hand and batting-balance drills, ending in a held, balanced finish. | club | professional | link yes, content no |
| 2 | [Gary Palmer - Batting drill to develop high hands good alignment and balance.](https://www.youtube.com/watch?v=CaAeVsqGD4s) | Gary Palmer | A drill for high hands, a straight swing through the hitting zone and a balanced, aligned finish to the drive. | club | coach | link yes, content no |

### F12: Hands stop at contact, or the follow-through is wild

Fit: **fair**. 3 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [DRIVE Like a Pro - Full Cricket Drive Technique Breakdown + Drills](https://www.youtube.com/watch?v=yeImrfgNJoM) | CoachCricXI - Online Cricket Coaching | Full breakdown of the cover and straight drive, with drills: straight swing through the line and hands through to the target. | club | coach | link yes, content no |
| 2 | [How to Play the PERFECT Cover Drive / Serious Cricket](https://www.youtube.com/watch?v=GlI27mVGzo4) | Serious Cricket | ECB Level 3 coach and former Hampshire player Richard Hindley walks through the cover drive: front foot to the line, head over the ball, controlled finish. | club | coach | link yes, content no |
| 3 | [Meg Lanning teaches the cover drive / Coaching Masterclass](https://www.youtube.com/watch?v=ONFlBAU88y0) | Cricket Australia | Former Australia captain Meg Lanning's masterclass on the cover drive: front shoulder towards the ball, foot to the line, full face, controlled finish. | all | professional | link yes, content no |

### F00: Keep it grooved (nothing major to fix)

Fit: **good**. 4 video(s), all links checked; content not yet watched.

| # | Video | Channel | What it shows | Level | Source | Verified |
|---|---|---|---|---|---|---|
| 1 | [In the nets with Pez: Drills to improve batting](https://www.youtube.com/watch?v=Fir4vlTsVHw) | Cricket Australia | Ellyse Perry shares her favourite batting drills: purposeful, repeatable net and throw-down work. | all | professional | link yes, content no |
| 2 | [How to Practice Cricket Batting at Home / Serious Cricket](https://www.youtube.com/watch?v=KhKhlJpjW10) | Serious Cricket | Richard Hindley's home-practice routine: simple solo and partner drills to keep repeating a sound drive. | all | coach | link yes, content no |
| 3 | [Cricket Batting Drill Instantly Makes Nets Better](https://www.youtube.com/watch?v=yRPo1Mw2lJ0) | PitchVision Sports | PitchVision's point that 'just batting' in the nets does little; a drill that gives each ball a target so good technique is repeated with purpose. | club | academy | link yes, content no |
| 4 | [Cricket Batting Techniques For Beginners - Front-Foot Drive with Bobble Feed](https://www.youtube.com/watch?v=vp6ZBJFAI3I) | CRICDOJO | Bobble-feed (rolled or bounced tennis ball) drive drill a parent or partner can feed; good for lots of repetitions. | beginner | academy | link yes, content no |

## Gaps: faults with no good video

- **F18 (back foot does not push the weight forward).** No video found that is aimed at this. The two listed cover weight transfer in general. Options:
  - Ask a coach on the list below for a 30-second clip.
  - Use a short written drill instead. PitchVision's "one-leg front foot drive" article fits: drive with the back foot lifted, so the weight has to go forward. It is a text article, so it is not in the JSON.
- **F16 (centre of mass stays back).** Same problem as F18. Toby Radford's short "head stacked over the feet" clip is the closest match. The channel that posted it is not confirmed.
- **F11 (off balance at the finish).** Only two candidates, and neither is short:
  - Hassan Azad's balance drill sits inside a longer video.
  - Gary Palmer's drill is about high hands and alignment as well as balance.
  - The many generic "balance drill" videos found came from channels that could not be identified, so they are not listed.
- **F09 (pause or hitch).** No video is about a hitch as such. The listed ones are timing and rhythm drills, which are the usual fix. The rubric says a brief settle of the front foot is normal: check that the chosen video does not teach against that.
- **F12 (hands stop or wild follow-through).** Covered only inside wider drive lessons. A short "hands through to the target, hold the finish" clip would fill this.
- **Governing-body coverage** is the general gap. An ECB or Cricket Australia junior drill series on the drive would be the best single addition.

## Embedding rules (read before anything goes live)

1. **Use YouTube's own embedded player only.** Use an `<iframe>` with `https://www.youtube-nocookie.com/embed/<VIDEO_ID>`, as stored in `embed_url`. YouTube's terms allow this; they do not allow:
   - downloading videos;
   - re-uploading them, or hosting copies ourselves;
   - cutting them into our own clips;
   - stripping YouTube's branding, or hiding or covering the player's controls.
2. **Use privacy-enhanced mode** (the `youtube-nocookie.com` domain). It stops YouTube from storing viewer information until the video is played, but it does not stop all storage once playback starts. Under UK cookie rules (PECR) that playback is still non-essential storage. So:
   - load the player only after the user taps a "Play video (YouTube)" placeholder;
   - say in the privacy notice that YouTube may then set cookies;
   - most viewers are juniors, so keep this conservative. See `docs/legal/` for the consent wording.
3. **Start times.** Add `?start=<seconds>` to the embed URL (the JSON does this whenever `start_seconds` is set). Do not try to "trim" a video in any other way.
4. **Credit the channel.** Show the channel name with a link to the original video next to every embed, for example "Video: CoachCricXI on YouTube".
   - Do not suggest that the channel endorses Batting IQ unless it has agreed to.
   - Do not put our own branding over their video.
5. **Check "embedding allowed" before going live.** Some owners turn embedding off. Then the player shows "Video unavailable" or "Watch on YouTube". Test every embed on the live site. If embedding is off, show a plain link, or pick another video.
6. **Re-check links every quarter.** Videos get deleted, made private, age-restricted or blocked by region. Each quarter:
   - open every link;
   - mark any dead entry and replace it;
   - record the check date.

   A simple automated check is to call YouTube's oEmbed endpoint for each ID (`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=<ID>&format=json`). A 401 or 404 response means the video cannot be embedded or is gone. Running it needs network access to youtube.com, which this session does not have.
7. **No monetisation tricks.** Do not autoplay, loop or mute-autoplay videos to inflate views, and do not put ads around embedded videos.
8. **Child safety.** Before a video is used, check the end screen and the suggested videos. With the `rel=0` parameter, suggestions come only from the same channel. Leave out any channel whose other uploads are unsuitable for juniors.

## Channels to ask for permission to feature more prominently

Embedding needs no permission. Featuring a channel prominently does: for example, a named "drill of the week" in the app's drill plan, or a request for a short custom clip for a gap above. Asking first is courteous and avoids any suggestion of endorsement. Suggested order:

| Channel / coach | Why | Used for |
|---|---|---|
| Cricket Australia (cricket.com.au) | Governing body; Meg Lanning and Ellyse Perry masterclasses | F05, F07, F12, F00 |
| New Zealand Cricket (NZC Coaching) | Governing body; its "Coaching Quick Fix" format suits a short-tip app | F15, F07 |
| CoachCricXI (Byron Fraser, Zendon Verster) | Widest coverage of our faults, drill-based, club level | 10+ faults |
| Toby Radford | Our rubric already draws on his coaching; he could fill the F16 and F18 gaps | F01, F02, F03, F04, F13, F16 |
| Serious Cricket (Richard Hindley, ECB Level 3) | UK coach, clear drive and home-practice videos | F05, F10, F12, F00 |
| Wisden (Mark Ramprakash series) | Former England batter; short masterclasses | F02, F03, F04, F09, F13, F17 |
| Gary Palmer (Palmer Batting Lab) | Technical batting coach; balance and alignment drills | F02, F03, F08, F11, F17 |
| PitchVision Academy | Established coaching site; drill and length-reading videos | F19, F00 |
| Somerset CCC Academy | County academy; cover-drive tips | F05, F07 |
| cricdojo | Junior drop-feed and bobble-feed drills that parents can copy | F05, F06, F10, F00 |

### Draft request email (for Alex to send)

> **Subject:** Featuring your batting drills in Batting IQ (with full credit)
>
> Dear [name],
>
> I'm Alex, and I'm building Batting IQ, an app that helps junior and club cricketers improve their front-foot drive. A player films a drive, and the app picks out the one thing to work on first, for example "front foot lands well away from the line of the ball". It then suggests a drill.
>
> Your video "[video title]" ([link]) explains exactly that fix, and we'd love to point players to it. At the moment we would show it only through YouTube's own embedded player, with your channel name and a link back to the original. We would never download, edit or re-upload your content.
>
> We'd also like to feature your drill more prominently, for example as a named drill in a player's practice plan, credited to you. Would you be happy for us to do that? If you'd rather we didn't embed your videos at all, just say and we won't.
>
> If it's of interest, we'd also be glad to talk about a short clip for a couple of faults where we couldn't find a good video. These are pushing off the back foot to drive the weight forward, and holding a balanced finish. We'd credit it, and we can discuss payment.
>
> Many thanks for the coaching you share. It's a big help to young players.
>
> Best wishes,
> Alex
> Batting IQ
> [email] · [website]

Keep a record of each reply, including any conditions such as a required credit wording or a no-embed request. Store it alongside this file before featuring that channel.
