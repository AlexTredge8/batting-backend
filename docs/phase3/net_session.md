# Net session: two-phone test and filming guidance (Phase 3)

Purpose: prove that the same ball gives the same answer from two allowed phone positions (Phase 5 entry test 2), and set the filming rules the app gives users. About 90 minutes in one net lane.

## Kit

- Two phones on stands or tripods, back camera, 1080p. Use 60 fps if the phone offers it; otherwise 30.
- A bowling machine or a steady bowler, with full-length deliveries for front-foot drives.
- 3–4 batters across levels (one beginner, one club, one good player). For under-18s, get parental consent first, as for the existing clips.
- A tape measure.

## Positions

- **Phone A (the standard):** directly behind the bowler's stumps, at chest height, with the batter filling about a third of the picture's height. Back camera, not mirrored.
- **Phone B (the allowed variation):** next to A, but 1 m to one side and about 30 cm higher. Same zoom.
- Keep both still for the whole session. Note both distances to the batting crease.

## What to film

1. **Pairs (the main job).** Each batter plays 10 drives, mixing off, straight and on. Both phones record every ball. Start both before the bowler runs in and stop after the follow-through. Clap once in view before each ball, so the two clips can be lined up. Target: 40 pairs.
2. **Bad set-ups for the gate (5 minutes).** One ball each with: phone A twice as far away; phone A moved 45° to the side; the front (selfie) camera, which mirrors the picture; and phone B behind the batter. The gate must reject or warn on each of these.

## Naming

Name the files `<batter>_<ball number>_A.mov` and `<batter>_<ball number>_B.mov`, for example `P07_03_A.mov`. The pairs sheet for test 2 is then made automatically:

```
python -c "import csv,glob,os; w=csv.writer(open('two_phone_pairs.csv','w',newline='')); w.writerow(['filename_a','filename_b']); [w.writerow([os.path.basename(a), os.path.basename(a)[:-6]+'_B'+a[-4:]]) for a in sorted(glob.glob('<folder>/*_A.*'))]"
```

## Afterwards

- Validate the key frames on phone A clips only, as for the existing 18.
- Run the Phase 5 measurements on all clips, then `phase5_entry_tests.py --two-phone two_phone_pairs.csv`.
- A measurement passes test 2 when the median difference between the two phones is at most a quarter of the spread between clips.
- If several measurements fail only because of phone B's position, tighten the filming rule rather than drop the measurement. That is what "allowed" means in the plan.
- The pair clips also count towards the 60-clip Phase 4 target (phone A only), once two coaches have scored them blind.
