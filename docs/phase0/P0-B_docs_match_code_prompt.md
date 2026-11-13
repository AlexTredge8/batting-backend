# P0-B — Make the docs match the code

Run after P0-A is committed. Paste everything below the line into a new Codex thread, in the `~/Documents/battingiq-phase0` clone, on branch `phase0-baseline`.

---

## Goal

Every project document names the same active rules as the code: A5, T2, S4, F4. Record in the changelog the rule changes of 13–15 May 2026 that are in the code but not in the docs. Scoring output must not change.

## Facts to start from (check each in the code before you write it)

- `config.PILLAR_RULE_MAX_DEDUCTION` holds A5, T2, S4, F4 only. `SUSPEND_A1`, `SUSPEND_A3` and `SUSPEND_F6` are `True`.
- `RULES_CHANGELOG.md` stops at 2.1.0 (11 May 2026), which lists 7 active rules (A1, A3, A5, T2, S4, F4, F6). `config.RULES_VERSION` is still `"2.0.0"`.
- The code holds later changes, dated in comments in `config.py` and `coaching_rules.py`: A3 suspended (13 May), A1 suspended (14 May), F6 re-suspended (14 and 15 May), the S4 sign convention fixed and its bands tightened (14 May), the T2 minimum threshold raised (14 May), the F4 threshold raised from 3 to 5 frames (14 May), and setup detector v4 restored as the default (`USE_SETUP_V4=True`, 14 May). Read the comments for the exact wording and numbers.
- `claude.md` and `ORCHESTRATOR.md` say 11 active rules and "21 rules". `claude.md` lists `ground_truth_scores.csv` and `annotation.csv`. Neither file exists. The coach scores are in `coach_ground_truth_from_screenshot.csv`.
- `ORCHESTRATOR.md` says "All 21 coaching rules scale thresholds relative to this baseline". This is false for the active set. Search for every read of `baseline[` and `baseline.get(` and list where `reference_baseline.json` is really used. I expect: suspended S1, the A6 and S1 entries in `collect_rule_measurements`, and the development-notes text in `scorer.py`.
- `latest_developments.md` says that setup stays at v1. The code default is v4.
- After P0-A, `Viv P_Offdrive Elite.mov` has the split value `reference` in `heldout_split.csv`. The held-out tables in `claude.md` and `heldout_discipline.md` still list it as held out.

## Tasks

1. `RULES_CHANGELOG.md`: add a `2.2.0` entry, dated today, that lists each 13–15 May change above with its date and reason, taken from the code comments. Then add: `Active rules: A5, T2, S4, F4 (4 active). Suspended: …. Deleted: A6, T3, T4, T5.`
2. `config.py`: set `RULES_VERSION = "2.2.0"`. This is the only code change allowed.
3. `claude.md`, `ORCHESTRATOR.md` and `README.md`: correct the rule counts, the active, suspended and deleted lists, and the data-file names. Correct the reference-baseline claim with the list from your search. In `claude.md`, mark the "Success criterion" section `Superseded (Nov 2026)` and replace it with these provisional targets, measured on 20 locked clips from unseen players, fully automatic:
   - system error is no more than 1.25 × coach-to-coach error, overall and per pillar;
   - no ability band has a mean bias above 5 points;
   - priority-fix match is within 10 percentage points of the coach-to-coach rate;
   - at least 80% of clips that meet the capture standard get a score;
   - the same clip gives the same score.

   State that the ceiling is measured in Phase 2, and that tier order is a sanity check only.
4. `heldout_discipline.md` and the held-out table in `claude.md`: move Viv P_Offdrive Elite to a "Reference clip" line with the reason (same file as `test_batting.mov`, md5 `0a1b3a74e1f123a97085c2fafbaa12ea`). Add one line: the player-level split in Phase 4 replaces this split, because AlexT_Ondrive Good is held out while Alex T_Straightdrive Good is in tuning.
5. `latest_developments.md`: correct the setup-detector line. Add a dated note that the 13–15 May changes are now logged as 2.2.0.
6. Docstrings and comments only: `coaching_rules.py` says "21 rules" in its module docstring and in `run_all_rules`. Change these to say 21 rules are defined and 4 are active.

## Constraints

- No change to logic, thresholds or flags. The `RULES_VERSION` string is the only code edit.
- After the edits, run `python -m pytest tests -q` and `BATTINGIQ_E2E=1 python -m pytest tests/test_reference_video_e2e.py -q`. Both must pass with the same counts as in P0-A.
- Then run `python run_analysis.py test_batting.mov --output-dir /tmp/p0b_check`. Compare the score and the four pillars with the P0-A auto row for `Viv P_Offdrive Elite.mov`. They must be the same.
- Commit as `P0-B: docs match code (A5, T2, S4, F4); RULES_VERSION 2.2.0`. Do not push.

## Deliverable

- The changelog entry, inline.
- A list of each file changed, with one line per change.
- The list of places that read the reference baseline.
- The test results and the score check.
