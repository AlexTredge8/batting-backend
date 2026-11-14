# Phase 1 and Phase 2: rubric, then the coach ceiling

## Links (private; share each from its Share menu)

- **Which drive is better?** (the left-or-right session): https://claude.ai/artifact/XXn9i9Xij479pt3Zn6eVLi
  Each coach picks their name and chooses the clips from their own zip; the clips play on their device only. Answers save to the page's store, one record per coach. To save, Cam and Viv must be Contributors (in your organisation) or invited by email as Editors; anyone else sees a warning and can save a file of their answers instead. Source: `docs/phase2/pairs_app.html`. Claude reads the answers with ArtifactData; `phase2_pairs.py from-artifact` turns them into pairs sheets.
- **Fourth coach test** (Alex only): https://claude.ai/artifact/TtMGpghZspSmVoBqzkaEfM
  Choose the pack's `model_inputs` folder and press Score. Claude scores each clip from still frames under Alex's account, with three of Alex's scored clips as references; results save to the page's store and as `Model_score_sheet.csv`. Source: `docs/phase2/model_coach_app.html`.

## One command on the Mac

```bash
cd ~/Documents/battingiq-phase0 && git pull --rebase && ./phase2_mac_pack.sh
```

This builds the blind pack next to the clip folder, `for_<coach>.zip` per coach (only the clips they may see, never the key), and the model frames.

## Phase 1: rubric (Alex, Cam, Viv)

The draft is [`docs/phase1/rubric_v0.md`](../phase1/rubric_v0.md). Its source of truth is `rubric_v0.json`; the tools read the IDs from there. It has 12 checkpoints that this camera can see, pillar anchors at 5/15/25, 12 faults in chain order, three proposed caps, and a "cannot judge" option.

Phase 1 is done when all three coaches accept it (edit the JSON and re-generate the page) and Alex can score a clip with it in under 5 minutes. Cam and Viv review the rubric, not the scores.

## Phase 2: blind scoring (about 8 hours for Alex, 1.5 hours each for Cam and Viv)

All commands run on the Mac in the Phase 0 clone. The pack goes **outside** the repo: it holds player clips and the private key.

```bash
. .venv310/bin/activate
VIDS="$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:"

# 1. Blind pack: random IDs, metadata stripped, a new order for each coach, no one scores themselves, 60 pairs each
python phase2_blind_pack.py --video-dir "$VIDS"      # writes ../blind_pack_<date>/ next to the clips

# 2. Send each coach their sheets/<name>_score_sheet.csv, pairs/<name>_pairs.csv, the clips/ folder,
#    README_for_coaches.txt and the rubric. Never send blind_key_PRIVATE.csv.
#    Alex re-scores too (his repeat error is the cheapest ceiling).

# 3. When the sheets come back, put them in <pack>/sheets/ and run:
python phase2_pairs.py fit --pairs-dir <pack>/pairs --key <pack>/blind_key_PRIVATE.csv --out <pack>/report
python phase2_agreement.py --sheets-dir <pack>/sheets --key <pack>/blind_key_PRIVATE.csv \
    --pairwise-scale <pack>/report/pairwise_scale.csv --out <pack>/report
```

`agreement_report.md` gives Alex's repeat error, the coach-to-coach error overall and per pillar, bias by band, fault-pick agreement, ICC(2,1) with a 95% interval, pillar correlation between coaches (below 0.5 sends the pillar back to Phase 1), the halo check, and the targets with the provisional numbers replaced. It contains IDs and filenames only, no images, so it can be committed to `baselines/phase2_<date>/`.

## Model as a fourth coach (consent given 14 Nov, on condition the data stays private)

```bash
python phase2_model_pack.py --pack <pack> --video-dir "$VIDS" --baseline-reports baselines/phase0_<date>/reports/auto
```

This writes six stills and a 13-frame contact strip per clip, plus `prompt.md` and `references.json`, locally. Alex confirmed on 14 Nov that he has consent from the people in the clips, provided their data stays private. The frames go to Claude only from the private Fourth coach test page, under Alex's own account.

Once they are done, save the model's answers as `<pack>/sheets/Model_score_sheet.csv`, in the same columns as a coach sheet. `phase2_agreement.py` then measures it exactly like a coach, without counting it in the human ceiling.

## Decision point (end of Phase 2)

Alex decides whether to continue to Phases 3–7 (about 64 hours) with the ceiling number in hand, or to stop with a known ceiling.
