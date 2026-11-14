# Phase 1 and Phase 2: rubric, then the coach ceiling

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

## Model as a fourth coach (approved spend under £5; waiting on consent)

```bash
python phase2_model_pack.py --pack <pack> --video-dir "$VIDS" --baseline-reports baselines/phase0_<date>/reports/auto
```

This writes six stills and a 13-frame contact strip per clip, plus `prompt.md`, locally. **Nothing is sent.** Two things must happen before any image goes to a provider: parental consent for the junior players, and a check of the provider's data terms. Both are Alex's to decide.

Once they are done, save the model's answers as `<pack>/sheets/Model_score_sheet.csv`, in the same columns as a coach sheet. `phase2_agreement.py` then measures it exactly like a coach, without counting it in the human ceiling.

## Decision point (end of Phase 2)

Alex decides whether to continue to Phases 3–7 (about 64 hours) with the ceiling number in hand, or to stop with a known ceiling.
