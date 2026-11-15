# Methods for scoring with few clips (survey, 15 Nov 2026)

Sources were read from abstracts and READMEs; the hosts that would have allowed a full check were blocked.

**What the survey supports.**
1. **Score order, not absolute numbers.** Train and test on comparisons: coach scores turned into within-coach pairs, plus swipe pairs. Fit Bradley–Terry with `choix` (MIT). The coach-to-coach agreement (Spearman 0.95) is the ceiling. Doughty et al. (CVPR 2018) reach 70–83% of pairs correctly ordered on skill videos.
2. **More comparisons, chosen well.** Active pair selection (ASAP, MIT) for the swipe tool. Reliability about 0.7 needs 10–14 comparisons per clip; about 0.9 needs 26–37 (Verhavert et al. 2019). For 300 clips that is about 1,800 judgements, roughly 5 hours. Repeat about 5% of pairs to measure each coach's consistency.
3. **Interpretable model.**
   - 10–20 pose features anchored on contact.
   - A regularised pairwise logistic model (linear Bradley–Terry with features), with signs constrained where coaching theory is clear.
   - Monotone gradient boosting only above about 150 clips.
   - Cross-validation grouped by batter, with bootstrap intervals. At 60 clips a Spearman of 0.7 has a 95% interval of about 0.54–0.81.
4. **Contact first.** Audio onset (the current method) with a pose fallback; padel hit detection reaches F1 92% from audio.
5. **Pose model.**
   - RTMPose-m through rtmlib on ONNX Runtime: about 11 ms per frame on CPU, and the most robust in independent comparisons.
   - MediaPipe was the weakest for accuracy.
   - Sapiens and the GolfDB code are non-commercial: not usable.
   - Check the licences of RTMPose's training data before commercial use.
6. **Video language models: wording only, never the score.** Our Gemini result agrees with a 2026 CVPR workshop study (Gemini 3.1 Pro and other models "only marginally above random" at judging action quality).
7. **Treat cricket accuracy claims with caution.** Re-implementations of published shot classifiers reached 10.6–57.7% against the 93–99% originally claimed (Kang 2025).

**Not for us yet:** end-to-end video quality models (CoRe, TSA, PECoP, E2E-Spot). They need a GPU and more than 1,000 clips.
