# Web video sources: can they be used? (research, 15 Nov 2026)

Question from Alex: can clips from the web give hundreds of training/validation clips? Desk research by a research agent (no video downloaded). Several sites could only be read through search-engine extracts; marked (S).

| Source | Drives | Angle | Labels | Commercial use |
|---|---|---|---|---|
| CricShot10 (Sen et al., Sensors 2021), github.com/ascuet/CricShot10 | ~188 cover, ~189 straight (S) | broadcast | shot type | No: README "research purposes only"; footage is YouTube broadcast |
| CricShot10k, github.com/Dihan69/CricShot10k | 793 cover, 419 straight | broadcast | shot type | No: "solely for non-commercial research" |
| CricStroke, github.com/KumailAbb/CricStroke | unknown share of 3,660 | broadcast **front pitch view** | shot type, frames | No / unclear: no licence; YouTube IDs only |
| Kaggle IPL 2023 / Hugging Face rokmr/cricket-shot | ~150–250 / repackaged CricShot10 | broadcast | shot type | No in practice: "Apache-2.0" label cannot clear broadcast copyright |
| CricRAG (arXiv 2609.26056) | 288 smartphone clips incl. straight drive (S) | smartphone, mixed skill | **coach quality labels** (stance, weight transfer, shot quality) | Unclear: on request from the authors. **Best fit: ask for a commercial licence.** |
| Wikimedia Commons (amateur club cricket) | dozens of clips, few from behind the bowler | mostly boundary | none | Yes with attribution, CC BY-SA 4.0 |
| Pexels / Pixabay | small, mixed | stock | none | No: their terms bar ML/AI use without written permission |
| YouTube CC BY | — | — | — | Copyright cleared if the uploader owns it, but downloading breaches YouTube's terms; ask creators for the files instead |

UK law: the text-and-data-mining exception (CDPA s29A) covers non-commercial research only; the government dropped the commercial extension (2023, and again in the March 2026 copyright and AI report).

**Conclusions.**
- Broadcast datasets hold roughly 1,200–1,600 drive clips, but none is licensed for a commercial app.
- Openly licensed sources hold an estimated 20–50 usable clips from behind the bowler. That is enough for a spot check, not for training.
- Realistic routes to hundreds of clips:
  1. App uploads, under the app's terms and a clear consent, with Cam rating them.
  2. Clubs and academies filming under a written consent and licence.
  3. Licensing from a coaching-content creator.
  4. Asking the CricRAG authors for their clips.
- Elite broadcast clips would add almost nothing on the hardest problem: ordering beginners correctly.
