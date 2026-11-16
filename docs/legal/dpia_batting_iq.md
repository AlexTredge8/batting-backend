Prepared by AI acting in the role of a data-protection reviewer. Not legal advice. Items marked HIGH RISK or UNVERIFIED should be decided by Alex knowingly.

# Data protection impact assessment (DPIA): Batting IQ

| | |
|---|---|
| Controller | [Alex Tredgett, trading as Batting IQ / Batting IQ Ltd], [Address] |
| ICO registration | [number] (see `ico_fee_and_registration.md`) |
| Prepared by | AI reviewer, 16 November 2026, for Alex Tredgett |
| Owner and sign-off | Alex Tredgett |
| Status | **Draft: not signed.** Do not launch the training opt-in or use any more under-18 clips until section 7 is signed. |
| Review | Before each new feature that touches videos or children's data, and at least every 12 months |

This follows the structure of the ICO's sample DPIA template and the Children's code Annex D template: need, processing, consultation, necessity and proportionality, risks, measures, sign-off (UNVERIFIED (snippet): S9, S8 in `dp_review.md`). Source marks follow `dp_review.md`. Facts about the system come from the repository [R].

---

## Step 1. Why a DPIA is needed

Batting IQ analyses phone videos of cricketers, many under 18, to give automated scores and tips, and (with consent) uses some videos to train its own model.

A DPIA is required because:
- **Children's data, in an online service likely to be used by children.** The Children's code says such services inevitably involve processing likely to result in high risk to children, so a DPIA is expected (standard 2) (UNVERIFIED (snippet): S8).
- **Innovative technology plus another criterion.** AI-based pose estimation and automated scoring is a new application of existing technology, combined with **evaluation or scoring** of individuals and **vulnerable people (children)**. The ICO's list requires a DPIA for innovative technology combined with another criterion (UNVERIFIED (snippet): S7).
- **Video of identifiable people**, including bystanders.
- **Use of the data for a new purpose** (model training), and **transfers** to US-based processors.

Conclusion: **DPIA required** (UK GDPR Article 35).

---

## Step 2. Describe the processing

### 2.1 Nature: how data is collected, used, stored and deleted

| Stage | What happens | Systems [R] |
|---|---|---|
| Sign-up | Name, email, date of birth, password (hashed) | Lovable web app; Supabase Auth |
| Upload | User picks a phone video of a front-foot drive. It goes to the private `battingiq-uploads` bucket. A row in `uploads` records it, with any training choice | Lovable; Supabase Storage and database |
| Analysis | The FastAPI server on Railway fetches the video, runs MediaPipe pose estimation (body points frame by frame), applies coaching rules, and returns a score, faults, tips and storyboard stills. Working files are deleted after each job (`api.py`) | Railway |
| Results | Stills saved under `analysis-media/` in the same bucket; signed links that expire after 1 hour | Supabase Storage |
| Share card (optional) | Made in the user's browser; the user shares it through the phone's share sheet. Nothing is posted by Batting IQ | User's device |
| Training and testing (consent only) | Consented clips are filtered by script, downloaded to Alex's encrypted Mac, pose points extracted, coach ratings collected (face-blurred clips, coded), model retrained, agreement measured. A manifest records which clips each run used | Mac; rating pages; Supabase |
| Development checks (one-off) | A few face-blurred clips sent to Google's Gemini API (paid tier) to compare with the app's analysis, within a set budget, never in the live app | Google Gemini API |
| Session filming (Phase 4) | Cam films players with paper consent forms; clips coded; 24-month deletion | Cam's phone; private storage |
| Withdrawal and deletion | Switch in "My videos and data"; Edge Function deletes the file and row; scripts remove local copies within 30 days | Supabase; Mac |

### 2.2 Scope

| | |
|---|---|
| Data | Video (body, kit, surroundings, sometimes faces and bystanders); pose points; scores, faults, tips; stills; account details; consent records; parent or guardian name and email; coach ratings |
| Special category data | **None intended.** Pose data is not used to identify anyone, so it is not special category biometric data (Issue 5 in `dp_review.md`). No health inferences |
| Criminal offence data | None |
| People | Account holders (13+); batters (all ages, many under 18); bystanders; parents and guardians; coaches who rate |
| Volume | Early stage: 18 original clips, Phase 4 session clips, app uploads (tens to low hundreds at launch; Alex to fill in) |
| Frequency | Continuous (every upload) |
| Keeping periods | See `training_data_terms_v2.md`, section 6 |
| Geography | UK users; processors in the UK, EU and USA |

### 2.3 Context

- Many users are juniors, some under 13 (through a parent's account).
- People expect to get coaching feedback. They would **not** expect their video to be used to train a commercial model unless asked.
- Cricket clubs work under ECB safeguarding guidance on filming children; parents are used to signing filming consent.
- The technology (pose estimation, AI scoring) is new to most users and could seem intrusive if not explained.
- Batting IQ is run by one person with a coach (Cam) helping. Limited time (Restriction 1 [R]). No solicitor (Alex's decision).
- Known concern in this area: misuse of children's images, and the accuracy of automated judgements about children's performance.

### 2.4 Purposes

1. Give users coaching feedback on their batting (the service).
2. Improve the accuracy of that feedback by training and testing the model on consented clips, checked against qualified coaches.
3. Keep the service secure and handle rights requests.

Benefit to users: cheap, quick coaching feedback, especially for juniors without regular one-to-one coaching. Benefit to Batting IQ: a working product and a lawful training set.

---

## Step 3. Consultation

| Who | How | Status | What to ask or record |
|---|---|---|---|
| Cam (qualified coach) | Conversation | To do | Is face-blurred, coded rating practical? Any safeguarding worries? |
| Club welfare or safeguarding officer | Email the forms and this summary | To do | Do the forms fit the club's filming policy? |
| 3 to 5 parents and 2 to 3 juniors (13 to 17) | A 3-question message: (1) Is the upload choice clear? (2) Would you say yes, and why? (3) Anything worrying? | To do | Record answers in a line each, no names |
| Processors | Read their DPAs and security pages | To do (see 6.1, M6) | Region, transfer safeguard, breach notice time |
| ICO | Free helpline or web chat, only for a specific question | Optional | Prior consultation (Article 36) is needed only if a high risk remains after the measures (UNVERIFIED (snippet): S9). This DPIA expects it will not |
| Information security | Self-check against the list in M7 | To do | — |

---

## Step 4. Necessity and proportionality

### 4.1 Lawful bases

| Processing | Basis | Reason |
|---|---|---|
| Accounts; analysis of your own video | Contract (Art 6(1)(b)) | Needed to give the service |
| Analysis of someone else's video; bystanders | Legitimate interests (Art 6(1)(f)) | See 4.3 |
| Training and testing; coach rating | Consent (Art 6(1)(a)) | See `dp_review.md` Issue 2 |
| Development checks with Gemini | Consent (separate tick on session forms) | Specific purpose; outside provider |
| Consent records; security | Legitimate interests | Proof of consent; safe service |
| Rights requests and complaints | Legal obligation (Art 6(1)(c)) | UK GDPR rights; s164A DPA 2018 |

**Article 9:** not engaged (no processing for unique identification; no health inference). **Decision for Alex:** keep it that way; any feature that matches people by body, movement or face needs a new DPIA and explicit consent.

### 4.2 Does the processing achieve the purpose, and is there a less intrusive way?

| Question | Answer |
|---|---|
| Is video needed? | Yes for the analysis: technique is in the movement. For training, much of the work can use pose points rather than the video; video is still needed for coach rating and for re-extracting points with better models |
| Could pose points alone be kept? | Partly. Measure M10: training runs keep pose points and drop local video copies once points and ratings exist, where practical |
| Are faces needed? | No. Faces are blurred for anyone outside the team; face points not used by the rules are dropped (M10) |
| Is location needed? | No. Strip it (M9) |
| Is a full date of birth needed? | Only at sign-up, to apply age protections. The training choice uses "Who is the batter?" and the profile age |
| Is the parent's name and email needed? | Yes, to show consent for under-18s and to confirm it. Kept only for the consent record period |
| Is an outside AI provider needed? | Not for the live app (never used there [R]). For development, a short budgeted check on face-blurred clips with separate consent |
| Keeping periods | Set and justified in `training_data_terms_v2.md` section 6 |
| Function creep | Purpose limited by the notice and `training_data_rules.md`; new purposes need new consent |

### 4.3 Legitimate interests assessment (analysis of videos of other people, and bystanders)

| Test | Assessment |
|---|---|
| Purpose | Giving the uploader (often a parent or coach) the coaching feedback they asked for. A legitimate, everyday interest. |
| Necessity | The analysis cannot be given without processing the video. No less intrusive way gives the same result. |
| Balancing | People filmed include children, whose interests weigh more (Art 6(1)(f), "in particular where the data subject is a child" (UNVERIFIED (snippet): S16)). Mitigations: private storage; no publishing; no identification; no training without the batter's or parent's permission; deletion periods; uploader asked to film only the batter. A batter filmed by a parent or their own coach would reasonably expect feedback to be produced. **Result: passes, with these measures.** |
| Right to object | Explained in the notice; on objection, delete the video unless there is a compelling reason (there will rarely be one) |

### 4.4 Information, rights and processors

- Information: full notice and child-friendly version (`training_data_terms_v2.md`), just-in-time text at upload (`consent_wording_v2.md`).
- Rights: account page tools; email route; complaints procedure (s164A); log.
- Processors: Article 28 terms with Supabase, Railway, Lovable, Google; coach agreement (M6, M8).
- Transfers: data bridge or UK Addendum per provider (M6).

### 4.5 Children's best interests (Children's code standard 1)

| Children's interest | How Batting IQ supports it |
|---|---|
| Learning and play | Gives useful technique feedback and drills |
| Safety from harm, including exposure of images | Private storage; faces blurred; no publishing; "Hide face" on by default on the share card |
| Privacy | Training off by default; parent and child both agree for 13 to 17; no profiling beyond the analysis |
| Being heard | Child-friendly text; player's own yes for 13 to 17; child can withdraw |
| Fair treatment | Score labelled beta; never used for selection; checked against coaches |
| Not being pressured | No nudges; no rewards for under-18s |

---

## Step 5. Risks

Likelihood: **remote / possible / probable**. Severity: **minimal / significant / severe**. Overall: **low / medium / high**. Assessed **before** the measures in step 6.

| # | Risk to individuals | Likelihood | Severity | Overall |
|---|---|---|---|---|
| R1 | **Unauthorised access to children's videos** (misconfigured bucket or row level security, leaked service role key, compromised admin account, lost Mac), leading to exposure or misuse of images | Possible | Severe | **High** |
| R2 | **Training use without valid consent**: an uploader agrees for someone else; a child types a made-up parent email | Probable (v1 flow) | Significant | **High** |
| R3 | **Location in video files** reveals where a child trains or lives | Probable | Significant | **High** |
| R4 | **Excessive retention**: stills kept after a video is deleted; no period for normal videos; pre-opt-in clips left on the Mac | Probable | Significant | **High** |
| R5 | **People not understanding** what happens, especially children (no full notice; Gemini not disclosed) | Probable | Significant | **High** |
| R6 | **Re-identification of face-blurred clips** by coaches or others who know the player (build, kit, club) | Possible | Significant | Medium |
| R7 | **Function creep into identification**: pose or face data later used to recognise or match people (would become special category biometric data) | Remote | Severe | Medium |
| R8 | **Outside AI provider** keeps or misuses clips | Possible | Significant | Medium |
| R9 | **Transfers** to US processors without valid safeguards | Possible | Significant | Medium |
| R10 | **Inaccurate or unfair scores** discourage a child or affect selection | Possible | Significant | Medium |
| R11 | **Share card** exposes a child's face publicly | Possible | Significant | Medium |
| R12 | **Withdrawal or deletion not carried out** in 30 days (manual steps on the Mac) | Possible | Significant | Medium |
| R13 | **Pressure to consent** (reward offer, interface nudges) | Possible | Minimal | Low |
| R14 | **Bystanders** (other children in the nets) captured without knowing | Probable | Minimal | Medium |
| R15 | **Health inference** from pose (for example "injury risk") | Remote | Significant | Low |
| R16 | **Breach not reported** or handled late, because there is no plan | Possible | Significant | Medium |

---

## Step 6. Measures to reduce risk

### 6.1 Measures

| ID | Measure | Risks | Effect | Residual risk | Approved by Alex (Y/N) |
|---|---|---|---|---|---|
| M1 | **v2 consent flow**: "Who is the batter?"; only the batter or parent or guardian can agree; 13 to 17 need their own yes and a parent's email confirmation; nothing used until confirmed; "someone else" cannot agree (`consent_wording_v2.md`) | R2, R13 | Reduced | Low | |
| M2 | **Training filter updated** in `training_data_rules.md` (relationship, confirmation, version); manifest per run | R2, R12 | Reduced | Low | |
| M3 | **Full privacy notice** plus child-friendly version, published at `/privacy` and linked at sign-up, upload and account page; Gemini disclosed | R5 | Reduced | Low | |
| M4 | **Retention**: 12-month period for normal videos; stills deleted with the video (backend change); 24-month inactive account deletion; quarterly check of training clips over 3 years | R4 | Reduced | Low | |
| M5 | **Delete pre-opt-in uploads** downloaded by `phase4_app_uploads.py`; record the date. Write the consent record for each of the 18 original clips, including commercial training and Gemini | R2, R4 | Eliminated (once done) | Low | |
| M6 | **Processor contracts and transfers**: accept or sign the Supabase, Railway, Lovable (check plan) and Google Cloud data processing terms; record region and transfer mechanism for each; keep PDFs with this DPIA | R8, R9 | Reduced | Low | |
| M7 | **Security checklist**: two-step sign-in on Supabase, Railway, Lovable, GitHub, Google and the privacy mailbox; FileVault on; service role key only in environment variables, rotated if exposed; API logs without request or response bodies; bucket stays private (re-check before launch); row level security tests from v1 Pass 1 re-run | R1 | Reduced | Medium (severity stays severe; likelihood remote) | |
| M8 | **Coach agreement**: one page for Cam and every rating coach (confidentiality, no downloads or screenshots, delete on request, report loss at once); clips coded and face-blurred | R1, R6 | Reduced | Low | |
| M9 | **Strip location metadata** on upload (backend or browser) and from any copy sent to rating tools or Gemini | R3 | Eliminated | Low | |
| M10 | **Minimise**: drop face points not used by the rules; training keeps pose points and ratings, and local video copies are deleted once no longer needed | R6, R7 | Reduced | Low | |
| M11 | **No identification rule** added to `training_data_rules.md` section 6: never build or run anything that recognises or matches a person from pose data or faces; tips about technique only, never health | R7, R15 | Reduced | Low | |
| M12 | **Gemini controls**: paid tier only; billing active and data terms accepted; only clips with box 3 ticked (or covered by the 18-clip records); face-blurred and location-stripped; never in the live app; log of each run | R8 | Reduced | Low | |
| M13 | **Score safeguards**: "beta, a guide, not a verdict" label; never shared with clubs or selectors by Batting IQ; agreement with coaches measured each round [R] | R10 | Reduced | Low | |
| M14 | **Share card**: "Hide face" on by default for everyone; under-18s see "Ask a parent or guardian before sharing a photo" | R11 | Reduced | Low | |
| M15 | **Withdrawal routine**: the two-weekly check in `training_data_rules.md` section 4, plus a 30-day calendar reminder; email withdrawals done within 3 working days | R12 | Reduced | Low | |
| M16 | **Filming guidance**: frame the batter only; avoid filming other children close up; blur when shown outside the team | R14 | Reduced | Low | |
| M17 | **Breach plan and log** (below) | R16, R1 | Reduced | Low | |
| M18 | **Rights and complaints**: form or email route, log, 30-day complaint acknowledgement, one-month replies | R5, R12 | Reduced | Low | |
| M19 | **No nudges**: v1 rules kept; reward never shown to under-18s or unknown ages | R13 | Reduced | Low | |

### 6.2 Breach plan (M17)

1. **Contain** (same day): make anything public private; rotate the Supabase service role key and any exposed secret; sign out sessions; remove the exposed copy.
2. **Assess** (within 24 hours): what data, whose (children?), how many, could it have been seen or copied, is it encrypted?
3. **Decide and report**: if a risk to people is likely, report to the ICO without undue delay and within 72 hours of becoming aware, even if details are incomplete (UNVERIFIED (snippet): S29). Exposure of children's videos to anyone outside the team: treat as likely high risk, so also tell the parents and players directly (Article 34, UNVERIFIED wording).
4. **Log every incident**, even unreported ones: date, what happened, data, people, decision, actions (UNVERIFIED (snippet): S29).
5. **Learn**: update this DPIA.

### 6.3 Residual risk summary

With M1 to M19 in place, the residual risks are **low**, except **R1 (unauthorised access)**, which stays **medium** because the harm from exposure of children's images would be severe however unlikely. A medium residual risk does not require prior consultation with the ICO; only an unmitigated **high** residual risk does (UNVERIFIED (snippet): S9).

**Without M1, M3, M5 and M9, risks R2 to R5 stay HIGH, and the training opt-in should not go live.**

---

## Step 7. Sign-off and outcomes

| Item | Name and date | Notes |
|---|---|---|
| Measures approved by | Alex Tredgett, ____________ | Integrate actions into the project plan, with dates |
| Residual risks approved by | Alex Tredgett, ____________ | If any high risk remains, consult the ICO before going ahead |
| DPO advice | Not applicable | Batting IQ does not need a DPO (no large-scale monitoring or special category data: UNVERIFIED test). Advice for this DPIA came from an AI reviewer (`dp_review.md`) |
| Summary of advice | Go ahead with the analysis service once M3, M4, M6, M7, M9 and M14 are done. Go ahead with the training opt-in only once M1, M2 and M5 are also done. Under-18 clips: only after parent email confirmation is built | |
| Advice accepted or overruled by | Alex Tredgett, ____________ | If overruled, give reasons |
| Consultation responses reviewed by | Alex Tredgett, ____________ | If departing from what people said, give reasons |
| This DPIA will be kept under review by | Alex Tredgett | Next review: [date, at most 12 months]; and before any new feature touching videos or children |

### Step 8. Alex's decisions (tick each, knowingly)

| Decision | Options | Recommendation | Decided |
|---|---|---|---|
| D1. Lawful basis for 13 to 17 account holders' analysis | Contract / legitimate interests | Contract, with paid plans bought by adults only | |
| D2. Parent or guardian for all under-18 training clips (stricter than the law's 13) | Yes / no | Yes | |
| D3. Email confirmation also for a parent's own-account upload (B3) | Yes / no | Optional; the parent is already signed in | |
| D4. Keeping period for normal videos | 12 months / other | 12 months | |
| D5. Keep using Gemini for development | Yes, with box 3 only / stop | Only with separate consent, within budget | |
| D6. Reward offer | Run / do not run | Do not run until offer terms exist | |
| D7. UK-only service | Yes / also EU | UK only | |
| D8. Accept residual medium risk R1 | Yes / no | Yes, with M7 checked | |
