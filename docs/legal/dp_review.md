Prepared by AI acting in the role of a data-protection reviewer. Not legal advice. Items marked HIGH RISK or UNVERIFIED should be decided by Alex knowingly.

# Batting IQ: data protection review of the training-consent drafts and the consent flow

Review date: 16 November 2026. Reviewer: AI, acting as a UK data-protection reviewer (UK GDPR, Data Protection Act 2018, PECR, the ICO Children's code and ICO guidance).

**What was reviewed**
- `docs/legal/training_data_terms_draft.md` (Terms of Service clause and privacy notice section, consent version `2026-11-v1`)
- `docs/legal/training_data_rules.md` (team rules for using clips)
- `docs/lovable/upload_training_consent_prompt.md` (upload screen, database, "My videos and data")
- `docs/phase4/consent_form_parent_guardian.md` (paper Forms A and B for Cam's sessions)
- `docs/goals_and_constraints.md`
- Code checked in passing: `pose_extractor.py` (MediaPipe pose on the Railway server), `api.py` (the server deletes its working folder after each job), `media_storage.py` (storyboard stills in the `battingiq-uploads` bucket under `analysis-media/`, signed links that expire after 1 hour), `phase5_video_judge_v2.py` (Gemini, paid key, face-blurred clips), `docs/lovable/results_tips_and_sharing_prompt.md` (share card with a "Hide face" switch).

**What this produces** (all in `docs/legal/`)
- this review;
- `training_data_terms_v2.md`: revised Terms of Service clause and privacy notice, with a summary of changes. Its part (a) replaces section 13.4 of `terms_of_service_v1.md` word for word (that file says data protection wording here wins);
- `consent_wording_v2.md`: revised upload-screen wording and session consent forms, including child-friendly versions;
- `dpia_batting_iq.md`: a completed DPIA for Alex to check and sign;
- `ico_fee_and_registration.md`: the ICO fee, which tier, and how to pay.

**Alex has decided not to use a solicitor.** The old drafts said "a solicitor must review this before launch". That is replaced by: Alex decides each HIGH RISK and UNVERIFIED item knowingly and writes the decision in the DPIA. Where a point is unclear, the ICO runs a free helpline and web chat for small organisations. That is a sensible substitute for a few specific questions; the current phone number is on ico.org.uk (UNVERIFIED: the number in the old drafts, 0303 123 1113, was not confirmed from a source read in this review).

---

## How to read the sources (important)

The method was: find a primary source or an official ICO page for each point, draft, then check each point against its source.

**Limitation.** From this session, **legislation.gov.uk, ico.org.uk, supabase.com, railway.com and ai.google.dev could not be reached** (their names did not resolve; the session's network does not allow them). So no primary text was read in full, and **every legal point in this pack is UNVERIFIED in the strict sense**. Points are marked as follows:

| Mark | Meaning |
|---|---|
| **[UNVERIFIED (snippet): Sx]** | Seen only in web search results: summarised extracts of the named source Sx (often the ICO or legislation.gov.uk page itself, sometimes a law firm's summary of it). The page itself could not be opened, so the exact wording was not read. Treat as UNVERIFIED, but more reliable than a plain UNVERIFIED point. |
| **UNVERIFIED** (on its own) | From the reviewer's own knowledge. No source, not even a search extract, was found in this review. Check before relying on it. |
| **[R]** | A fact taken from this repository. |

The numbered sources (S1, S2, …) are listed at the end. To upgrade an [UNVERIFIED (snippet)] or UNVERIFIED point, open the listed page and check it. It takes a few minutes each, and the most important ones are flagged in the action list.

---

## Summary: top 10 actions in priority order

| # | Action | Risk if not done | Where |
|---|---|---|---|
| 1 | **Do not switch on the training opt-in (Lovable Pass 1) with the `2026-11-v1` wording.** Change the upload flow so that only the batter (18+) or the batter's parent or guardian can say yes. A coach, friend or other uploader cannot consent for someone else. Use the `2026-12-v2` wording. | HIGH | Issue 3, `consent_wording_v2.md` |
| 2 | **Do not use any under-18 clip for training or coach rating until the parent or guardian has confirmed by email** (a "confirm" link), and, for players aged 13 to 17, the player has also said yes themselves. | HIGH | Issues 3 and 4 |
| 3 | **Pay the ICO data protection fee now** (tier 1, £52 a year, or £5 less by direct debit, [UNVERIFIED (snippet)]). Processing has already started, so this is overdue. | HIGH (legal duty; fixed penalty up to £400 for tier 1, [UNVERIFIED (snippet)]) | `ico_fee_and_registration.md` |
| 4 | **Check, complete and sign the DPIA before launch** and before any more children's clips are used. | HIGH | `dpia_batting_iq.md` |
| 5 | **Publish one full privacy notice for the whole app** (not only the training section), with a short child-friendly version, before launch. It must name Google Gemini, the retention periods and the lawful bases. | HIGH | Issue 7, `training_data_terms_v2.md` |
| 6 | **Put processor contracts in place and record them:** accept or sign the Supabase, Railway and Google Cloud data processing terms; check whether Lovable's DPA covers your plan; check each one's UK transfer safeguard and data region. | HIGH | Issues 11 and 12 |
| 7 | **Clean up old data:** delete the pre-opt-in app uploads that `phase4_app_uploads.py` downloaded; write up the consent record for each of the 18 original clips, including whether it covered commercial training and sending face-blurred clips to Google Gemini. | HIGH | Issues 3 and 10 |
| 8 | **Fix retention and minimisation gaps:** delete storyboard stills when a video is deleted; set a keeping period for videos not shared for training; strip location data from uploaded videos; turn "Hide face" on by default on the share card for everyone. | MEDIUM | Issues 8 and 9 |
| 9 | **Set up the simple processes the law expects:** a complaints procedure (a legal duty since 19 June 2026, [UNVERIFIED (snippet)]), a rights-request log, a breach plan and log, and a one-page record of processing. | MEDIUM | Issues 13 to 16 |
| 10 | **Written confidentiality and data terms for Cam and any coach who rates clips**, and a rule that no one builds any feature that recognises who someone is from pose data or faces. | MEDIUM | Issues 5 and 11 |

---

## Issue-by-issue review

Each issue gives: the finding, the risk level, what to change, and the source.

### Issue 1. Lawful basis for providing the analysis (score, tips, storyboard)

**Finding.** The draft relies on contract (UK GDPR Article 6(1)(b)) for storing and analysing an uploaded video. That is right for an adult account holder who is also the batter. It has two gaps.

1. **Children who hold their own account.** The ICO says that if you rely on contract with a child, you must consider whether the child is competent to enter the contract. If you doubt it, consider another basis such as legitimate interests. The ICO also notes there is no fixed age of contractual capacity in England, Wales and Northern Ireland, and that in Scotland children under 16 largely lack it. The Children's code (Annex C) adds that if a child's contract is voided, contract stops being a valid basis. [UNVERIFIED (snippet): S13, S14]
2. **Batters who are not the account holder.** When a parent, coach or friend uploads a video of someone else, that batter is not party to the contract. Contract can only cover the data of the person who is party to it (Article 6(1)(b) refers to a contract "to which the data subject is party": UNVERIFIED wording). The same applies to bystanders in the background.

**Risk: MEDIUM.**

**What to change.**
- Adult account holder who is the batter: **contract** (Article 6(1)(b)).
- Account holders aged 13 to 17: **contract**, provided the service they sign up for is simple and free at that level. Any paid plan for an under-18 should be bought by a parent. Alex may instead choose **legitimate interests** for all under-18 account holders, which avoids the capacity question. Decide once, before launch, and record it in the DPIA: the ICO's guidance is that you should not swap lawful basis later without good reason (UNVERIFIED).
- A batter filmed and uploaded by someone else, and any bystanders: **legitimate interests** (Article 6(1)(f)): the uploader's and Batting IQ's interest in giving the coaching feedback the uploader asked for. The balancing test must give extra weight to children: Article 6(1)(f) says the individual's interests override "in particular where the data subject is a child". [UNVERIFIED (snippet): S16] The legitimate interests assessment is in the DPIA (section 5).
- **No under-13 account holders.** Under-13s use the app through a parent's account. (Section 9 of the DPA 2018 only bites where consent is the basis, but keeping under-13s off their own accounts simplifies the Children's code duties and makes the parent responsible for the upload.) [UNVERIFIED (snippet): S3, S4]
- The privacy notice states these bases in a table (done in `training_data_terms_v2.md`).

**Sources:** S13, S14, S16; S3 and S4 for the age of 13.

### Issue 2. Lawful basis for training and testing the model: consent or legitimate interests?

**Finding.** The draft uses **consent** (Article 6(1)(a)), asked separately for each video at upload. The alternative is legitimate interests.

**Recommendation: keep consent.** Reasons:
1. **Children.** Many batters are under 18. Under legitimate interests the balancing test must give children's interests particular weight [UNVERIFIED (snippet): S16]. Commercial model training on identifiable videos of children is hard to justify as something they would reasonably expect.
2. **Reasonable expectations.** People upload a video to get coaching feedback. Few would expect it to be used to build a commercial AI model unless asked. The ICO's work on generative AI said training purposes must be explicit and specific, that reuse for training is not automatically part of the original purpose, and that "invisible processing" weighs against the controller. [UNVERIFIED (snippet): S17] (That work was about web scraping, which is a harsher case, but the reasoning carries over.)
3. **It matches the business goal.** Goal 5 in `goals_and_constraints.md` is "clips we may lawfully use commercially", by opt-in [R]. Consent also gives people a clear switch, which suits the Children's code.
4. **Rights clarity.** With consent, withdrawal is simple and absolute. With legitimate interests you would have to handle objections case by case.

**The cost of consent:** it must meet every condition below, and you must stop using a clip when consent is withdrawn.

**Conditions for valid consent** (UK GDPR Articles 4(11) and 7; ICO consent guidance):

| Condition | Draft status | Change |
|---|---|---|
| **Freely given**: not a condition of the service; no detriment for saying no [UNVERIFIED (snippet): S15] | Met: optional, unticked, analysis identical either way | Keep. Reward offer: adults only, keep it modest, and the reward is kept on withdrawal (the ICO says some incentive may be possible, [UNVERIFIED (snippet): S15]; the exact limits were not read: UNVERIFIED). |
| **Specific and granular**: separate consent for separate purposes [UNVERIFIED (snippet): S15] | Partly met: one tick covers training, testing and coach rating | Acceptable as one purpose ("improve Batting IQ's analysis"). Add a **separate** optional tick if face-blurred clips may be sent to an outside AI service (Google Gemini) during development. |
| **Informed**: who the controller is, what, why, how long, the right to withdraw [UNVERIFIED (snippet): S15] | Mostly met | Add the controller's name, outside AI services, and the link to the full notice. |
| **Unambiguous, by a clear positive action** | Met: unticked box, ticked by the user | Keep. |
| **Withdrawable as easily as it was given** (Article 7(3), [UNVERIFIED (snippet): S15]) | Met: a switch in "My videos and data" | Keep. Make sure email withdrawals are actioned within the promised time. |
| **Given by the right person**: the data subject, or for a child the holder of parental responsibility | **Not met** (see Issue 3) | Change the flow. |
| **Provable** (Article 7(1), UNVERIFIED wording) | Met: version, time, who agreed, event log | Keep, and add the new fields in Issue 3. |

**Under-13s and section 9 DPA 2018.** For an information society service (most apps and websites count) relying on consent, a child must be at least 13 to consent for themselves; under 13, consent must be given or authorised by the holder of parental responsibility, and you must make reasonable efforts to verify that, taking available technology into account. [UNVERIFIED (snippet): S3, S4] What counts as "reasonable" depends on the risk. [UNVERIFIED (snippet): S4]

**Batting IQ's choice is stricter: a parent or guardian for all under-18s.** UK law lets a 13- to 17-year-old consent alone. The stricter rule is lawful and defensible here (commercial use, video of a child, sport safeguarding norms). But the Children's code also asks you to respect a child's growing ability to decide. So for 13 to 17 the v2 flow asks for **both**: the young person's own yes, and the parent's or guardian's confirmed yes. Either can withdraw.

**A newer option, noted but not recommended now.** Since the Data (Use and Access) Act 2025 (DUAA), "scientific research" expressly includes commercial research, and consent can be given to "an area" of scientific research. [UNVERIFIED (snippet): S21] That could help later (for example for broader research use), but it is not needed for this purpose, its safeguards add work, and it does not remove the need for a lawful basis. Not recommended at this stage.

**Risk: HIGH** until the changes in Issue 3 are made; **LOW** after.

**Sources:** S3, S4, S15, S16, S17, S21.

### Issue 3. Who can give consent (the biggest gap in the current flow)

**Finding.** The `2026-11-v1` upload screen lets any account holder tick "let us use this video" if the batter is 18 or over, with the helper "If the batter isn't you, only tick this if they have agreed" [R]. Clause [X].6 has the uploader promise that the batter agreed [R].

Consent must come from the person the data is about (the data subject), or, for a child, the holder of parental responsibility (Article 4(11) definition; Article 8(1)) (UNVERIFIED wording; the parental-responsibility point is [UNVERIFIED (snippet): S4]). An uploader's promise that someone else agreed is **not** that person's consent, and Batting IQ cannot show it had their consent. The same problem arises if a young player types a made-up parent's name and email.

**Risk: HIGH.**

**What to change** (built into `consent_wording_v2.md`):
1. **Ask "Who is the batter?"** with three answers: "Me", "My child (I'm their parent or guardian)", "Someone else".
2. **"Me", account holder 18 or over:** the adult tick, as now.
3. **"Me", account holder 13 to 17:** the young-person version. The player ticks their own yes and gives a parent or guardian's email. Nothing is used until the parent or guardian clicks a confirm link in an email.
4. **"My child":** the parent or guardian ticks. For a child aged 13 to 17, they also confirm they have talked to their child and the child is happy. For a child under 13, the parent's agreement is enough, but the code still asks you to consider the child's views: the wording invites the parent to explain it to the child.
5. **"Someone else":** no training option. Show: "Only the batter, or their parent or guardian, can agree to this. They can upload the video from their own account." (A later feature could send the batter a request link. Not needed for launch.)
6. **Parent or guardian confirmation by email** for every under-18 clip, with the clip held back (`guardian_confirmed_at` empty) until confirmed. The email is a service message about the permission, not marketing.
7. Change the database: add `batter_relationship` ('self', 'own_child', 'other'), `player_agreed` (boolean, for 13 to 17), `guardian_confirmed_at` (timestamp) and `guardian_confirm_token_hash`; set `consent_version` to `2026-12-v2`. Update the rule in `training_data_rules.md`, section 3, so scripts also require `consent_given_by = 'self' or guardian_confirmed_at is not null`. (The Lovable prompt needs a Pass 1b for this; this review does not rewrite the prompt.)
8. **Existing `2026-11-v1` rows:** none should exist yet, because the opt-in has not been switched on. If any exist, use only rows with `consent_given_by = 'self'` where the account holder is the batter, or ask again.

**Sources:** S3, S4, S15.

### Issue 4. Children: which Children's code standards apply, and what the app must do

The Children's code (Age Appropriate Design Code) applies to online services "likely to be accessed by children" (under 18). Batting IQ is aimed partly at junior cricketers, so it is in scope. The code has 15 standards [UNVERIFIED (snippet): S10]. Since the DUAA, UK GDPR Article 25 (data protection by design) also requires providers of such services to take account of "children's higher protection matters", starting with how children can best be protected and supported when using the service [UNVERIFIED (snippet): S18]. Commentators expect the code to be the benchmark for that duty [UNVERIFIED (snippet): S18]. (Whether section 81 is fully in force: the main DUAA data protection changes commenced on 5 February 2026 [UNVERIFIED (snippet): S20]; the exact commencement of section 81 was not confirmed: UNVERIFIED.)

**Risk: HIGH** overall until the items below are done.

| Standard (number) | What it asks [UNVERIFIED (snippet): S10, S11, S12] | What Batting IQ must do, concretely |
|---|---|---|
| 1. Best interests of the child | Make the child's best interests a primary consideration in design. | Record in the DPIA why each feature is good for young players. Keep the score labelled "beta, a guide, not a verdict". Never show scores to clubs or selectors. Never use children's clips for marketing. Use the ICO's best-interests self-assessment once (UNVERIFIED that it is still published in the same form). |
| 2. DPIA | Do a DPIA that covers children's risks, of different ages. | `dpia_batting_iq.md`, signed before launch. Review it before each new feature that touches children's data. |
| 3. Age-appropriate application | Take a risk-based approach to recognising users' ages, or apply the code's standards to all users. | Ask date of birth at sign-up (it is already used if present [R]), and block under-13 accounts (direct them to a parent's account). For the training option, which carries the most risk, apply the child protections to anyone under 18 and to anyone whose age is unknown (the v1 flow already hides the option until the age question is answered [R]). The code treats age bands as a guide; the bands commonly used are 0–5, 6–9, 10–12, 13–15 and 16–17 (UNVERIFIED: the bands themselves were not confirmed from a source read). Batting IQ's main bands: under 13 (parent's account), 13–17, adults. |
| 4. Transparency | Clear, prominent, age-appropriate privacy information; "bite-sized" explanations at the moment data is used; prompts to talk to an adult. [UNVERIFIED (snippet): S11] | A short child-friendly privacy summary for 13–17 (in `training_data_terms_v2.md`). Just-in-time text at the upload screen for 13–17 (in `consent_wording_v2.md`), including "Talk to your parent or guardian before you say yes". |
| 5. Detrimental use of data | Do not use children's data in ways shown to be harmful. | No streaks, leaderboards or public rankings for juniors. Scores never shared with anyone except the account holder unless the user shares them. |
| 6. Policies and community standards | Uphold your own published terms. | Do what the notice says (30-day removal, face blurring, deletion). The withdrawal and deletion checks in `training_data_rules.md` section 4 are the evidence. |
| 7. Default settings | High privacy by default. [UNVERIFIED (snippet): S10] | Training opt-in off by default and never remembered (already in the v1 flow [R]). **Share card: "Hide face" on by default for every user** (today it starts on for under-18s and unknown ages, and off for adults [R]; on for everyone is the higher-privacy default and also protects children who appear in an adult's video). Videos private (bucket is private [R]). |
| 8. Data minimisation | Collect and keep only the minimum needed for each element of the service. | Yes/no age question for the training option, not a second date of birth (done [R]). Strip location metadata on upload. Keep only the pose points the rules use. Set keeping periods (Issue 8). |
| 9. Data sharing | Do not disclose children's data without a compelling reason. | No sharing with clubs, coaches outside the rating panel, or advertisers. Coaches see face-blurred clips with codes, not names [R]. |
| 10. Geolocation | Location off by default. [UNVERIFIED (snippet): S10] | The app does not ask for location. Phone videos often carry the place they were filmed in their metadata (UNVERIFIED in general terms; it depends on the phone's settings): **strip it on upload** (and from any copy sent to Gemini or rating tools). |
| 11. Parental controls | If parents can monitor a child, tell the child. [UNVERIFIED (snippet): S10] | Batting IQ has no monitoring feature. A parent who uploads a child's video sees that child's results, which is expected. If a "parent dashboard" or parent-linked junior account is ever added, the junior must see that it is on. The guardian confirmation email is not monitoring. |
| 12. Profiling | Profiling off by default unless there is a compelling reason. [UNVERIFIED (snippet): S10] | The technique score is automated evaluation of how a person moves, which is arguably profiling (UK GDPR Article 4(4) mentions "movements": UNVERIFIED wording). It is the core service the user asked for, which is the kind of compelling reason the code accepts (UNVERIFIED). **No other profiling**: no ad targeting, no recommendations based on behaviour. The old notice said "we do not profile you"; v2 says this more accurately. |
| 13. Nudge techniques | Do not nudge children to give more data or weaken privacy. [UNVERIFIED (snippet): S10] | v1 rules already ban pop-ups, highlights and remembered choices [R]. Keep the reward banner hidden from under-18s and unknown ages [R]. For 13–17 the parent form is hidden behind a plain text button [R]: keep. |
| 14. Connected toys and devices | — | Not applicable (no connected device). |
| 15. Online tools | Prominent, accessible tools to exercise rights. | "My videos and data" (withdraw, delete) [R]. Add "Download my data" and "Close my account" or a clear email route (the footer already gives one [R]). Add a "Report a concern" link. |

**Sources:** S10, S11, S12, S18, S20.

### Issue 5. Is pose data biometric (special category) data? (UK GDPR Articles 4(14) and 9)

**The test.**
- "Biometric data" (Article 4(14)) is personal data resulting from specific technical processing relating to the physical, physiological or behavioural characteristics of a person, which allow or confirm their unique identification. [UNVERIFIED (snippet): S6; the "allow or confirm" words came from an ICO report summary, not the statute itself]
- It is **special category** data (Article 9(1)) only when processed **for the purpose of uniquely identifying** a person. The ICO says this directly: biometric data is special category biometric data when you use it to uniquely identify (recognise) someone; not all biometric data is special category. [UNVERIFIED (snippet): S5]
- Photos and videos are not automatically special category or biometric data. They become biometric only when processed by specific technical means that allow unique identification (Recital 51). [UNVERIFIED (snippet): S6] The ICO also says a digital photo is not automatically biometric data. [UNVERIFIED (snippet): S6]
- Biometric data can still be special category for another reason, for example if it reveals health. [UNVERIFIED (snippet): S5]

**Applying it to Batting IQ.**
- MediaPipe pose extraction [R] is specific technical processing of physical and behavioural characteristics (body position and movement). Whether its output could "allow" identification is debatable: movement patterns can in principle single someone out, but a few seconds of a cricket drive from a phone is a weak identifier, and the system is not built or used to recognise anyone. MediaPipe's points include a few coarse face points (nose, eyes, ears, mouth corners) (UNVERIFIED count), but these are not a face template.
- **Conclusion: the pose data is personal data (it is linked to a video and an account). It may arguably fall within the wide definition of biometric data, but it is not special category data, because Batting IQ does not process it for the purpose of uniquely identifying anyone.** So Article 9 does not apply, and no Article 9 condition (such as explicit consent) is needed. [UNVERIFIED (snippet): S5]
- The videos themselves are personal data (the person is identifiable), not special category data, as long as no face recognition or other identification is run on them. [UNVERIFIED (snippet): S6]

**What would change this** (and must not happen without a new DPIA and explicit consent):
- matching a batter across videos by their body or movement ("is this the same player?");
- automatically picking out which person in a group video is the account holder by their features;
- face recognition of any kind, including in rating tools;
- inferring health (for example "injury risk" or "back problem") from pose. That could make it health data (special category) regardless of identification. Keep tips about technique only.

**Risk: LOW** with the safeguards below; **HIGH** if any feature above is added.

**What to change.**
- Keep the statement "We do not use your video or measurements to recognise who anyone is" (it is in the drafts [R]), and add it as a hard rule in `training_data_rules.md` section 6 (done in this review's recommendations; add one line: "Never build or run anything that recognises or matches a person from pose data or faces").
- Data minimisation: store only the pose points the rules use; drop face points not needed (the rules use head position, so keep the nose or head centre only).
- Training data should be keyed by upload id, not by name.

**Sources:** S5, S6.

### Issue 6. Google Gemini (paid tier) used during development

**Finding.** `phase5_video_judge_v2.py` sends face-blurred clips to the Gemini API with a paid-tier key [R]. Face-blurred clips are still personal data: the body, kit, club colours and setting can identify someone to people who know them (the v1 Form A says this itself [R]).

What the Gemini terms say (from archived versions of the terms page; not read in full) [UNVERIFIED (snippet): S27]:
- On **Paid Services**, Google does not use prompts or responses to improve its products.
- A March 2026 version adds that Google **logs prompts and responses for a limited period**, solely to detect and prevent breaches of its Prohibited Use Policy (the period was not seen: UNVERIFIED).
- Once a Cloud Billing account is active, all Gemini API use counts as a Paid Service for how Google uses the data.
- For apps offered to users in the UK, developers may use only Paid Services.
- Paid processing is covered by Google's data processing terms (UNVERIFIED which document: probably the Cloud Data Processing Addendum).

**Problems.**
1. **Transparency:** neither the Phase 4 forms nor the draft notice tell people their clip may go to Google. (The notice lists Google only as "[only if used]" [R].)
2. **Scope of consent:** for the 18 original clips, the consent people gave Alex directly must have covered this. The record required by `training_data_rules.md` section 2 should say so [R].
3. **Processor terms and transfers:** Google is a processor here. The data processing terms must be accepted on the Cloud account, and the transfer safeguard checked (Issue 12).

**Risk: MEDIUM.**

**What to change.**
- Disclose Gemini in the privacy notice and the session forms, with a **separate optional tick** on the forms (done in v2).
- Send only clips whose consent covers it; strip location metadata and blur faces first (the script already blurs [R]).
- Check in the Google Cloud console that billing is active on the project that owns the key and that the data processing terms are accepted; save a screenshot with the DPIA.
- Keep to Restriction 2: never in the live app [R].

**Sources:** S27; [R].

### Issue 7. Transparency (UK GDPR Articles 12 to 14)

**Finding.** The draft notice covers only training. There is no full privacy notice for the app [R: the Lovable prompt says "if there is no privacy page, leave the link out"]. Articles 13 and 14 require the information to be given at the time data is collected (Article 13) or within a reasonable period when collected from someone else (Article 14), for example a batter filmed by a parent or coach (UNVERIFIED article wording; standard content listed below from the reviewer's knowledge).

Gaps in the draft:
- no full notice for the analysis itself, accounts, storyboard stills, share card, emails;
- Gemini not disclosed (Issue 6);
- retention for normal videos and stills left blank [R];
- no mention of the complaints procedure now required (Issue 15);
- no child-friendly version (Children's code standard 4, [UNVERIFIED (snippet): S11]);
- "We do not profile you" is not accurate (Issue 4, standard 12);
- people filmed by someone else (Article 14) are not addressed. Where telling them directly is impossible or would take disproportionate effort, the notice on the website is the usual answer (UNVERIFIED), but the uploader should be asked to tell them, which the v2 terms do.

**Risk: HIGH** (no notice at all at launch would be a clear breach).

**What to change.** Publish the full notice in `training_data_terms_v2.md` part (b) at `/privacy`, link it from sign-up, the upload screen and the account page, and publish the child-friendly summary next to it.

Standard notice contents (UNVERIFIED as a list, from Articles 13 and 14): controller identity and contact; purposes and lawful bases; legitimate interests relied on; recipients; transfers and safeguards; retention; rights; right to withdraw consent; right to complain to the ICO; whether data is required by contract; automated decision-making; source of data if not from the person. All are covered in v2.

**Sources:** S11; Articles 12 to 14 (UNVERIFIED wording).

### Issue 8. Data minimisation and retention (Article 5(1)(c) and (e))

**Findings.**
- Storyboard stills under `analysis-media/` stay after a video is deleted [R]. The draft notice discloses this, but keeping images of a child after the parent deletes the video is hard to justify.
- No keeping period for videos not shared for training [R].
- Pre-opt-in uploads were downloaded to the Mac by `phase4_app_uploads.py` with no consent check [R]. The rules already say delete them [R]: this must actually be done and recorded.
- Video files may carry the location where they were filmed (Issue 4, standard 10).
- 3 years for training clips, and "video plus 1 year" for consent records, are reasonable for a small model-building project if justified in the DPIA. No source sets a fixed period; the test is "no longer than necessary" (Article 5(1)(e), UNVERIFIED wording).

**Risk: MEDIUM.**

**What to change.** Proposed periods (Alex to confirm; written into v2):

| Data | Proposed period |
|---|---|
| Uploaded video, not shared for training | 12 months from upload, or sooner if deleted |
| Storyboard stills and the analysis result | While the account is open; stills deleted **with** the video when the user deletes it |
| Whole account | Deleted 24 months after last sign-in, after an email warning |
| Video shared for training, and its measurements and ratings | Up to 3 years from consent, or until withdrawn or deleted (removed from training sets within 30 days) |
| Consent records (including guardian name and email) | As long as the video, plus 1 year |
| Phase 4 session clips | 24 months from the session (as Form A says) |
| Server working copies (Railway) | Deleted after each job (`api.py` already does this [R]) |
| Gemini copies | Not stored by Batting IQ; Google's abuse-logging period applies (UNVERIFIED length) |

Also: strip location metadata on upload (backend change: re-mux without metadata before storing, or strip in the browser). Keep pose points to those used.

**Sources:** Article 5 (UNVERIFIED wording); [R].

### Issue 9. Security (Article 32)

**What is already good** [R]: private bucket; row level security on `uploads`; service role key only on servers; signed links that expire after 1 hour; a database trigger that protects the consent record; a rule against WhatsApp, email and public links; encrypted devices; working copies deleted after each job.

**Gaps.**
- Multi-factor sign-in on every admin account (Supabase, Railway, Lovable, GitHub, Google Cloud, the privacy mailbox) is stated as a promise in the notice but not confirmed [R].
- The service role key is used by local scripts on the Mac [R]: rotate it if the Mac or a script log is ever exposed; never print it.
- Storyboard stills are returned inline as `data:` images [R]: make sure the API logs on Railway do not record request or response bodies.
- The share card blurs only the top quarter of the image unless face boxes are added [R]. Turn "Hide face" on by default for all users.
- Supabase backups keep deleted videos for a while (UNVERIFIED period; depends on plan). Say so in the notice ("removed from backups within [x] days").

**Risk: MEDIUM.**

**What to change.** A one-page security checklist in the DPIA (section 7), checked by Alex: MFA everywhere; FileVault on; key rotation; logs without bodies; backups period recorded.

**Sources:** Article 32 (UNVERIFIED wording); [R].

### Issue 10. The 18 original clips and the Phase 4 forms

**Findings.**
- The 18 original clips are covered only by consent given to Alex directly [R]. The rules ask for a written record of each [R]. If any consent was verbal, or did not cover commercial training, or did not cover sending blurred clips to Google, it is weak.
- The Phase 4 forms are good overall: consent basis, assent for 12 and over, coded names, 24-month deletion [R]. Gaps: no outside AI tick; no controller identity beyond "Alex Tredgett, Batting IQ"; no link to the full notice; no complaints route to Batting IQ; the session consent is on paper, so section 9 (online services) does not apply, but the same parental-consent approach is right for children.

**Risk: MEDIUM** (HIGH for any clip used beyond what its person agreed).

**What to change.** Complete the record for the 18; get a one-line written yes for anything not clearly covered. Use the v2 forms in `consent_wording_v2.md` for any new session.

### Issue 11. Processors, and Article 28 contracts

**The law.** A controller must use only processors that give sufficient guarantees, under a written contract that sets out the Article 28(3) terms: act only on documented instructions; confidentiality of staff; Article 32 security; sub-processors only with authorisation and the same terms flowed down; help with rights requests and security; delete or return data at the end; give information and allow audits. [UNVERIFIED (snippet): S23] Accepting a provider's standard data processing terms online counts as a written contract if it contains these terms (UNVERIFIED as a general rule, but standard practice).

| Processor | Role | What was found | What to do | Risk |
|---|---|---|---|---|
| **Supabase** | Database, storage, sign-in | A published DPA at supabase.com/legal/dpa; a summary says it takes effect when you accept Supabase's terms and incorporates the EU SCCs with the UK International Data Transfer Addendum. Contracting entity varies by version (Supabase, Inc. or Supabase Pte.). [UNVERIFIED (snippet): S24] | Check whether the Supabase project is in **your own** Supabase account or was created through **Lovable Cloud** (then Lovable is your processor and Supabase is Lovable's sub-processor). Record the project region (Settings, General). Download the current DPA PDF and keep it with the DPIA. | MEDIUM |
| **Railway** | Runs the FastAPI analysis server | A standard DPA at railway.com/legal/dpa referring to UK GDPR and the DPA 2018 and to the EU SCCs; can be signed through a self-service form on Railway's compliance page. A UK addendum was not confirmed. [UNVERIFIED (snippet): S25] | Sign the DPA through the self-service form. Choose the region closest to the UK if available (UNVERIFIED which regions exist) and record it. Check whether it includes the UK Addendum; if not, see Issue 12. | MEDIUM |
| **Lovable** | Builds and hosts the web app | A DPA exists; Lovable's page says it is included with Business or Enterprise plans; a signed November 2025 version names Lovable Labs Incorporated and includes SCCs and a UK addendum. A third-party review says Lovable may use data for model training unless you opt out through support. [UNVERIFIED (snippet): S26] | Check your plan. If it does not include the DPA, either upgrade, or confirm that no personal data passes through Lovable's own systems (the app's data lives in Supabase). Ask Lovable support to confirm training opt-out for your project. | MEDIUM |
| **Google (Gemini API)** | One-off development checks | Paid-tier terms: no training on your data; limited abuse logging; data processing terms for paid services. [UNVERIFIED (snippet): S27] | Accept the data processing terms on the Cloud account. Record the date of each development run and which clips went. | MEDIUM |
| **Cam and rating coaches** | Rate clips for Batting IQ | No written agreement found [R]; the notice promises they "agree in writing to keep them confidential" [R]. | A one-page coach agreement: use clips only to rate for Batting IQ; keep them confidential; no copies, downloads or screenshots; delete on request; report any loss at once. (They act under Batting IQ's authority; for a self-employed coach, a short Article 28-style agreement is safest: UNVERIFIED whether they are "processors" or persons acting under the controller's authority under Article 29.) | MEDIUM |
| **Email provider** (for guardian confirmations and the privacy inbox) | Sends and receives email | Not yet chosen [R] | Use a provider with standard data processing terms; record it. | LOW |

**Sources:** S23 to S27.

### Issue 12. International transfers (UK GDPR Chapter V)

**The options.**
- **UK-US data bridge** (the UK Extension to the EU-US Data Privacy Framework): in force since 12 October 2023; works only if the US recipient has self-certified under the DPF **and** the UK Extension. [UNVERIFIED (snippet): S28] Its status in November 2026 was not confirmed (UNVERIFIED; commentators flagged that it depends on the EU framework surviving legal challenge [UNVERIFIED (snippet): S28]).
- **UK International Data Transfer Agreement (IDTA)** or the **UK Addendum** to the EU Standard Contractual Clauses, usually built into a provider's DPA. [UNVERIFIED (snippet): S24, S26]
- **Adequacy**: transfers to a country the UK has found adequate (for example the EEA) need no extra safeguard (UNVERIFIED list).
- A transfer risk assessment is expected when relying on the IDTA or Addendum (UNVERIFIED; the DUAA changed the test to "not materially lower" protection: UNVERIFIED detail).

| Provider | Likely mechanism | Check |
|---|---|---|
| Supabase | UK Addendum in its DPA [UNVERIFIED (snippet): S24]; data bridge if certified (not confirmed) | Region of the project; DPA version; dataprivacyframework.gov list |
| Railway | EU SCCs in its DPA [UNVERIFIED (snippet): S25]; UK Addendum not confirmed | DPA text; DPF list |
| Lovable | SCCs and UK addendum in its DPA [UNVERIFIED (snippet): S26] | Plan; DPA |
| Google | Data bridge or the UK Addendum in Google's terms (UNVERIFIED) | Google Cloud data processing terms |

**Risk: MEDIUM.**

**What to change.** For each provider, write one line in the record of processing: country, mechanism, date checked, link. The v2 notice says this in plain words.

**Sources:** S24 to S28.

### Issue 13. Controller obligations: record of processing, DPO, EU users

**Record of processing (Article 30).** Organisations with fewer than 250 staff need only document processing that is not occasional, **or** could result in a risk to people's rights and freedoms, **or** involves special category or criminal offence data. [UNVERIFIED (snippet): S22] No sign that the DUAA changed this for the UK. [UNVERIFIED (snippet): S22] Batting IQ's uploads are regular (not occasional) and involve children, so **the exemption does not apply: keep a record.** A short table is enough; one is at the end of this review (Annex A).

**Data protection officer (Article 37).** Not required: Batting IQ is not a public authority, and its core activities are not large-scale regular and systematic monitoring or large-scale special category data (UNVERIFIED wording of the test). Name Alex as the privacy contact instead.

**EU users.** If the app is offered to people in the EU (for example Ireland), the EU GDPR may also apply and an EU representative may be needed (EU GDPR Articles 3(2) and 27: UNVERIFIED). **Simplest fix:** say the service is for people in the UK, and do not market it elsewhere. **Risk: LOW.**

**ICO fee.** Batting IQ must pay. Tier 1 (micro organisation: turnover up to £632,000 or no more than 10 staff), **£52 a year** since 17 February 2025, with a £5 discount by direct debit; no exemption applies. [UNVERIFIED (snippet): S1, S30] Full steps in `ico_fee_and_registration.md`. **Risk: HIGH until paid** (simple fix).

**Sources:** S1, S22, S30.

### Issue 14. DPIA requirement (Article 35 and the ICO's list)

**The law.** A DPIA is required where processing is likely to result in high risk. Article 35(3) gives three examples, and the ICO's Article 35(4) list adds more. [UNVERIFIED (snippet): S7, S9] The ICO's list requires a DPIA for **innovative technology combined with another criterion** (for example evaluation or scoring), and the ICO says innovative technology includes new applications of existing technology, including AI. [UNVERIFIED (snippet): S7] Commentary on the list says it also covers targeting children and use of biometric data relating to children. [UNVERIFIED (snippet): S7] The Children's code says online services in its scope inevitably involve processing likely to result in high risk to children, so a DPIA is expected (standard 2). [UNVERIFIED (snippet): S8]

**Applying it.** Batting IQ: children's data, video of identifiable people, AI-based automated scoring (evaluation or scoring), new use of pose estimation, commercial model training. **A DPIA is required.** If a high risk remains after the measures, the ICO must be consulted before processing (Article 36). [UNVERIFIED (snippet): S9] The DPIA in this pack concludes the residual risk is medium at worst once the measures are in, so prior consultation should not be needed, provided Alex completes the HIGH actions.

**Risk: HIGH** until signed.

**Sources:** S7, S8, S9.

### Issue 15. Rights handling and complaints

**Rights.** Access, rectification, erasure, restriction, objection (for legitimate interests processing), portability (for consent and contract processing), and withdrawal of consent (UNVERIFIED article-by-article detail). Reply within one month (UNVERIFIED: Article 12(3); extendable for complex cases). Since 5 February 2026, a subject access search need only be reasonable and proportionate, and the clock can be paused while you wait for clarification you reasonably need. [UNVERIFIED (snippet): S20]

**Complaints (new duty).** Since **19 June 2026**, every controller must have a way for people to complain about its handling of their data (new section 164A DPA 2018): acknowledge a complaint within **30 days**, take appropriate steps to respond, and make it easy to complain, for example with an electronic form. [UNVERIFIED (snippet): S19] The ICO can decline to act on a complaint not first put to the controller. [UNVERIFIED (snippet): S19]

**Children's rights.** A child can exercise their own rights if competent. The old draft's "from about age 12" is a common rule of thumb in Scotland's law and ICO guidance (UNVERIFIED). Keep it as guidance, not a rule.

**Risk: MEDIUM.**

**What to change.**
- Add a "Privacy request or complaint" form or email route on `/privacy` and the account page.
- Keep a simple log (date received, what asked, identity check, date acknowledged, date done). For complaints, acknowledge within 30 days (aim for 3 working days).
- Withdrawal by email: set `training_consent = false` within 3 working days, as the rules say [R].
- Erasure: delete the video, stills, measurements, ratings and local copies (the rules' section 4 [R], plus stills).
- Access: a copy of the videos, results and consent records.

**Sources:** S19, S20.

### Issue 16. Breach process (Articles 33 and 34)

**The law.** Report a personal data breach to the ICO without undue delay and, where feasible, within **72 hours** of becoming aware of it, unless it is unlikely to result in a risk to people. The clock starts when you become aware, not when the investigation ends. You can report in stages and must explain any delay. Record **all** breaches, including ones you do not report. [UNVERIFIED (snippet): S29] Tell the people affected without undue delay if the breach is likely to result in a **high** risk (Article 34, UNVERIFIED wording). Processors must tell you without undue delay (Article 33(2), UNVERIFIED wording).

**Risk: MEDIUM** (no plan exists [R]).

**What to change.** A breach plan (in the DPIA, section 7):
1. Contain: revoke keys, make anything public private, rotate the service role key, sign out sessions.
2. Assess within 24 hours: what data, whose (children?), how many, could it be seen or copied?
3. Children's videos exposed to anyone outside the team: treat as likely high risk. Report to the ICO within 72 hours (online form; the ICO has a 72-hour guide for small organisations [UNVERIFIED (snippet): S29]) and tell the parents and players.
4. Log every incident, even small ones (for example a clip shown to the wrong coach).

**Sources:** S29.

### Issue 17. Smaller points

| Point | Finding | Change | Risk |
|---|---|---|---|
| Consent record basis | The draft relies on "legal obligation" (Article 6(1)(c) with 7(1)) to keep consent records. Debatable whether Article 7(1) is a legal obligation in the Article 6(1)(c) sense (UNVERIFIED). | v2 uses legitimate interests (being able to prove consent and defend claims). Either is defensible; legitimate interests avoids the argument. | LOW |
| Automated decisions | The score and tips are automated but have no legal or similarly significant effect. The DUAA rewrote the rules on automated decisions (new Articles 22A to 22D: UNVERIFIED detail). | Keep "a guide, not a verdict"; never let a score be used by clubs for selection. Notice says so. | LOW |
| Bystanders | Other players or children in the nets can appear in videos. | Filming tips: frame the batter; v2 terms ask uploaders to avoid filming others close up; blur when shown outside the team. | LOW |
| Reward offer | Adults only, kept on withdrawal [R]. | Keep it modest and off by default (it is [R]). Never show it to under-18s or unknown ages (already the rule [R]). | LOW |
| PECR | Guardian confirmation emails are service messages, not marketing. Any marketing email needs consent or the "soft opt-in" for customers (PECR regulation 22: UNVERIFIED). Cookies or analytics in the Lovable app need consent unless strictly necessary or within the DUAA analytics exemption (PECR regulation 6 as amended: UNVERIFIED detail). | Check the Lovable app for analytics scripts. If any, add a cookie banner or remove them. | LOW |
| "Training can't be undone" | Accurate for a model trained on pose features: the model does not hold copies of the video. | v2 keeps it, says "does not contain a copy of your video". | LOW |
| Turning 18 | No legal rule found that a parent's consent lapses at 18 (UNVERIFIED). | v2: at 3 years the clip is deleted anyway; the player can withdraw at any time. Optional: email a reminder at 18 if the date of birth is known. | LOW |

---

## Annex A. Record of processing (Article 30), short form

Controller: [Alex Tredgett, trading as Batting IQ / Batting IQ Ltd], [address], [privacy email]. ICO registration: [number]. No DPO required; privacy contact: Alex.

| Activity | People | Data | Purpose | Lawful basis | Recipients and processors | Transfers | Keeping period | Security |
|---|---|---|---|---|---|---|---|---|
| Accounts and sign-in | Users (13+) | Name, email, date of birth, password hash | Run accounts | Contract | Supabase; Lovable | See Issue 12 | Account life + 24 months inactive | MFA for admins; RLS |
| Video analysis | Batters, bystanders | Video, pose points, scores, tips, stills | Give the analysis | Contract (account holder); legitimate interests (others) | Supabase; Railway | See Issue 12 | 12 months (video); account life (stills, results) | Private bucket; signed links; working copies deleted |
| Training and testing | Batters who agreed (and parents) | Video, pose points, coach ratings, consent record, guardian name and email | Improve the analysis | Consent | Supabase; Railway; coaches (blurred); Google Gemini (dev only, if ticked) | See Issue 12 | 3 years / withdrawal (+30 days) | As above; encrypted Mac; manifests |
| Consent records | As above | Version, time, who agreed, guardian details, events | Prove consent | Legitimate interests | Supabase | See Issue 12 | Video + 1 year | Trigger-protected log |
| Phase 4 sessions | Players, parents | Video, code, forms, contact | Test and improve | Consent | Cam; coaches; storage | — | 24 months | Paper forms locked away; codes |
| Rights requests and complaints | Requesters | Request details | Handle requests | Legal obligation | Email provider | — | 2 years after closing (proposal) | Restricted mailbox |
| Breach log | Affected people | Incident details | Accountability | Legal obligation | — | — | 3 years (proposal) | — |

---

## Sources

All marked [UNVERIFIED (snippet)]: seen only as summarised web search results, because the primary sites were blocked in this session. URLs are for Alex to open and check.

| ID | Source | URL | What it supports |
|---|---|---|---|
| S1 | ICO data protection fee page; Lewis Silkin, "Fee payable to the ICO to increase in February" (2025); gov.uk consultation outcome | https://ico.org.uk/for-organisations/data-protection-fee/ ; https://www.lewissilkin.com/insights/2025/02/06/fee-payable-to-the-ico-to-increase-in-february-102jzb9 ; https://www.gov.uk/government/consultations/data-protection-fee-regime-proposed-changes/outcome/data-protection-fee-regime-government-response | Tiers, £52/£78/£3,763 from 17 Feb 2025; tier 1 criteria |
| S2 | ICO, Fixed penalties for failure to pay the data protection charge | https://ico.org.uk/about-the-ico/our-information/fixed-penalties-for-failure-to-pay-the-data-protection-charge/ | £400 / £600 / £4,000 penalties; up to £4,350 with aggravating factors (secondary) |
| S3 | DPA 2018 s9 | https://www.legislation.gov.uk/ukpga/2018/12/section/9 | Age 13 for ISS consent; preventive or counselling services excluded |
| S4 | ICO, What are the rules about an ISS and consent? | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/children-and-the-uk-gdpr/what-are-the-rules-about-an-iss-and-consent | Parental responsibility under 13; reasonable efforts to verify; risk-based |
| S5 | ICO, Biometric data guidance: key data protection concepts | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/guidance-on-biometric-data/key-data-protection-concepts | Special category only when used to uniquely identify |
| S6 | UK GDPR Article 4 and Recital 51; ICO, What is special category data? | https://www.legislation.gov.uk/eur/2016/679/article/4 ; https://www.mishcon.com/uk-gdpr/recital/no-51 ; https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/what-is-special-category-data | Biometric definition; photographs |
| S7 | ICO, Examples of processing likely to result in high risk | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/data-protection-impact-assessments-dpias/examples-of-processing-likely-to-result-in-high-risk | Innovative technology plus another criterion |
| S8 | ICO Children's code, standard 2 (DPIAs) | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/age-appropriate-design-a-code-of-practice-for-online-services/2-data-protection-impact-assessments/ | DPIA expected for in-scope services |
| S9 | ICO, How do we do a DPIA? | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/data-protection-impact-assessments-dpias/how-do-we-do-a-dpia | DPIA steps; prior consultation |
| S10 | ICO, Age appropriate design code (v2.1 PDF); FAQs on the 15 standards | https://ico.org.uk/media/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/age-appropriate-design-a-code-of-practice-for-online-services-2-1.pdf ; https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/faqs-on-the-15-standards-of-the-children-s-code/ | The 15 standards |
| S11 | ICO Children's code, standard 4 (transparency) | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/age-appropriate-design-a-code-of-practice-for-online-services/4-transparency/ | Bite-sized, just-in-time, talk to an adult |
| S12 | ICO Children's code, standard 3 (age appropriate application) | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/age-appropriate-design-a-code-of-practice-for-online-services/3-age-appropriate-application/ | Age ranges as a guide |
| S13 | ICO Children's code, Annex C (lawful basis) | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/age-appropriate-design-a-code-of-practice-for-online-services/annex-c-lawful-basis-for-processing | Contract and child's capacity |
| S14 | ICO, What do we need to consider when choosing a basis for processing children's personal data? | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/children-and-the-uk-gdpr/what-do-we-need-to-consider-when-choosing-a-basis-for-processing-children-s-personal-data/ | Competence; alternative basis |
| S15 | ICO consent guidance (what is valid consent; what's new) | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/consent/what-is-valid-consent ; https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/consent/whats-new | Freely given, granular, withdrawal, incentives |
| S16 | ICO legitimate interests guidance | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/a-guide-to-lawful-basis/legitimate-interests/ | Three-part test; extra care for children |
| S17 | ICO, Response to the consultation series on generative AI (Dec 2024), and law-firm summaries (Skadden, HCR) | https://ico.org.uk/about-the-ico/what-we-do/our-work-on-artificial-intelligence/response-to-the-consultation-series-on-generative-ai/ ; https://www.skadden.com/insights/publications/2025/01/ico-publishes-outcomes-of-genai-consultation | Specific purposes; necessity; invisible processing |
| S18 | DUAA 2025 s81 (Article 25 children's higher protection matters); ICO data protection by design and default | https://www.legislation.gov.uk/ukpga/2025/18/section/81 ; https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/guide-to-accountability-and-governance/data-protection-by-design-and-default | Children's higher protection matters |
| S19 | CMS and DLA Piper on DUAA complaints (s164A DPA 2018, from 19 June 2026, SI 2026/82) | https://cms.law/en/gbr/legal-updates/data-use-and-access-act-2025-new-statutory-rules-on-handling-data-protection-complaints-from-19th-june-2026 ; https://privacymatters.dlapiper.com/2026/06/uk-new-complaints-handling-rules-under-duaa-take-effect-on-19-june-2026-are-you-ready/ | Complaints duty |
| S20 | DLA Piper, Commencement of the DUAA data protection provisions (Feb 2026); ICO, A guide to subject access | https://privacymatters.dlapiper.com/2026/02/uk-commencement-of-the-data-protection-provisions-in-the-data-use-and-access-act/ ; https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/subject-access-requests/a-guide-to-subject-access/ | 5 Feb 2026 commencement; reasonable and proportionate searches; pausing the clock |
| S21 | Stephenson Harwood, DUAA key changes to scientific research | https://www.stephensonharwood.com/insights/duaa-2025-key-changes-to-scientific-research-in-uk-data-law/ | Commercial research; consent to an area of research |
| S22 | ICO, Who needs to document their processing activities? | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/documentation/who-needs-to-document-their-processing-activities/ | Article 30(5) limits |
| S23 | ICO, What needs to be included in the contract? | https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/contracts-and-liabilities-between-controllers-and-processors-multi/what-needs-to-be-included-in-the-contract/ | Article 28(3) terms |
| S24 | Supabase DPA (and third-party summary) | https://supabase.com/legal/dpa | SCCs and UK Addendum; acceptance with terms |
| S25 | Railway DPA and compliance docs | https://railway.com/legal/dpa ; https://docs.railway.com/maturity/compliance | DPA; self-service signing |
| S26 | Lovable DPA; Waimakers review (June 2026) | https://lovable.dev/data-processing-agreement ; https://www.waimakers.com/en/resources/ai-data-security/lovable | Plan-dependent DPA; UK addendum; training opt-out |
| S27 | Gemini API Additional Terms of Service (current and archived versions) | https://ai.google.dev/gemini-api/terms ; https://ai.google.dev/gemini-api/terms-archive/terms_12_18_25 | Paid-tier data use; abuse logging; UK paid-only |
| S28 | IAPP, "UK-U.S. data bridge becomes law, takes effect 12 October" and other 2023 commentary | https://iapp.org/news/a/uk-u-s-data-bridge-becomes-law-takes-effect-12-october | Data bridge mechanics |
| S29 | ICO, 72 hours: how to respond to a personal data breach; ICO report a breach | https://ico.org.uk/for-organisations/sme-web-hub/72-hours-how-to-respond-to-a-personal-data-breach ; https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/ | 72 hours; log all breaches |
| S30 | Secondary guides on fee exemptions and direct debit discount (Start Up Loans; Briffa) | https://www.startuploans.co.uk/business-guidance/what-is-the-ico-data-protection-fee-and-do-i-need-to-pay-it ; https://www.briffa.com/blog/ico-fees-and-gdpr-fines-for-small-businesses-and-sole-traders/ | Exemptions; £5 direct debit discount |

**Self-review.** Each point above was checked against its listed source as far as the search extracts allowed. Points where the extract did not show the exact wording, or where no source was found, are marked UNVERIFIED. The most important ones to check by opening the page: the current ICO fee and direct debit discount (S1); the ICO biometric guidance wording (S5); the Supabase, Railway and Lovable DPAs and their UK transfer terms (S24 to S26); the current Gemini terms (S27); and the data bridge status in 2026 (S28).
