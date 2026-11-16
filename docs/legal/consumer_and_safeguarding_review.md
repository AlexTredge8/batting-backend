# Batting IQ: consumer-law, advertising and safeguarding review (v1, 16 Nov 2026)

> **Prepared by AI acting in the role of a consumer-law and safeguarding reviewer. Not legal advice. Items marked HIGH RISK or UNVERIFIED should be decided by Alex knowingly.**

**What this covers.** The licence users grant over their videos, the "free month for sharing clips" incentive, subscription terms for the planned prices, AI coaching tips and drill safety, marketing claims, the share card, safeguarding for coach review and Cam's filming session, and ambassadors. Two companion drafts sit next to this file: `terms_of_service_v1.md` and `safeguarding_policy_v1.md`.

**What it does not cover.** Data protection (lawful basis, validity of consent, DPIA, ICO fee). Another reviewer is writing that in `dp_review.md`, `training_data_terms_v2.md`, `consent_wording_v2.md`, `dpia_batting_iq.md` and `ico_fee_and_registration.md`. Where an issue has a data protection side, this review says so and defers to those files. If they disagree with this file on a data protection point, theirs wins.

**Files read.** `docs/goals_and_constraints.md`, `docs/legal/training_data_terms_draft.md`, `docs/legal/training_data_rules.md`, `docs/lovable/upload_training_consent_prompt.md` (including the incentive banner), `docs/phase4/consent_form_parent_guardian.md`, `docs/phase4/filming_guide_for_cam.md`, `docs/gtm/go_to_market_research.md`, `docs/lovable/results_tips_and_sharing_prompt.md` (share card and drill safety line).

## How the sources were checked (read this before relying on anything)

- **Page fetches were blocked.** This session could not open legislation.gov.uk or gov.uk ("host not allowed"). It could run web searches. So every "Checked" point below was checked against **search-result extracts** of the named source (the statute, the regulator's page, or a law-firm summary of it), **not** against the full text. Section numbers and quotations should be confirmed on legislation.gov.uk before anything is published.
- **Labels used:**
  - **Checked (official extract):** seen in an extract from legislation.gov.uk, ico.org.uk, asa.org.uk, ecb.co.uk or nspcc.org.uk.
  - **Checked (secondary):** seen only in a reputable law firm's or club's summary of the source.
  - **UNVERIFIED:** from the reviewer's own knowledge; not seen in any source in this session.
- **Risk ratings:** HIGH (likely breach, enforcement or harm to a child if launched as is; decide before launch), MEDIUM (fix before paid launch or before the club channel), LOW (tidy up).

---

## Summary table

| # | Issue | Risk | One-line fix |
|---|---|---|---|
| 1 | Licence over uploaded videos | MEDIUM | Mostly fair. Add service providers, coach-you-choose, editing acts (blur, crop, frames), share card; no indemnity from consumers; plain summary at upload. |
| 2 | Incentive: free Premium month for 3 training clips | **HIGH** (if switched on as drafted) | Keep OFF until written offer terms exist, the month does not roll into paid, the DP reviewer agrees, and the 18+ gate is enforced. Reword "Share". |
| 3 | Subscription terms and checkout | **HIGH** (at paid launch) | Pre-contract information, express request and acknowledgement, confirmation email, 14-day cancellation, online cancel button, reminders. DMCC subscription rules expected from 1 Jan 2027. |
| 4 | AI coaching tips and drills: liability and safety | MEDIUM | Fair disclaimer (not "at your own risk"), carve-out for death or personal injury caused by negligence, Cam signs off drills, stronger junior helmet line. |
| 5 | Marketing claims ("as good as a coach") | **HIGH** (if used) | Do not use. Use evidenced claims only, with a claims file. |
| 6 | Share card (juniors) | **HIGH** | Under-18 and unknown age: no name, face blur locked on, no referral code, check bystanders; check the blur really covers the face. |
| 7 | Coach and club review of juniors' videos | **HIGH** (before building it) | Parent approval per coach link, club confirms coaches are vetted, no private messaging with under-18s, access log. Online Safety Act scope check first. |
| 8 | Cam's filming session | **HIGH** (until done) | Club approval and welfare officer sign-off, a second adult present, Cam's vetting current, phone cloud sync off, names list on paper. |
| 9 | Ambassadors and influencers | MEDIUM | "Ad" label on every post, written agreement, approved claims list, no under-18 ambassadors. |
| 10 | Housekeeping: company details, complaints, governing law, minors' contracts | MEDIUM | Fill in placeholders; adults buy paid plans; complaints route; England and Wales law without removing Scottish or Northern Irish consumer rights. |

---

## Issue 1. The licence users grant over their uploaded videos

**Current wording.** `training_data_terms_draft.md` clause [X]: analysis use always (X.1); optional, unticked, per-video permission to "copy, store and analyse" and "make measurements" to (a) train, test and improve, (b) let qualified coaches who work with us watch and rate, (c) check agreement (X.2); no sale, licensing to others, publication, marketing or identification (X.3); withdrawal at any time, removal from training and test sets within 30 days, "training already done can't be undone" (X.4); under-18s need a parent or guardian (X.5); user promises (X.6); "you keep any rights you have" (X.7); 3-year keeping period (X.8).

**Assessment: MEDIUM. It is clear, narrow and mostly fair. Six gaps.**

| Gap | Why it matters | Change | Source |
|---|---|---|---|
| 1a. No licence term covering the **service providers** that store and process videos (Supabase, Railway, Lovable). | Strictly, copying by a host is our copying. Users should not be surprised. | Add: "and to let the companies that host and run Batting IQ for us do these things on our behalf, under contract." | Transparency requirement, CRA 2015 s.68 (UNVERIFIED wording). |
| 1b. **Coach review chosen by the user** (Coach and Club tiers) is not covered. X.2(b) covers only coaches who work with Batting IQ. | A player who links to their own club coach is letting a third party view the video. That needs its own permission, separate from training. | Add a separate licence: "If you choose to link your account to a coach or club, you let us show your videos and results to that coach or club until you unlink." Under-18s: parent or guardian approves each link (see Issue 7). | CRA 2015 s.62 fairness and s.68 transparency (UNVERIFIED wording); CMA37 (revised 22 Jul 2026, Checked (secondary)). |
| 1c. **Editing acts** are not mentioned: blurring faces, cropping, slowing down, extracting still frames for the storyboard and share card. | The person who made the film has moral rights, including the right to object to derogatory treatment. Blurring is very unlikely to be "derogatory", but saying so removes doubt. | Add: "We may blur, crop, slow down or take still frames from your video to give you the service and to protect privacy." Do **not** ask consumers to waive moral rights; describing the acts is enough. | CDPA 1988 ss.77–80 (moral rights), s.87 (consent and waiver): UNVERIFIED. |
| 1d. **Share card** creation not covered. | The card is made in the user's browser from their own frame, and we never post it. Saying so supports the "no publication" promise. | State that the card is made for the user, at their request, and that we do not post it anywhere. | Consistency with X.3. |
| 1e. Who owns the film. | Copyright in a phone video belongs to its **author: the producer and principal director**, which for a phone clip is usually whoever filmed it, often a parent, friend or coach, **not the batter**. So the user may not own what they upload. X.6 already asks the user to confirm they have the right to share. | Keep X.6. Do **not** add an indemnity from consumers ("you will pay our losses if…"); it is likely unfair against a consumer and adds nothing in practice. Say instead that we may remove a video if we are told the filmer or batter did not agree (X.10 already does). | CDPA 1988 s.9 and s.10 (film authorship, joint authorship): Checked (secondary: case summary and commentary). s.5B (definition of film): Checked (secondary). Unfairness of consumer indemnities: CRA 2015 Sch 2 (indicative list), UNVERIFIED as to the specific paragraph. |
| 1f. **Performers' rights and image rights.** | Performers' rights cover dramatic and musical performances, readings and variety acts. A net session is almost certainly not a "performance", and English law has no general image right. So nothing extra is needed, but a catch-all costs nothing. | Optional: "any rights you or the batter have in the video" (already in spirit in X.7). | CDPA 1988 s.180(2): UNVERIFIED. Sporting events are not copyright works (FAPL v QC Leisure / Murphy, CJEU): Checked (secondary). |

**Is the training licence fair and transparent?**
- **Fair, if kept as drafted.** It is optional, unticked, per-video, narrow in purpose, revocable, and saying no changes nothing. That is the opposite of the "wide, irrevocable, worldwide licence to use your content for any purpose" that the CMA has historically criticised in consumer terms (UNVERIFIED as to specific CMA37 passages; the revised CMA37 adds a stand-alone transparency principle and says digital content licences must be fair even where statutory rights do not apply: Checked (secondary)).
- **"Training already done can't be undone"** is fair because it is said clearly up front and the model does not contain the video. Keep it next to the tick box, not only in the terms.
- **It is not a core term** (not the price or the main subject matter), so it can be assessed for fairness whatever its prominence (CRA 2015 s.64: UNVERIFIED; the search confirmed only that s.64 deals with core-term exclusions).
- **Royalty-free is fine** for an optional gift of data, provided the incentive (Issue 2) is described honestly.
- **Data protection side:** whether consent here is valid, and the retention periods, are for `dp_review.md` and `training_data_terms_v2.md`.

**What to change.** The licence clause in `terms_of_service_v1.md` (section 13) already includes 1a to 1d. Bring `training_data_terms_v2.md` (the DP reviewer's file) in line, or point it at the ToS clause, so there is one wording.

---

## Issue 2. The incentive: "Share 3 drives for training and get a month of Premium free"

**Current design.** `upload_training_consent_prompt.md` step 7: a slim banner, feature flag `trainingIncentiveBanner: false` (OFF until Alex decides), shown only to profiles aged 18+, never on the account page, never ticks anything. Small print: "Optional, for players aged 18 and over. Your analysis is the same either way, and you keep the reward if you change your mind later." Reward given by hand. Draft terms X.9 match.

**Assessment: HIGH RISK if switched on as drafted, because the offer has no written terms and the end of the free month is undefined. Otherwise the design is sound.**

**2a. Data protection angle (coordinate with `dp_review.md`).** Consent must be freely given. The ICO says you can incentivise consent to some extent, but must not unfairly penalise people who refuse or withdraw (Checked (official extract and secondary): ICO "What is valid consent?" and ICO consent-or-pay guidance). The design already does the three things that matter:
- refusing costs nothing (the analysis is the same);
- withdrawing does not claw back the reward (if it did, withdrawal would carry a penalty);
- under-18s never see it (ICO Children's code standard 13, nudge techniques: Checked (official extract)).

The final call on whether consent stays "freely given" belongs to the DP reviewer. This review assumes they agree, subject to 2b to 2e.

**2b. It is a sales promotion, so the CAP Code applies (Section 8).** The banner must state, or link one tap away to, the **significant conditions**: how to take part, the closing date, the nature of the reward, eligibility and any limits (CAP rule 8.17: Checked (official extract)). Promotions aimed at children are likely to need a closing date (Checked (official extract)); ours is not aimed at children, but a closing date is still expected in most cases.

Write short offer terms covering:
- **Who:** UK residents aged 18 or over, one reward per person.
- **What counts:** 3 separate front-foot drive videos, each with "Help improve Batting IQ" ticked at upload, each showing the account holder or another adult who agreed. Say whether a clip we later reject (unusable or not a drive) still counts. **Recommendation: it counts if it was a genuine upload of a drive,** so that people are not judged on quality.
- **The reward:** one calendar month of Premium, given within [7] days, by hand.
- **What happens at the end of the free month (the key gap):** it must **end without charge** and must not roll into a paid plan. If it did roll over, it would be a free trial that converts into a subscription. That would bring in the subscription reminder rules (Issue 3) and the CAP "free" rule, which bars calling something free if the consumer has to pay anything beyond the unavoidable costs of responding (CAP rule 3.23: Checked (official extract)). If the user already pays for Premium, add a month to their paid period instead.
- **Withdrawal:** you keep the reward if you later withdraw (as now).
- **Closing date:** [date].
- **Promoter:** [company details].

**2c. Wording.** "Share 3 drives" can be read as "post publicly". Change to: **"Let us use 3 of your drives to train Batting IQ and get a month of Premium free."** Small print as now, plus "Offer terms" link.

**2d. Gaming and other people's clips.** A reward invites uploading other people's videos to reach 3. The X.6 promises cover it legally, but add a line to the offer terms: "Each video must show you, or an adult who has agreed. Videos of under-18s never count towards the offer."

**2e. Age gate.** Age comes from the profile date of birth or a yes/no question. That is self-declared. If a reward claimant turns out to be under 18, give no reward and remove the clips from training unless a parent or guardian agrees. CAP treats under-16s as children for Section 5 and incentivised peer-to-peer activity by children falls within the Code (Checked (official extract)). Our stricter line (18) is right.

**Decision for Alex (HIGH RISK, decide knowingly):** whether to run the offer at all before the DP reviewer has signed off. Recommendation: keep the flag OFF until the offer terms exist and `dp_review.md` agrees.

---

## Issue 3. Subscription terms for the planned prices

**Planned (from `go_to_market_research.md`):** Free (3 analyses a month); Premium £6.99 a month or £39.99 a year, annual as the default; Season Pass £24.99 (April to September); a 14 to 30-day trial; Coach tier about £14.99 a month for 25 players; club pack £249 to £399 a year; referral code on share cards.

**Assessment: HIGH RISK at paid launch unless the checkout and cancellation flow is built to the rules below. Nothing is charged yet, so nothing is in breach today.**

**3a. Rules in force now (Consumer Contracts Regulations 2013, "CCRs").**
- **Before the order:** give the information in Schedule 2: who we are and our address, the total price including VAT, how often it is charged, that it renews automatically, the minimum term, how to cancel, the 14-day cancellation right and when it is lost, functionality and compatibility of the digital content (CCRs reg 13 and Sch 2: Checked (official extract) for Sch 2 items (n) and (o); the rest UNVERIFIED). The key facts must be shown **directly before** the pay button, and the button must say something like "Pay now" or "Subscribe and pay" (CCRs reg 14: UNVERIFIED).
- **After the order:** confirm the contract on a durable medium, normally an email, including the cancellation information and the user's express request (CCRs reg 16: Checked (official extract)).
- **Cancellation period:** 14 days from the day the contract is made (CCRs regs 29 and 30: Checked (official extract) that the period starts when the contract is entered into).
- **Is Premium "digital content" or a "service"?** Probably a mix: the app is digital content; the analysis on our server looks like a service. The rules differ:
  - **Digital content:** the user loses the right to cancel once supply starts, but only if they gave **express consent** and **acknowledged** they would lose the right. Without both, they can cancel and owe nothing (CCRs reg 37: Checked (official extract)).
  - **Services:** a service can start in the 14 days only if the user **expressly asks**. If they then cancel, they pay a **proportionate amount** for what they had (CCRs reg 36: heading and Sch 2 references Checked (official extract); the proportionate-payment mechanics UNVERIFIED).
- **Recommendation (simple, fair, works under both and under the coming DMCC rules):** at checkout, one unticked box: "Start my Premium now. I understand that if I cancel within 14 days I'll get a refund minus a fair amount for the days I've used." Then refund pro rata by day if they cancel within 14 days. This is more generous than reg 37 allows us to be, which is fine.
- **No pre-ticked paid extras** (CCRs reg 40: UNVERIFIED number). Pre-selecting **annual** as the default plan is not a pre-ticked extra, but show both prices equally clearly, with no "confirmshaming" wording on the monthly option.
- **"Save 52%"** (annual against 12 × £6.99 = £83.88) is accurate, provided the monthly price is a genuine price people pay (CAP Section 3 price rules: UNVERIFIED rule numbers).
- **Total price up front, including VAT** (if registered). Hidden or "drip" fees are a banned practice under the DMCC Act unfair commercial practices rules in force since **6 April 2025** (commencement Checked (secondary); drip pricing as a listed practice UNVERIFIED).

**3b. The DMCC Act 2024 subscription contracts regime (Part 4, Chapter 2).**
- **Commencement: not yet in force today (16 Nov 2026).** The government brought it forward to **1 January 2027** for new subscription contracts. The draft **Consumer Subscription Contracts (Cancellation and Information) Regulations 2026** were laid on **19 October 2026** with draft guidance (Checked (secondary): Osborne Clarke, Lewis Silkin (21 Oct 2026), Winston Taylor, Harper James, Carson McDowell, DWF). Not confirmed from a commencement order or the final regulations: **UNVERIFIED as to the final date and final wording.** Earlier dates (spring 2026, autumn 2026, spring 2027) are superseded.
- **What it will require (from the Act and the draft regulations, per secondary sources):**
  - **Key pre-contract information** given separately and prominently (Checked (secondary)).
  - **Reminder notices:** before a free or discounted trial ends; before an annual (12-month or longer) plan renews; and for monthly plans, periodically, reported as every six months (Checked (secondary)).
  - **Cooling-off:** an initial 14-day cooling-off period, and a **renewal cooling-off period** of 14 days after a free trial converts or an annual plan renews (Checked (secondary)).
  - **Easy exit:** a user who signed up online must be able to end it online, in a single straightforward process. An email address alone is not enough. A clearly labelled button on the account page is (draft reg 32 and guidance: Checked (secondary)).
  - **End-of-contract notice** confirming the exit (Checked (secondary)).
  - **Refunds** within a fixed period (reported as 14 days) (UNVERIFIED).
  - Breach of certain implied terms (reminders, easy exit) gives the consumer a right to cancel without penalty, and the CMA can fine up to 10% of global turnover (fines Checked (secondary)).
- **Scope:** the regime covers contracts that renew or continue automatically. A **Season Pass that does not auto-renew** is likely outside it (s.254 definition: Checked (secondary) that it defines "subscription contract"; the auto-renewal element UNVERIFIED). **Recommendation: make the Season Pass non-renewing**, with one reminder email offering to buy next season.
- **Recommendation:** build the paid flow to the DMCC standard from day one, even if paid launch is before 1 January 2027. It is cheap to do once and avoids rework:
  - a "Cancel subscription" button in Account, two taps, no retention maze (one optional "are you sure" screen with the end date is fine);
  - automatic emails: confirmation; trial-ending reminder 3 to 7 days before conversion; annual renewal reminder 14 to 30 days before; monthly reminder every 6 months; end-of-contract confirmation;
  - a 14-day refund window after first payment, after a trial converts, and after an annual renewal.
- **App stores:** if paid plans are sold through Apple or Google in-app purchase, refunds and cancellation run through the store, and its rules and the regime both apply. Who is the "trader" for DMCC purposes in that case is UNVERIFIED; check the final guidance.

**3c. Free trial (14 to 30 days).** If a card is taken and the trial converts automatically, it is in the DMCC regime (trial reminder and renewal cooling-off) and the CAP "free" rule applies: say clearly before sign-up that it becomes £X a month/year on [date] unless cancelled. **Simplest: a no-card trial** that simply ends.

**3d. Coach and Club tiers.** A club, or a coach buying for their coaching business, is not a "consumer" (an individual acting wholly or mainly outside their trade, business, craft or profession: CRA 2015 s.2(3), UNVERIFIED wording). Business terms are controlled by the Unfair Contract Terms Act 1977 instead (reasonableness test; negligence causing death or personal injury still cannot be excluded: UNVERIFIED). **But many club coaches are unpaid volunteers buying with their own money, who may well be consumers.** Recommendation: give every individual account (including Coach tier) the full consumer rights in the ToS; put only the club pack on business terms (`terms_of_service_v1.md` section 15 does this).

**3e. Changing prices.** A term letting us change the price or terms without a valid reason and without letting the user leave is on the CRA "grey list" of terms that may be unfair (CRA 2015 Sch 2, indicative list: Checked (official extract) that the list exists and is non-exhaustive; paragraph numbers UNVERIFIED). ToS: 30 days' notice, change applies from the next renewal, cancel free before it starts.

---

## Issue 4. AI-generated coaching tips: liability, disclaimer and drill safety

**Current design.** Results page shows a "priority fix", a tip, a drill and a safety line: *"Use a tennis ball or a soft ball for partner drills, keep the feeder out to the side, and wear a helmet for any throw-downs with a hard ball."* The score has a BETA pill and "a guide, not a verdict". Tips come from rubric v0.3, drafted by us, to be edited by Cam (a qualified coach).

**Assessment: MEDIUM.** The product is honest about being beta. What is missing is a fair liability clause and a coach sign-off on the drills.

**4a. What the law says we cannot exclude.**
- **Death or personal injury caused by negligence** cannot be excluded or restricted by any consumer contract term **or consumer notice** (so an on-screen disclaimer counts too). This is absolute, not a fairness test (CRA 2015 s.65 and s.62 carve-out: Checked (official extract)).
- **Paid services must be performed with reasonable care and skill**, judged by the standards of the trade (CRA 2015 s.49: Checked (secondary)). Liability for this cannot be excluded (CRA 2015 s.57: heading Checked (official extract); detail, including the rule that a limit cannot stop the user recovering the price paid, UNVERIFIED).
- **Paid digital content** must be of satisfactory quality, fit for a particular purpose made known, and as described (CRA 2015 ss.34–36: Checked (secondary)). If it damages the user's device or other content through lack of reasonable care, we must repair or compensate (s.46: Checked (official extract)). Whether s.46 applies to free content is UNVERIFIED; assume it does.
- **Free tier:** the Chapter 3 quality rights apply only where the content is paid for (or comes with paid goods, services or content) (CRA 2015 s.33: UNVERIFIED). The revised CMA37 says free digital content licences must still be fair (Checked (secondary)). The negligence rules still apply.
- **Everything else** (for example, a cap on liability for financial loss) must pass the fairness test (CRA 2015 s.62: Checked (official extract) that it is the general fairness requirement).

**4b. What not to write.** "Use at your own risk", "we accept no liability for any injury" or "the AI may be wrong and we are not responsible". These are not binding as to negligent injury, and are on the grey list (excluding liability for death or personal injury). They may also mislead users about their rights, which is itself a regulatory risk (CRA 2015 Sch 2 para 1: Checked (secondary: CMA37 historic examples page); misleading-rights point UNVERIFIED).

**4c. Fair wording to use (in the ToS section 6, and short versions on screen).**
- On the results page, under the tip (short): **"Tips are suggestions from a computer analysis of one clip. They can be wrong. If something doesn't feel right, check with a coach."**
- On the drill box (replace the current safety line): see 4d.
- In the ToS: what the analysis is; that it is beta; that it does not know about the user's body, injuries or experience; that it is not a substitute for a qualified coach; that we use reasonable care and skill and have qualified coaches check the tips and drills; and the statutory carve-out: **"Nothing in these terms limits our liability for death or personal injury caused by our negligence, or for fraud, or any other liability that the law does not allow us to limit."**

**4d. Drill safety notes.**
- **Ask Cam to sign off every drill** (name, date and version in `fault_tips_v0_3.json`). A qualified coach's review is the best evidence that we used reasonable care and skill in writing them.
- **Strengthen the safety line** for all users:
  > "Warm up first. Make sure there is space around you and nobody stands within reach of your swing. Use a tennis ball or soft ball for partner drills, with the feeder standing to the side, not in line with the shot. With a hard ball, always wear a helmet with a faceguard and proper protective kit. Stop if anything hurts."
- **Extra line for under-18s:** "Players under 18: do these drills with an adult supervising, and always wear a helmet with a faceguard when batting against a hard ball, in nets and in matches." ECB safety guidance requires young players to wear helmets with faceguards when batting against a hard ball (UNVERIFIED: an ECB "Safety Measure for Wearing of Helmets by Young People 2026" document appeared in searches, but its content was not read).
- Re-check drills with a hard bat swing near a partner (for example "Side-on to contact" has a partner at mid-off): say the partner must stand well out of reach of the bat.

**4e. Not a medical device.** Coaching technique for a sport is not a medical purpose, so medical device rules should not apply, provided we never claim to diagnose, prevent or treat injury (ASA "pulse check for health apps" page appeared in search; not read: UNVERIFIED). Avoid "prevents injury" claims.

---

## Issue 5. Marketing claims ("as good as a coach")

**Assessment: HIGH RISK if used.** Today's evidence is 18 test clips and coach agreement still being measured (goal 2 in `goals_and_constraints.md` is to "match real coaches as closely as possible", which is an aim, not a result).

- **Objective claims need documentary evidence before publication.** Without it the ASA will treat them as misleading. Marketers must not dress up objective claims as opinion to avoid this (CAP rules 3.1 and 3.7: Checked (official extract)).
- **Comparisons between an AI product and a non-AI alternative** (a human coach) need sufficient evidence; the ASA's AI advice is to not make the claim without it (Checked (official extract): ASA guidance on AI claims). The ASA has recently upheld a complaint about an AI app where there was no evidence its advice was supervised by suitably qualified professionals (Antipadia Enterprises Ltd, ASA ruling 2026: Checked (official extract), detail not read).
- **The same claims in social posts, the app store listing, the website and ambassador posts are all "marketing communications".** Misleading claims are also unfair commercial practices under the DMCC Act, enforceable directly by the CMA since 6 April 2025 (Checked (secondary)).

**Do not use:** "as good as a coach", "replaces your coach", "coach-level accuracy", "ECB approved" or ECB logos, "proven to improve your batting", "prevents injury", user numbers or star ratings we cannot prove.

**Safe to use now:** "AI feedback on your front-foot drive in minutes"; "Tips written with qualified ECB coaches" (only if Alex and Cam hold ECB coaching qualifications, and name the level, e.g. "[ECB Level X]"); "Beta: your score is a guide, not a verdict".

**Usable later, with evidence:** "Agrees with qualified coaches on the main fault in X of Y clips (independent test, [month year], [n] clips)". Keep a **claims file**: the claim, the test, the date, the sample, who checked it. Re-test before reusing a number.

**Reviews and testimonials:** must be genuine. Fake or concealed incentivised reviews are a banned practice under the DMCC Act from 6 April 2025 (Checked (secondary)). Never reward users for app-store reviews or ratings unless the reward does not depend on the review being positive and the incentive is disclosed (UNVERIFIED detail of CMA fake reviews guidance, 4 Apr 2025).

---

## Issue 6. The share card (junior images, consent, face blur default)

**Current design** (`results_tips_and_sharing_prompt.md` Pass 2): a 1080 × 1350 image made in the browser from the **contact frame**, with "{First name}'s BattingIQ", score, BETA pill, priority fault, the app's address. A "Hide face" switch blurs the **top 25% of the image** (or a `face_box` if the backend sends one). Default: **on** for under-18s and unknown age, **off** for adults. Under-18s see "Ask a parent or guardian before sharing a photo." The user shares it themselves through the phone's share sheet. Nothing is posted by us. The research plan adds a watermark and **a referral code worth a free analysis**.

**Assessment: HIGH RISK for juniors as designed. Fine for adults.**

| Problem | Why | Change | Source |
|---|---|---|---|
| 6a. **Name plus image of a child.** | Cricket guidance: if a child is named, do not use an identifiable image; if the image is used, do not name them. | Under-18 or unknown age: **no name on the card** ("My BattingIQ"). | ECB photography guidance as reproduced by clubs and counties: Checked (secondary). UNVERIFIED in ECB's own text. |
| 6b. **Blur can be switched off by a child.** | A 12-year-old can turn it off and share an identifiable image with their first name. | Under-18: **blur locked on**; no switch (show "Faces are always blurred for players under 18"). Unknown age: on by default, switch allowed only after confirming 18+. | Children's code standard 7 (high privacy by default): Checked (official extract). Alex's own rule: faces blurred whenever shown outside the team (`goals_and_constraints.md`). |
| 6c. **Top-25% blur may miss the face.** | At contact, a batter leaning into a drive often has their head lower than the top quarter, especially if the frame includes the bowler's end or sky. | Test on all 18 clips. Prefer the backend `face_box` from pose head points. If neither covers the face, do not show a photo for under-18s; use a plain panel. Rename the switch "Blur face" and only promise what it does. | Accuracy of the promise: CAP rule 3.1 / consumer law misleading actions (Checked (official extract) for rule 3.1). |
| 6d. **Bystanders**, often other children in the nets. | The user may not have consent from them or their parents. | Helper text: "Check nobody else can be recognised in the photo." For under-18 uploads, crop to the batter box when the backend can supply it. | NSPCC Sport photography guidance (respect wishes of children and parents who don't want photos shared): Checked (official extract). |
| 6e. **Referral code on a junior's card.** | A reward for getting friends to sign up turns a child's post into incentivised peer-to-peer marketing, which falls under the CAP Code including Section 5 (children). | **No referral code** on cards from under-18 or unknown-age accounts. Adults: fine, with a clear "Offer terms" link. | CAP guidance on brand ambassadors and peer-to-peer marketing by children: Checked (official extract). |
| 6f. **Link to the analysis.** | Analyses are private. | Keep: home page only, never the analysis page (already in the prompt). | — |
| 6g. **Data protection side** (whether the card is "sharing" by us; the under-18 photo default) | — | Defer to `dp_review.md`. In our view the user, not Batting IQ, discloses the card, because it is made on their device and they choose where to send it. | UNVERIFIED (DP reviewer to confirm). |

---

## Issue 7. Coach and club review of players' videos (Coach and Club tiers)

**Planned:** coaches with up to 25 players; club packs. Not built yet.

**Assessment: HIGH RISK to build without the controls below**, because it gives adults access to videos of children.

**7a. Safeguarding controls (ECB Safe Hands applies to affiliated clubs).** Affiliated clubs must adopt the ECB Safe Hands policy (current version March 2026) and have their own safeguarding statement and a club safeguarding (welfare) officer (Checked (official extract): ECB Safe Hands Policy 2026 and Safeguarding Regulations 2026 exist; detailed content not read). Build these in:
1. **Club account holder** confirms in writing that every coach given access to under-18s is appointed by the club and has been vetted under the ECB's safeguarding requirements (in practice a current ECB DBS check: UNVERIFIED detail), and names the **club welfare officer**.
2. **Parent or guardian approves each link** between an under-18 player and a coach or club, by email confirmation. The player can unlink at any time; so can the parent.
3. **No private messaging** between a coach account and an under-18 account. If comments are added, the parent can see them. Coaches should not text, WhatsApp or privately message juniors; arrangements go through parents (ECB guidance on social media, texts, apps and messaging as reproduced by clubs: Checked (secondary)).
4. **Coaches cannot download** junior videos or share them outside the app. Watermark the coach view with the coach's name.
5. **Access log** (who viewed what, when), kept for [12] months, available to the club welfare officer on request.
6. **Report a concern** button that goes to Batting IQ's safeguarding lead, with the route in `safeguarding_policy_v1.md`.
7. **Unlink on leaving:** when a coach leaves a club, the club removes them; access ends immediately.

**7b. Online Safety Act 2023 (likely relevant once coach review exists).** A "user-to-user service" is one where content uploaded or shared by a user may be encountered by another user (OSA s.3: Checked (official extract) that s.3 defines it; exact wording UNVERIFIED). Today, a player's video is seen only by them and by Batting IQ, so the app is probably **not** a user-to-user service. Once a player's video can be seen by a coach account, it probably is. The "limited functionality" exemption covers only comments and reviews on the provider's own content, not user-uploaded video (OSA Sch 1 para 4: Checked (official extract)). **Before building coach review, do an Online Safety Act scope check**, and if in scope, the illegal content risk assessment and children's access assessment (Ofcom codes: UNVERIFIED as to current deadlines and content).

**7c. Data protection side:** whether the club is a separate controller, the parent approval mechanism and access logging are for `dp_review.md` and `dpia_batting_iq.md`.

---

## Issue 8. Cam's filming session (Phase 4)

**Current documents** (`filming_guide_for_cam.md`, `consent_form_parent_guardian.md`) are already good: welfare officer told, signed forms, under-18 parent consent plus player assent, no filming without a form or if the child says no on the day, location tagging off, codes not names, behind-the-bowler view, no close-ups of faces, private folder, no WhatsApp, delete from phone, forms kept apart from clips, 24-month deletion.

**Assessment: HIGH RISK only until the gaps below are closed.** They are mostly practical.

| Gap | Change | Source |
|---|---|---|
| 8a. The welfare officer is **told**, but nobody **approves**. | Get the club's written approval (club official or welfare officer) for the session, date, who films and why. Follow the club's own photography policy. | ECB photography guidance: filming needs approval from a club official; clubs should display a clear notice when filming takes place: Checked (secondary). |
| 8b. No **notice** on the day. | Put up a notice at the nets: "Filming for coaching technique research today. Ask Cam if you do not want to be filmed." | As above. |
| 8c. **Second adult.** The guide does not say who else is present. | At least one other adult (club coach, team manager or parent) present throughout when under-18s are filmed. Film only in open view, in the net lane. Parents welcome to watch. | ECB Safe Hands good practice on avoiding unobserved one-to-one situations with children: UNVERIFIED (Safe Hands 2026 text not read). |
| 8d. **Cam's vetting and training.** | Confirm Cam's ECB DBS check and safeguarding course are current, and that the club knows Cam is filming in a Batting IQ capacity, not only as a club coach. | ECB Safeguarding Regulations 2026: existence Checked (official extract); content UNVERIFIED. |
| 8e. **Phone cloud backup.** Clips on a personal phone usually sync to iCloud Photos or Google Photos automatically. "Delete from your phone" may leave copies in the cloud and in "Recently deleted". | Before the session: turn off photo sync for the camera app, or use a device without personal cloud backup. After upload: delete the clips, then empty "Recently deleted", then check the cloud photo library. Alex confirms in writing when done. | Storage and deletion: ECB guidance says coaching video must be stored securely and destroyed when no longer needed or on request: Checked (secondary). |
| 8f. **Names list "on paper or in your phone notes".** Phone notes also sync to the cloud. | Paper only, kept with the signed forms in a locked place, or an encrypted file. | As above. |
| 8g. **Children without consent in shot** (background, next net). | Keep the "check the picture before each player" rule. Ask the club to schedule the session when the next lane is empty or adults only. | NSPCC Sport photography guidance (photographers should know which children do not have consent): Checked (secondary). |
| 8h. **Under-12s.** The form says Cam explains and asks. | Keep. Add: if the child seems unsure, do not film. | NSPCC Sport: respect children's wishes: Checked (official extract). |
| 8i. **Concerns on the day.** | Add one line to the guide: "If you have any concern about a child's welfare, tell the club welfare officer the same day. If a child is in immediate danger, call 999." | ECB / club guidance: incidents recorded and reported to the welfare officer or ECB safeguarding team: Checked (secondary). |
| 8j. **WhatsApp.** | Keep "no clips by WhatsApp". Also: do not set up a WhatsApp group with junior players; contact goes to parents. WhatsApp's UK minimum age is 13 (lowered from 16 in April 2024), so a group could legally include juniors, which is why the rule is needed. | WhatsApp age: Checked (secondary: UK press). ECB messaging guidance: Checked (secondary). |
| 8k. **Who is responsible for the data.** | Defer to `dp_review.md` and `dpia_batting_iq.md` (Batting IQ as controller; the club as a separate party). | — |

---

## Issue 9. Ambassadors and influencer disclosure

**Plan:** county and women's players and coaching YouTubers, on revenue share; referral codes.

**Assessment: MEDIUM.**
- **Every ambassador post about Batting IQ is an ad** once there is payment, revenue share, a free account, a gift or an affiliate code. The ASA treats it as advertising where the brand has paid and has some control; the CMA treats payment or a free product alone as enough for consumer law (Checked (secondary): CMA/ASA influencer guidance, updated January 2026).
- **Label:** "Ad" (or "Advert") clearly and up front, on every post, story and video, not hidden among hashtags or only in the bio. "@BattingIQ", "#gifted", "#sp" or "#collab" alone are not enough (CAP rules 2.1 and 2.3, ASA/CMA Influencers' guide: Checked (secondary)). Falsely implying someone is acting as an ordinary consumer is also a banned practice under DMCC Act Sch 20 (Checked (secondary), schedule number UNVERIFIED).
- **Batting IQ is responsible too.** Both the brand and the influencer can be held responsible (Checked (secondary)).
- **Claims:** ambassadors may only make claims on the approved list (Issue 5). No "as good as a coach".
- **No under-18 ambassadors.** CAP Section 5 applies to incentivised marketing by children (Checked (official extract)), and junior safeguarding makes it a poor fit.
- **Ambassadors filming juniors** for Batting IQ content: only with the club's approval and written parent consent, faces blurred, no names. Better: they post only themselves.
- **Contracted players:** county or women's professional players may have contracts that limit personal endorsements. Ask them to check (UNVERIFIED).
- **Written agreement:** disclosure duty, approved claims, no junior content, our right to ask for edits or removal, end on notice, and payment terms.

---

## Issue 10. Housekeeping

- **Company details:** trading name, legal name, company number (if a company), geographic address, email, VAT number (if registered) must be on the site and in the terms (Electronic Commerce (EC Directive) Regulations 2002 reg 6 and CCRs Sch 2: UNVERIFIED). Placeholders are in the ToS.
- **Minors and contracts:** contracts with under-18s are generally not enforceable against them except for necessaries (common law; Minors' Contracts Act 1987: UNVERIFIED). So: **paid plans bought by adults only**; under-18s use the app with a parent or guardian's permission. Minimum age for an account: **Alex to decide** (draft says 13, with under-13s only through a parent's account). Data protection age rules: defer to `dp_review.md`.
- **Complaints:** a named route and a response time. Tell consumers which ADR body we use, or that we do not use one, when a complaint cannot be settled (Alternative Dispute Resolution for Consumer Disputes Regulations 2015: UNVERIFIED).
- **Governing law:** England and Wales. Consumers living in Scotland or Northern Ireland keep their local mandatory protections and can bring claims in their local courts (UNVERIFIED).
- **Solicitor:** the draft terms say a UK solicitor should review before launch. This review agrees, especially for Issues 2, 3 and 7.

---

## Top 10 actions, in priority order

1. **Keep the incentive banner OFF** until there are written offer terms (closing date, eligibility, what counts, the free month ends without charge), the DP reviewer agrees consent stays freely given, and the wording says "let us use" not "share". (Issue 2, HIGH)
2. **Fix the share card for juniors before release:** under-18 and unknown age get no name, blur locked on, no referral code, a bystander warning; test that the blur actually covers the face on all 18 clips. (Issue 6, HIGH)
3. **Close the gaps in Cam's session before filming:** club's written approval and a notice on the day, a second adult present, Cam's vetting current, phone cloud sync off and "Recently deleted" emptied, names list on paper. (Issue 8, HIGH)
4. **Never use "as good as a coach"** or similar. Start a claims file; use only evidenced claims. (Issue 5, HIGH)
5. **Do not build coach or club review of juniors' videos** until parent approval per link, club confirmation of vetted coaches and welfare officer, no private messaging, no downloads, an access log, and an Online Safety Act scope check are in place. (Issue 7, HIGH)
6. **Build the paid checkout to the DMCC standard from day one:** key information before the pay button, express request and acknowledgement box, confirmation email, 14-day pro-rata refund, cancel button in Account, reminder emails, renewal cooling-off. Make the Season Pass non-renewing and the trial no-card. Check the final subscription regulations (expected in force 1 Jan 2027) before launch. (Issue 3, HIGH at paid launch)
7. **Adopt `terms_of_service_v1.md`** after solicitor review, with the AI-advice and safety clause and the statutory carve-outs; remove any "use at your own risk" wording anywhere in the app. (Issue 4, MEDIUM)
8. **Ask Cam to sign off every drill**, and replace the drill safety line with the stronger version, including the under-18 helmet-and-faceguard and adult supervision line. (Issue 4, MEDIUM)
9. **Align the video licence** in the ToS and `training_data_terms_v2.md`: service providers, coach-you-choose, editing acts, share card, no consumer indemnity. (Issue 1, MEDIUM)
10. **Ambassador agreement and housekeeping:** "Ad" on every post, approved claims, no under-18 ambassadors; fill in company details, complaints route and governing law; show `safeguarding_policy_v1.md` to the club welfare officer. (Issues 9 and 10, MEDIUM)

---

## Self-review against each source

| Source | What was relied on | Status |
|---|---|---|
| Consumer Rights Act 2015 s.65, s.62 | Negligence causing death or personal injury cannot be excluded by term or notice; s.62 is the fairness requirement and does not apply to s.65 matters. | Checked (official extract, legislation.gov.uk via search) |
| CRA 2015 s.49, ss.34–36, s.46, s.57 | Reasonable care and skill; digital content quality; device damage; s.57 heading. | s.46 Checked (official extract); s.49 and ss.34–36 Checked (secondary); s.57 detail UNVERIFIED |
| CRA 2015 s.2(3), s.33, s.64, s.68, Sch 2 paragraph numbers | Consumer definition, free content, core terms, transparency, grey list paragraphs. | UNVERIFIED (Sch 2 existence and non-exhaustive nature Checked (official extract)) |
| CMA37 unfair contract terms guidance | Revised 22 July 2026; stand-alone transparency principle; digital content licences must be fair even where free. | Checked (secondary) |
| Consumer Contracts Regulations 2013 regs 13, 16, 29–30, 36, 37, Sch 2 | Information, confirmation, 14-day period, services and digital content cancellation. | reg 37, reg 16, Sch 2 (n)–(o), start of period Checked (official extract); reg 36 mechanics, reg 14 button, reg 40 UNVERIFIED |
| DMCC Act 2024 Part 4 Ch 1 (unfair commercial practices) | In force 6 April 2025; CMA direct enforcement; fines up to 10%; fake reviews banned. | Checked (secondary); Sch 20 numbering UNVERIFIED |
| DMCC Act 2024 Part 4 Ch 2 (subscriptions) | Expected in force 1 January 2027; draft regulations laid 19 October 2026; reminders, cooling-off, easy exit (draft reg 32), end-of-contract notice. | Checked (secondary only); final date and wording UNVERIFIED |
| CAP Code rules 2.1, 2.3, 3.1, 3.7, 3.23, 8.17, Section 5 | Ads identifiable; substantiation; "free"; promotion significant conditions; children under 16; peer-to-peer marketing. | 3.1, 3.7, 3.23, 8.17, Section 5 and peer-to-peer guidance Checked (official extract, asa.org.uk); 2.1/2.3 Checked (secondary); other rule numbers UNVERIFIED |
| ASA AI claims guidance; Antipadia ruling | AI vs non-AI comparisons need evidence. | Checked (official extract), ruling detail not read |
| CMA/ASA influencer guidance (Jan 2026 update) | "Ad" label; scope of what is an ad. | Checked (secondary) |
| ICO consent guidance; Children's code standards 1, 7, 13 | Incentives allowed if no penalty; high privacy defaults; no nudging. | Checked (official extract and secondary). Final DP view deferred to `dp_review.md` |
| Online Safety Act 2023 s.3, Sch 1 para 4 | U2U definition; limited functionality exemption covers comments and reviews on provider content only. | Checked (official extract); application to coach review is the reviewer's view |
| CDPA 1988 ss.5B, 9, 10 | Film authorship (producer and principal director). | Checked (secondary) |
| CDPA 1988 s.180(2), ss.77–87 | Performers' rights scope; moral rights and consent. | UNVERIFIED |
| ECB Safe Hands Policy (March 2026), Safeguarding Regulations 2026 | Clubs must adopt Safe Hands; welfare officer role; regulations exist. | Checked (official extract) for existence and adoption; content UNVERIFIED |
| ECB photo, video and live streaming guidance | Club approval for filming; notice; parents can refuse; named child not pictured; video as a coaching aid stored securely and destroyed. | Checked (secondary: club and county reproductions); ECB's own text UNVERIFIED |
| ECB social media, texts and messaging guidance | No texting or private messaging juniors; go through parents. | Checked (secondary, club copies of 2018 guidance); current ECB version UNVERIFIED |
| ECB helmet guidance for young players | Helmet with faceguard against a hard ball. | UNVERIFIED |
| NSPCC Sport (CPSU) photography and filming guidance | Have a policy; written consent; respect refusals; photographers know who has no consent. | Checked (official extract and secondary) |
| WhatsApp minimum age | 13 in UK since April 2024. | Checked (secondary: press) |
