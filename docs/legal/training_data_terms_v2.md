Prepared by AI acting in the role of a data-protection reviewer. Not legal advice. Items marked HIGH RISK or UNVERIFIED should be decided by Alex knowingly.

# Batting IQ: Terms of Service clause and Privacy Notice (version 2)

Replaces `training_data_terms_draft.md` (v1). Consent wording version: **`2026-12-v2`**. Reasons and sources for every change are in `dp_review.md` (issue numbers given below). Source marks follow `dp_review.md`: **UNVERIFIED (snippet)** means seen only in search extracts; **UNVERIFIED** on its own means no source was found.

> **Read this first (Alex).**
> - **No solicitor.** You have decided not to use one. So each HIGH RISK or UNVERIFIED item below is your decision: tick it off in the DPIA (`dpia_batting_iq.md`, section 8) when you have decided.
> - **One wording only.** Section 13.4 of `terms_of_service_v1.md` must be replaced with part (a) below, word for word. On data protection, this file wins.
> - **Keep it in line with the app.** The upload screen (`consent_wording_v2.md`), the team rules (`training_data_rules.md`) and this notice must say the same. If one changes, change the others and give the consent text a new version.
> - **Only promise what is built.** Lines marked *[only if true]* describe things to build first. Do not publish them until they are done.
> - **Fill in** every `[square bracket]`.

Placeholders: `[Controller]` (for example "Alex Tredgett, trading as Batting IQ" or "Batting IQ Ltd, company number …"), `[Address]`, `[Privacy email]`, `[ICO registration number]` (see `ico_fee_and_registration.md`), `[Effective date]`.

---

## Summary of changes from v1 (redline summary)

| # | Change | Why (see `dp_review.md`) | Risk it fixes |
|---|---|---|---|
| 1 | **Only the batter, or the batter's parent or guardian, can agree to training use.** A coach, friend or other uploader can no longer agree on someone else's behalf. Old clause [X].6 ("the batter has agreed") is deleted. | Consent must come from the person the data is about, or a holder of parental responsibility (Issue 3) | HIGH |
| 2 | **Under-18s: two yeses.** Players aged 13 to 17 say yes themselves, and a parent or guardian confirms by email. Under 13: parent or guardian only, through the parent's account. Nothing is used until the email is confirmed. | UK GDPR Article 8 (CORRECTED (https://www.legislation.gov.uk/eur/2016/679/article/8): the draft cited DPA 2018 s9, which was repealed on 31 December 2020); Children's code; reasonable efforts to verify, Article 8(2) (Issues 2 to 4) | HIGH |
| 3 | **Full privacy notice for the whole app**, not only training. | Articles 13 and 14 (Issue 7) | HIGH |
| 4 | **Google Gemini disclosed**, with its own separate yes for session clips. | Transparency; specific consent (Issue 6) | MEDIUM |
| 5 | **Lawful bases per group**: contract for your own account; legitimate interests for people filmed by someone else and bystanders; consent for training; legitimate interests (not legal obligation) for consent records. | Issues 1 and 17 | MEDIUM |
| 6 | **"We do not profile you" replaced** with an accurate statement: the score is automated analysis of how you move; there is no other profiling. | Children's code standard 12 (Issue 4) | MEDIUM |
| 7 | **Biometric statement made explicit**: pose measurements are never used to recognise anyone. | Articles 4(14) and 9 (Issue 5) | LOW (HIGH if breached) |
| 8 | **Keeping periods filled in** (proposals for Alex): 12 months for ordinary videos; stills deleted with the video; accounts closed after 24 months without sign-in. | Article 5(1)(e) (Issue 8) | MEDIUM |
| 9 | **Location data removed from videos** *[only if true]*. | Children's code standard 10 (Issue 4) | MEDIUM |
| 10 | **Complaints procedure** added (in force since 19 June 2026). Rights section updated for the DUAA. | s164A DPA 2018; DUAA (Issue 15) | MEDIUM |
| 11 | **Transfers**: a mechanism for each company, to be checked and filled in. | Chapter V (Issue 12) | MEDIUM |
| 12 | **Child-friendly summary** for 13 to 17. | Children's code standard 4 (Issue 4) | MEDIUM |
| 13 | **UK only**: the service is for people in the UK. | EU GDPR reach (Issue 13) | LOW |
| 14 | "Our coaching AI does not contain or show your video" changed to "does not contain a copy of your video". | Accuracy (Issue 17) | LOW |
| 15 | Questions for the solicitor removed; replaced by Alex's decisions in the DPIA. | Alex's decision not to use a solicitor | — |

---

## (a) Terms of Service clause (replaces `terms_of_service_v1.md` section 13.4 and the v1 clause [X])

**13.4 Using your videos to improve Batting IQ (optional)**

**13.4.1 What you always give us.** When you upload a video, you let us store and analyse it so that we can give you your score, tips and storyboard. We need this to provide the service. It does not include training our coaching AI.

**13.4.2 What you can choose to give us.** When you upload a video, you can choose to let us use it to train and test Batting IQ's coaching AI. This is optional. The box is unticked unless you tick it, and saying no changes nothing else in the app. If you say yes, you give us a non-exclusive, royalty-free permission to copy, store and analyse that video, and to make measurements from it, only to:
- (a) train, test and improve Batting IQ's analysis of batting technique;
- (b) let qualified cricket coaches who work with us watch it, with the face blurred, and rate the technique; and
- (c) check that our scores and tips agree with those coaches.

**13.4.3 Who can say yes.** Only these people can give this permission:
- **the batter**, if they are 18 or over and it is their own account;
- **the batter, aged 13 to 17, together with their parent or guardian.** The player says yes, and the parent or guardian confirms by email. We do not use the video until the parent or guardian has confirmed;
- **the batter's parent or guardian**, using their own account, for a child of any age under 18.

If you upload a video of someone else (for example as a coach, team-mate or friend), you cannot give this permission for them. They can upload the video from their own account, or their parent or guardian can.

**13.4.4 What we will not do.** We will not:
- sell your video;
- licence it to anyone else for their own use;
- publish it, or use it in adverts, marketing or social media;
- use it, or the measurements made from it, to recognise who anyone is.

We will only do any of the first three if we ask you separately and you agree. We will never do the last. If we show a video to anyone outside the Batting IQ team, we blur the face first.

**13.4.5 Changing your mind.** You can withdraw your permission for any video at any time in Account, under "My videos and data", or by emailing [Privacy email]. A player can withdraw even if their parent or guardian agreed, and a parent or guardian can withdraw for their child. When you withdraw:
- we stop using the video for anything new; and
- we remove it, and the measurements and ratings made from it, from our training and test sets within 30 days.

Training already done can't be undone, but our coaching AI does not contain a copy of your video. Withdrawing does not affect anything we lawfully did before you withdrew.

**13.4.6 Your promises.** When you give this permission, you confirm that:
- you are one of the people in 13.4.3;
- you have the right to share the video: you filmed it, or the person who filmed it agrees; and
- you have tried not to film other people close up. If other people can be seen clearly and you are not sure they would mind, do not tick the box.

**13.4.7 Who owns what.** You keep any rights you have in your video. We own the improvements we make to Batting IQ, including our coaching AI.

**13.4.8 How long we keep it.** We keep a video you have shared for training for up to 3 years from the date you agreed, or until you withdraw or delete it, if sooner. Our Privacy Notice has the details.

**13.4.9 Rewards (only if we run an offer).** From time to time we may offer a reward for sharing videos for training, for example a free month of Premium.
- Rewards are only for players aged 18 and over sharing their own videos.
- The offer's own terms say what you need to do.
- If you withdraw your permission later, you keep the reward.
- Not taking part never costs you anything.

**13.4.10 Removing videos.** We may remove any video from our training sets at any time, for example if it is not suitable or we are told the right person did not agree.

---

## (b) Privacy Notice (publish at `/privacy`)

### Batting IQ Privacy Notice

Last updated: [Effective date]. Consent wording version: 2026-12-v2.

**Are you under 18?** There is a shorter version written for you at the end of this notice. Please read it with a parent or guardian.

#### 1. Who we are

[Controller], [Address], is responsible for your personal data (the "controller"). ICO registration number: [ICO registration number]. Contact: [Privacy email]. Batting IQ is a small business. Our privacy contact is Alex Tredgett.

Batting IQ is for people in the UK.

#### 2. What we collect

| What | Where it comes from |
|---|---|
| **Account details**: name, email address, date of birth, password (we store only a scrambled version) | You |
| **Videos** you upload. They show the batter's body, kit and surroundings, and may show other people in the background. | You |
| **Measurements** we make from each video: body positions frame by frame, timings, scores, faults and tips, and still pictures for your storyboard | Our analysis |
| **Your choices**: whether you agreed to training use, when, the wording version, who agreed, and any withdrawal | You |
| **Parent or guardian details** (for under-18 training permission): name, email address, and when they confirmed | The parent or guardian, or the player |
| **Coach ratings** of videos shared for training | Qualified coaches who work with us |
| **Messages** you send us, including privacy requests and complaints | You |

*[only if true]* When you upload a video, we remove the location data that phones can store inside video files.

**What we do not collect.** We do not use face recognition. We do not use your video or measurements to recognise who anyone is. We do not try to work out anything about your health, ethnicity or other sensitive matters, and our tips are about technique only.

#### 3. Why we use it, and our lawful basis

| What we do | Lawful basis under UK GDPR |
|---|---|
| Run your account | **Contract** (Article 6(1)(b)): we need it to provide the service you signed up for |
| Store and analyse videos you upload of **yourself**, to give you your score, tips and storyboard | **Contract** (Article 6(1)(b)) |
| Analyse a video of **someone else** you upload (for example your child, or a player you coach), and any people in the background | **Legitimate interests** (Article 6(1)(f)): giving you the coaching feedback you asked for. We have weighed this against the interests of the people filmed, especially children (see "Legitimate interests" below) |
| Use a video to train and test our coaching AI, and have coaches rate it | **Your consent** (Article 6(1)(a)), given separately for each video. For under-18s, see section 7 |
| Send a face-blurred session clip to an outside AI service during development | **Your consent** (Article 6(1)(a)), a separate yes on the session form |
| Keep a record of each permission and withdrawal | **Legitimate interests** (Article 6(1)(f)): being able to show we had permission |
| Keep the service secure and stop misuse | **Legitimate interests** (Article 6(1)(f)): running a safe service |
| Answer privacy requests and complaints | **Legal obligation** (Article 6(1)(c)) |

*Decision for Alex: for account holders aged 13 to 17, the table uses contract. You may instead choose legitimate interests for all under-18 account holders (see `dp_review.md`, Issue 1). Pick one before launch.*

**Legitimate interests.** Where we rely on legitimate interests, our interest is giving coaching feedback on a video the uploader chose to send us, and keeping that service safe. We limit the effect on the people filmed: videos stay private, are never published, are never used to recognise anyone, are never used for training without the right person's permission, and are deleted on the timetable in section 6. You can object at any time (section 9).

**Automated analysis.** Your score and tips come from automated analysis of how you move in the video. That analysis is the service you asked for. It is a guide, not a verdict. It does not make any decision with legal or similarly significant effects on you, and we do not use it for anything else: no advertising, no profiles for other purposes, and no sharing with clubs or selectors.

#### 4. Using videos to improve Batting IQ (only if you agree)

**What we use.** For each video shared for training: the video; the measurements made from it; coach ratings; and the record of permission.

**Who can agree.** The batter (18 or over, own account); or a player aged 13 to 17 together with a parent or guardian, who confirms by email; or a parent or guardian, from their own account, for a child under 18. Someone who uploads a video of another person cannot agree for them.

**What we use it for.** Only to improve Batting IQ's analysis of batting technique: training and testing our coaching AI, and checking it against qualified coaches. We will not use shared videos for any other product, for advertising or marketing, or sell or share them for anyone else's use. If we ever want to use them for something new, we will ask you first.

**Face blurring.** Whenever we show a video, or a still from it, to anyone outside the Batting IQ team (including coaches using our rating tools), we blur the face first. Coaches see videos with a code, not a name. Even so, someone who knows a player might recognise their build or kit.

**Outside AI services.** While we are developing Batting IQ, we have used, and may use, a paid AI service (Google's Gemini API) to check our own analysis, on a small number of face-blurred clips, as a one-off and never in the live app. Under its paid terms, Google does not use what we send to improve its own products, but keeps records for a limited time to detect misuse (UNVERIFIED (snippet): S27 in `dp_review.md`; the time limit is UNVERIFIED). We only send a clip if the person agreed to this separately (on the session consent form). We do not send app uploads to outside AI services.

**Changing your mind.** Switch off "Use to improve Batting IQ" in Account, under "My videos and data", or email us. We stop using the video for anything new and remove it, and its measurements and ratings, from our training and test sets within 30 days. Training already done can't be undone, but our coaching AI does not contain a copy of your video.

#### 5. Who can see your data, and who helps us

**Inside Batting IQ.** Only you (and anyone you share your account with) and named Batting IQ admins can open your videos through the app. Qualified coaches who rate training videos sign a confidentiality agreement and see face-blurred clips without names.

**Companies that process data for us** (processors), under contracts that require them to protect it:

| Company | What it does for us | Where the data may be processed | Transfer safeguard *[Alex to check and fill in]* |
|---|---|---|---|
| Supabase | Database, video storage and sign-in | [project region, for example London]; support may be from the USA | [UK Addendum to the EU Standard Contractual Clauses in Supabase's DPA, or the UK-US data bridge if certified] |
| Railway | Runs our analysis server | [region]; company based in the USA | [UK Addendum / IDTA / data bridge] |
| Lovable | Builds and hosts the app's web pages | [check]; company based in the USA (UNVERIFIED) | [UK addendum in Lovable's DPA] |
| Google (Gemini API, paid) | One-off development checks on face-blurred clips (only if you agreed) | USA and elsewhere | [UK-US data bridge or the UK Addendum in Google's terms] |
| [Email provider] | Sends permission emails and receives privacy requests | [check] | [check] |

We do not sell your data, and we do not share it with advertisers, clubs or anyone else for their own use, unless the law requires us to (for example a court order), or to protect a child from harm.

**Sharing your result.** If you make a share card of your result, it is made on your phone and you choose where to send it. We do not post it anywhere. "Hide face" is on to start with.

**International transfers.** Some of these companies are based in, or give support from, the USA. When your data leaves the UK, we rely on one of: the UK-US "data bridge" (where the company is certified under it); the UK International Data Transfer Agreement or the UK Addendum to the EU Standard Contractual Clauses; or another safeguard UK law allows. You can ask us for details.

#### 6. How long we keep it

| What | How long |
|---|---|
| A video **not** shared for training | 12 months from upload, or sooner if you delete it *[Alex to confirm the period]* |
| Your analysis results and storyboard stills | While your account is open. When you delete a video, we delete its stills too *[only if true: backend change]* |
| A video shared for training, and its measurements and coach ratings | Up to 3 years from the date you agreed. Sooner if you withdraw (out of training and test sets within 30 days) or delete it |
| Results worked out from many videos together (for example averages, and our trained coaching AI) | Kept. They do not identify you |
| Records of permissions and withdrawals (including a parent or guardian's name and email) | As long as the video, plus 1 year |
| Session clips filmed by our coach (with a signed form) | 24 months from the session |
| Your account | Until you close it. If you do not sign in for 24 months, we email you, then close it and delete your data *[only if true]* |
| Privacy requests and complaints | 2 years after we close them |
| Backups | Deleted data can stay in our providers' backups for up to [x] days, then it is gone *[Alex to check the Supabase plan]* |

#### 7. Children

Many of our users are juniors. We follow the ICO's Age Appropriate Design Code (the Children's code), and design Batting IQ with children's protection in mind.

- **Accounts.** You must be 13 or over to have your own account. Younger players use Batting IQ through a parent or guardian's account.
- **Best interests first.** We design Batting IQ to help young players improve. We never use a junior's data in a way that could harm them, never show their scores to clubs or selectors, and never use their videos for marketing.
- **High privacy by default.** Training use is off unless chosen, on every upload. We never remember a previous "yes". "Hide face" on the share card starts on.
- **Training use of under-18s' videos needs a parent or guardian.** A player aged 13 to 17 says yes themselves, and their parent or guardian confirms by email. We do not use the video until they confirm. For a child under 13, only a parent or guardian can agree, from their own account. Either the player or the parent or guardian can withdraw at any time. We are stricter than UK law, which lets children aged 13 and over agree to some online uses by themselves, because these are videos of young people and the use is commercial.
- **No nudging.** We do not use rewards, reminders, pop-ups or highlighted buttons to encourage under-18s to share. Reward offers are never shown to them.
- **Clear explanations.** There is a short version of this notice for young players, below.
- **No profiling beyond the analysis, no face recognition, no publishing.**
- **No location.** We do not ask for your location. *[only if true]* We remove location data stored in video files.

#### 8. Keeping your data safe

- Videos are kept in private cloud storage. Links to them expire after an hour.
- Only our servers hold the keys that give wider access, never the app.
- Team accounts use strong passwords and two-step sign-in.
- Copies made for training are kept on encrypted computers and deleted on the timetable above.
- We never send videos by messaging apps or post them to public links.
- If something goes wrong that puts your data at risk, we will tell the ICO within 72 hours where the law requires, and tell you if there is a high risk to you.

#### 9. Your rights

You have the right to:
- **withdraw consent** at any time, as easily as you gave it (Account, "My videos and data", or email us);
- **get a copy** of the data we hold about you;
- **correct** it;
- **delete** it ("Delete this video", or email us);
- **object** to how we use it where we rely on legitimate interests, and **restrict** our use of it;
- **move** it to another service (data portability).

To use a right, use the account page or email [Privacy email]. It is free. We reply within one month. If we need more information to find what you asked for, or to check who you are, the month starts once we have it. We search for your data in a reasonable and proportionate way.

**Children's rights.** Rights belong to the child. A parent or guardian can act for a young child. Older children can usually act for themselves, and we take their wishes into account. A player can always withdraw permission for training use, even if their parent or guardian agreed.

#### 10. Complaints

If you are unhappy with how we have handled your data, tell us first at [Privacy email] or through the form at [link]. We will acknowledge your complaint within 30 days (we aim for 3 working days), look into it, and tell you the outcome.

You can also complain to the Information Commission (formerly the Information Commissioner's Office, ICO): ico.org.uk, 0303 123 1113. *[Alex: the new name and the number are VERIFIED (https://www.legislation.gov.uk/uksi/2026/1015/made; https://ico.org.uk/for-organisations/data-protection-fee/data-protection-fee/paying-the-data-protection-fee/). Delete this note before publishing.]*

#### 11. Changes

If we change how we use your data, we will update this notice and the version number. If we want to use shared videos for something new, we will ask you again.

---

## (c) Privacy notice for players aged 13 to 17 (publish next to the full notice)

### Your videos and Batting IQ: the short version

*Please read this with a parent or guardian.*

**Who we are.** Batting IQ is a small UK business run by Alex Tredgett. We give you tips on your batting from a phone video. If you have a question, email [Privacy email].

**What happens to your video.**
- We look at your video with a computer program that tracks how your body moves. That's how we work out your score and tips.
- Your video is private. Only you, anyone who uses your account, and a few people at Batting IQ can open it.
- We never put your video online, in adverts or on social media.
- We never use it to work out who you are. There is no face recognition.
- Your score is a guide, not a verdict. We never show it to your club or coaches unless you share it yourself.

**Helping us improve Batting IQ (only if you choose).**
- When you upload, you can say yes to letting us use the video to make our coaching AI better.
- It is your choice. If you say no, nothing changes. Your tips are exactly the same.
- If you are under 18, you **and** your parent or guardian both need to say yes. We send them an email to check. We only use the video after they confirm.
- If you say yes, cricket coaches who help us might watch it to check our tips. We blur your face first, and they don't see your name.
- You can change your mind any time, even if your parent or guardian said yes. Go to Account, then "My videos and data", and switch it off. We take it out within 30 days.

**How long we keep things.** Ordinary videos: up to 12 months. Videos you shared to help us: up to 3 years, unless you change your mind or delete them sooner.

**Sharing your result.** If you make a share card, "Hide face" starts switched on. Ask a parent or guardian before you share a photo of yourself.

**Your rights.** You can ask us for a copy of your data, ask us to fix it, or ask us to delete it. Just email us. It's free.

**Worried about something?** Talk to a parent, guardian, teacher or coach, or email us. You can also contact the Information Commission (ico.org.uk). They make sure organisations look after people's data properly.
