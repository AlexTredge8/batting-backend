Prepared by AI acting in the role of a data-protection reviewer. Not legal advice. Items marked HIGH RISK or UNVERIFIED should be decided by Alex knowingly.

# Consent wording, version 2: upload screen and session consent forms

Consent wording version: **`2026-12-v2`**. Replaces the wording in `docs/lovable/upload_training_consent_prompt.md` (`2026-11-v1`) and `docs/phase4/consent_form_parent_guardian.md`. Reasons are in `dp_review.md` (Issues 2 to 6 and 10). The privacy notice that this wording points to is `training_data_terms_v2.md`.

> **Notes for Alex.**
> - **HIGH RISK: do not switch on the v1 upload opt-in.** It lets an uploader agree for someone else, and uses an unconfirmed parent email for under-18s. Use this wording and the flow in Part 1 instead.
> - The Lovable prompt needs a new pass (call it Pass 1b) to build the flow in Part 1. The database changes are listed in Part 1, section 1.8. This file gives the words; it does not rewrite the prompt.
> - Do not reword the app text without changing the version.
> - The child-friendly wording is for ages 13 to 17 in the app, and 12 to 17 on paper (Form A's player agreement). The Children's code asks for information suited to the child's age (VERIFIED (https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/age-appropriate-design-a-code-of-practice-for-online-services/4-transparency/)).

---

## Part 1. Upload screen (the app)

### 1.1 How the flow works

| Step | Shown when | What happens |
|---|---|---|
| A. "Who is the batter?" | Always, for signed-in users (signed-out users see no training section) | Three choices, none selected: "Me", "My child (I'm their parent or guardian)", "Someone else" |
| B1. Adult, own video | "Me", and the account holder is 18 or over (from date of birth) | One unticked box (1.3) |
| B2. Young player, own video | "Me", and the account holder is 13 to 17 | Young-person version (1.4): the player's own tick, plus a parent or guardian's email. Nothing is used until the parent or guardian confirms by email (1.6) |
| B3. Parent's own child | "My child" | Parent version (1.5): the parent ticks; for a child of 13 to 17 they also confirm they have talked to the child |
| B4. Someone else | "Someone else" | No tick. Short explanation (1.2, `otherNote`) |
| B5. Age unknown | "Me", and there is no date of birth on the profile | Ask: "Are you 18 or over?" Yes goes to B1; No goes to B2 |

Account holders under 13 are not allowed (sign-up blocks them and points to a parent's account).

Rules that apply to everyone (unchanged from v1): nothing pre-selected or remembered; no pop-ups, badges, highlight colours or reminders; the choice never blocks or slows the upload; the analysis is identical whatever is chosen; the reward banner is never shown on B2, B3, B4 or B5, or to anyone under 18.

### 1.2 Common text

| Key | Text |
|---|---|
| `sectionTitle` | Help improve Batting IQ (optional) |
| `whoQuestion` | Who is the batter in this video? |
| `whoMe` | Me |
| `whoChild` | My child (I'm their parent or guardian) |
| `whoOther` | Someone else |
| `ageQuestion` (B5 only) | Are you 18 or over? |
| `otherNote` | Only the batter, or their parent or guardian, can agree to us using a video to improve Batting IQ. If they'd like to help, they can upload it from their own account. Your analysis is exactly the same either way. |
| `whatLink` | What does this mean? |
| `privacyLink` | Read our privacy notice |
| `privacyLinkYoung` | Read the short version for young players |

### 1.3 Adult, own video (B1)

| Key | Text |
|---|---|
| `selfCheckbox` | Yes, Batting IQ can use this video of me to train and test its coaching AI. |
| `selfHelper` | Optional. Your analysis is exactly the same either way. You can change your mind at any time. |

**`whatPoints` (adult "What does this mean?")**
- Batting IQ is run by [Controller]. Questions: [Privacy email].
- We keep your video private, in secure storage. Only the Batting IQ team and the qualified coaches who help us can see it.
- We use it only to improve how Batting IQ analyses batting: to train our coaching AI and to test that its scores and tips are right.
- Qualified cricket coaches may watch it and rate the technique. We blur the face first, and they don't see your name.
- We never sell it. We never publish it, or use it in adverts, marketing or social media, unless we ask you separately and you say yes.
- We never use it, or the measurements we make from it, to recognise who anyone is. There is no face recognition.
- We don't send app videos to outside AI services.
- We keep it for up to 3 years, then delete it.
- You can change your mind at any time in Account, under "My videos and data". We then stop using it for anything new and remove it from our training sets within 30 days. Training already done can't be undone, but our coaching AI doesn't contain a copy of your video.
- Saying no changes nothing else in the app.

### 1.4 Young player, own video (B2, ages 13 to 17)

| Key | Text |
|---|---|
| `youngTitle` | Want to help make Batting IQ better? |
| `youngBody` | This is your choice. If you say no, nothing changes, and your tips are exactly the same. If you say yes, a parent or guardian needs to say yes too. Talk to them first. |
| `youngReveal` | I've talked to my parent or guardian. Show me the form |
| `youngCheckbox` | Yes, I'm happy for Batting IQ to use this video of me to help train its coaching AI. |
| `guardianEmailLabel` | Your parent or guardian's email |
| `guardianEmailHelper` | We'll send them one email to ask if they agree. We only use your video if they say yes. We won't email them about anything else. |
| `youngAfterUpload` (message after upload, if ticked) | Thanks. We've emailed your parent or guardian. We won't use this video to train our AI unless they say yes. You can change your mind in Account, under "My videos and data". |

**`whatPointsYoung` ("What does this mean?" for 13 to 17)**
- Your video stays private. Only you and a few people at Batting IQ can open it.
- We use it to teach our computer program to give better batting tips.
- Some cricket coaches who help us might watch it to check our tips. We blur your face first, and they don't see your name.
- We never put it online, in adverts or on social media.
- We never use it to work out who you are. There is no face recognition.
- We keep it for up to 3 years, then delete it.
- You can change your mind any time, even if your parent or guardian said yes. Go to Account, then "My videos and data". We take it out within 30 days.
- If you're not sure, say no. Nothing bad happens.

Validation (same style as v1): the email must look valid and must not be the account holder's own email. Messages: "Please enter a valid email address." / "Please enter your parent or guardian's email, not your own." Under them, the text button "Upload without sharing for training".

### 1.5 Parent or guardian, own child (B3)

| Key | Text |
|---|---|
| `parentTitle` | Help improve Batting IQ with this video of your child (optional) |
| `parentBody` | Your child's analysis is exactly the same either way. If your child is old enough to understand, please talk to them about it first. Their wishes matter, and they can change their mind later. |
| `parentChildAge` | How old is your child? (Under 13 / 13 to 17) |
| `parentNameLabel` | Your full name |
| `parentCheckbox` | I am this player's parent or guardian. I have read "What does this mean?" and I agree that Batting IQ may use this video to train and test its coaching AI. |
| `parentTalkedCheckbox` (13 to 17 only) | I have talked to my child about this, and they are happy for the video to be used. |
| `parentHelper` | You or your child can change your mind at any time in Account, under "My videos and data", or by emailing [Privacy email]. |

The adult `whatPoints` list (1.3) is used, with "you" meaning "your child" where it refers to the batter. Show `privacyLinkYoung` as well as `privacyLink`.

For B3 the parent's own account email is used for the confirmation email (1.6). The parent is signed in to their own account, so the confirmation can be a single click in the email or on a confirmation screen; Alex may decide to skip the email for B3 because the parent is the account holder (record the decision in the DPIA).

### 1.6 Parent or guardian confirmation email (B2, and B3 if used)

**Subject:** Please confirm: can Batting IQ use [player's first name]'s video?

**Body:**

Hello,

[Player's first name] uploaded a batting video to Batting IQ on [date] and said they're happy for us to use it to help train our coaching AI. Because they're under 18, we only do this if a parent or guardian agrees.

What it means:
- The video stays private. Only the Batting IQ team, and qualified cricket coaches who help us, can see it. Coaches see it with the face blurred and no name.
- We use it only to improve how Batting IQ analyses batting.
- We never sell it, publish it or use it in marketing, and never use it to recognise anyone.
- We keep it for up to 3 years. You or [first name] can change your mind at any time, and we take it out of our training sets within 30 days.
- If you say no, or do nothing, nothing changes: [first name]'s analysis is exactly the same.

[Button: I'm their parent or guardian, and I agree]
[Link: I don't agree]

Full details: [privacy notice link]. Questions: [Privacy email].

This link works for 14 days. If you didn't expect this email, you can ignore it, and we won't use the video.

[Controller], [Address]

**Confirmation page** (after the button): asks for the parent or guardian's full name and a tick: "I am [first name]'s parent or guardian and I agree." Then: "Thank you. You or [first name] can change your mind at any time by emailing [Privacy email], or in [first name]'s account under 'My videos and data'."

*If not confirmed in 14 days:* the consent stays unconfirmed, the video is never used for training, and the record is cleared (guardian email deleted) after 30 days.

### 1.7 Account page ("My videos and data") changes

Keep the v1 Pass 2 wording, with these changes:
- Status line for an unconfirmed under-18 video: "Waiting for your parent or guardian to confirm. We won't use it until they do."
- Status line for a confirmed one: "Shared for training since {date}, with agreement from {guardian name}."
- Turning on a video later (from the account page): allowed only for an adult's own video (B1). Otherwise the helper: "To share this video for training, upload it again. For players under 18, a parent or guardian must agree."
- Add to the footer: "Want a copy of your data, or to make a complaint? Email [Privacy email] or use [form link]."

### 1.8 Database changes (for the Lovable Pass 1b)

On `public.uploads`, add:
- `batter_relationship text` check in ('self', 'own_child', 'other');
- `player_agreed boolean` (13 to 17, own video);
- `guardian_confirmed_at timestamptz` and `guardian_confirm_token_hash text` (store only a hash of the email link token);
- the trigger must refuse `training_consent = true` when `batter_relationship = 'other'`.

Change the training filter in `training_data_rules.md` section 3 to add:

```sql
  and batter_relationship in ('self', 'own_child')
  and (uploader_under_18 is false or guardian_confirmed_at is not null)
  and consent_version >= '2026-12-v2'
```

---

## Part 2. Session consent forms (paper, for filmed net sessions)

> **Note for Alex (delete before printing).** Check the forms with the club's safeguarding officer (the ECB now calls the welfare officer the "Club Safeguarding Officer": CORRECTED (https://www.ecb.co.uk/about/policies-and-regulations/safeguarding/safeguarding-training)), and follow the club's filming policy and the ECB's guidance on photos and filming. Fill in the contact line, the privacy notice link and the session date. Keep signed forms locked away, apart from the clips. These forms are on paper, so the online-services age rule (UK GDPR Article 8; CORRECTED (https://www.legislation.gov.uk/eur/2016/679/article/8): the draft cited DPA 2018 section 9, which was repealed on 31 December 2020) does not apply, but the same parent-and-child approach is right for children.

### Form A: players under 18 (parent or guardian consent, and the player's own agreement)

**About this filming**

**Who we are.** Batting IQ is a UK app that gives feedback on a cricket front-foot drive from a phone video. It is run by [Controller], [Address]. ICO registration number: [number]. Contact: Alex Tredgett, [Privacy email]. Full privacy notice: [link].

**What we film.** On [date], our coach, Cam, will film your child playing 2 to 4 front-foot drives in the nets. The phone stands behind the bowler, so it mostly sees the batter's body and bat from a distance. We do not film faces close up. We record each player under a code (for example P07), not a name. Only Cam keeps the list that links codes to names.

**Why.** To test and improve how the app judges batting technique. Qualified coaches watch the clips and rate the technique, and we check the app's scores against theirs. We also use measurements of body position from the clips to train and tune the app. We never use the clips, or those measurements, to recognise who anyone is (no face recognition).

**Where we keep the clips.** In private cloud storage with a password and two-step sign-in. Only the Batting IQ team and named coaches can open it. We do not send clips by WhatsApp or post them online.

**Faces blurred.** If we show a clip to anyone outside that group, we blur the face first.

**Outside AI service (only if you tick box 3).** While we develop the app, we may send a few face-blurred clips to a paid AI service (Google's Gemini) to check our own results. Google's paid terms say it does not use them to improve its own products, though it keeps records for a limited time to detect misuse. This is optional and separate.

**No publicity.** We will not publish the clips or use them in marketing, adverts or social media. If we ever want to, we will ask you again, separately.

**How long we keep them.** We delete the clips, and the measurements made from them, within 24 months of the session date, or sooner if you ask. Results worked out from many clips together (for example an average, or the app's trained model) do not identify your child and are kept.

**It is your choice.** Taking part is optional. Saying no, or withdrawing later, does not affect your child's coaching or team selection in any way.

**Withdrawing.** You or your child can withdraw at any time, before or after filming, all at once or just one box. Email [Privacy email] or tell Cam, giving the child's name and the session date. We then delete your child's clips and measurements (or stop the use you withdrew from) within one month.

**The law.** Under UK GDPR, our lawful basis is your consent. You can ask to see the clips we hold, ask us to correct or delete them, or object. If you are unhappy, tell us first at [Privacy email]; we will acknowledge your complaint within 30 days. You can also complain to the Information Commission, which replaced the Information Commissioner's Office (ico.org.uk). We follow the ICO's Children's code: we collect only what we need, keep it only as long as we need it, and put your child's best interests first.

**Parent or guardian consent.** Please tick each box you agree to. You can tick box 1 only.

- [ ] **1.** I have read this form. I agree that my child may be filmed, and that the clips and measurements may be used to test, train and improve the Batting IQ app, as described above.
- [ ] **2.** I agree that my child's clips may be shown, face-blurred and without a name, to qualified coaches in rating tools. (Someone who knows your child might still recognise their build or kit.)
- [ ] **3.** I agree that a few of my child's face-blurred clips may be sent to an outside AI service (Google Gemini) during development, as described above.

Child's name: ______________________ Age: ______

Parent or guardian name: ______________________

Signature: ______________________ Date: __________

Phone or email (only to contact you about this filming): ______________________

**Player's agreement (ages 12 to 17)**

*Please read this, or ask an adult to read it with you.*

- Cam will film you playing a few drives in the nets.
- We'll use the video to help make a cricket app give better tips.
- Coaches who help us may watch it. We blur your face first and don't show your name.
- If you and your parent tick box 3, a computer service run by Google may check a few blurred clips too.
- We won't put your video online or in adverts, and we won't use it to work out who you are.
- We delete it within 2 years.
- You can say no now, or change your mind later. Just tell Cam or your parent. Nobody will be cross, and it won't change your coaching or your place in a team.

- [ ] I'm happy to be filmed, and for the video to be used as my parent or guardian has ticked above.

Player's name: ______________________ Signature: ______________________ Date: __________

**For players under 12.** Cam reads this to the player before filming: "I'm going to film you playing a few drives. It's to help make a cricket app give better tips. Coaches might watch it, but we'll blur your face. We won't put it online. Is that OK? You can say no, and that's fine." If the player says no, or seems unsure, Cam does not film them.

### Form B: adult players (18 and over)

**Who we are.** Batting IQ is a UK app that gives feedback on a cricket front-foot drive from a phone video. It is run by [Controller], [Address]. ICO registration number: [number]. Contact: Alex Tredgett, [Privacy email]. Full privacy notice: [link].

**What we film.** On [date], Cam will film you playing 2 to 4 front-foot drives in the nets, from behind the bowler. We do not film faces close up. We record you under a code, not your name. Only Cam keeps the list that links codes to names.

**Why.** To test, train and improve how the app judges batting technique. Qualified coaches rate the clips, and we check the app's scores against theirs. We never use the clips or measurements to recognise who anyone is.

**Storage and access.** Private cloud storage with a password and two-step sign-in. Only the Batting IQ team and named coaches can open it. If we show a clip to anyone else, we blur the face first.

**Outside AI service (only if you tick box 3).** During development we may send a few face-blurred clips to Google's Gemini (paid service) to check our results. Google's paid terms say it does not use them to improve its own products, though it keeps records for a limited time to detect misuse.

**No publicity.** No publication, marketing, adverts or social media without your separate permission.

**How long.** We delete the clips and their measurements within 24 months of the session date, or sooner if you ask.

**Your choice.** Taking part is optional and does not affect your coaching or team selection. You can withdraw at any time, from everything or one box: email [Privacy email] or tell Cam. We act within one month.

**The law.** Under UK GDPR, our lawful basis is your consent. You can ask to see, correct or delete your clips, or object. Complaints: tell us first at [Privacy email] (we acknowledge within 30 days); you can also complain to the Information Commission, formerly the ICO (ico.org.uk).

- [ ] **1.** I agree to be filmed, and for the clips and measurements to be used to test, train and improve the Batting IQ app, as described above.
- [ ] **2.** I agree that my clips may be shown, face-blurred and without a name, to qualified coaches in rating tools.
- [ ] **3.** I agree that a few of my face-blurred clips may be sent to an outside AI service (Google Gemini) during development.

Name: ______________________ Signature: ______________________ Date: __________

Phone or email (only to contact you about this filming): ______________________

---

## Part 3. What changed from v1, in one table

| v1 | v2 | Why (`dp_review.md`) |
|---|---|---|
| Any uploader could tick "if the batter isn't you, only tick this if they have agreed" | Only the batter, or their parent or guardian | Issue 3 (HIGH) |
| Under-18: a typed name, email and tick, no check | Player's own yes (13 to 17) plus parent or guardian confirming by email; nothing used until confirmed | Issues 2 to 4 (HIGH) |
| Age question "Is the batter 18 or over?" | "Who is the batter?" plus the account holder's age | Issue 3 |
| No controller name in the app text | Controller and contact in "What does this mean?" | Issue 7 |
| Paper forms silent on outside AI | Separate, optional box 3 for Google Gemini | Issue 6 |
| Paper forms: one complaint route (ICO) | Complaint to Batting IQ first (30-day acknowledgement), then the ICO | Issue 15 |
| Child text for 12 and over only | Also a short script for under-12s, and a 13 to 17 version in the app | Issue 4 |
| Version `2026-11-v1` | Version `2026-12-v2` | — |
