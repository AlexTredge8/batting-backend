# Alex: actions and decisions (legal pack, 16 November 2026)

Prepared by AI. Not legal advice. Written in simplified technical English (ASD-STE100 style).

Sources are in `verification_log.md`. A **[U]** mark means "not verified". Do a check before you rely on an item with this mark.

## A. Do now (before 30 November 2026)

**A1. Pay the ICO data protection fee.** The fee is **£47 by direct debit** or **£52 by card** (tier 1). Payment is overdue, because the app already gets uploads.
1. Decide the address. The ICO publishes it. A sole trader must give the main place of business. A company gives its registered office.
2. Go to https://ico.org.uk/for-organisations/data-protection-fee/register/ and select "Start your new registration application".
3. Enter the legal name, "Batting IQ" as trading name, address, staff number and turnover. Select tier 1.
4. Pay. Ask for direct debit to get the £5 discount.
5. Keep the registration reference (it starts with Z, A or C). Put it in the privacy notice and Forms A and B.

**A2. Set up a complaint route.** The law has required this since 19 June 2026.
1. Make a complaint form or a "Privacy complaint" email address.
2. Acknowledge each complaint in 30 days or less.
3. Tell the person the result.

**A3. Start two logs:** a breach log and a rights-request log. Record all breaches, also small breaches.

**A4. Delete the pre-opt-in app uploads** on the Mac. Write the date of deletion in the DPIA.

**A5. Write a consent record for each of the 18 original clips.** If a clip has no written consent for training and for Gemini, get a written yes or do not use the clip.

## B. Do before the consent opt-in goes live

1. Use the `2026-12-v2` wording only. Use the "Who is the batter?" flow.
2. Let only the batter (18 or older) or a parent agree. Do not let a coach or friend agree.
3. Send a confirmation email to the parent for each under-18 clip. Do not use the clip until the parent confirms.
4. Block accounts for persons under 13. Send them to a parent's account.
5. Complete and sign the DPIA. The law requires a DPIA for an online service for children.
6. Publish the full privacy notice at `/privacy`. Put in the ICO number. Name the regulator as the "Information Commission (ico.org.uk)".
7. Accept the data processing terms for Supabase, Railway, Google Cloud and Lovable. Record the data region of each. **[U]**
8. Set "Hide face" to on for all users. Remove location data from uploaded videos.

## C. Do before paid plans launch

1. Show the price, renewal period, auto-renewal and cancel method immediately before the pay button.
2. Label the pay button "Subscribe and pay". If the label is not clear, the customer is not bound.
3. Show the total price. Include VAT and all fees that the customer must pay.
4. Add an unticked box: "Start my Premium now". Give a pro-rata refund if the customer cancels in 14 days.
5. Send a confirmation email after each purchase.
6. Put a "Cancel subscription" button in Account. The law requires online cancellation for online sign-ups.
7. Send reminders: before a trial ends, before a yearly renewal, and each 6 months for monthly plans.
8. Give 14 days to cancel after a trial converts and after a yearly renewal.
9. Read the final subscription regulations. The draft says they start on **1 January 2027**. Parliament has not approved them yet.
10. Make the Season Pass non-renewing. Make the trial a no-card trial.
11. Keep a claims file. Publish a performance claim only when you have test evidence.
12. Select the ADR text in ToS 22.3. If you name an ADR provider, make sure it has accreditation under the DMCC Act. **[U]**

## D. Decisions only you can make

| Decision | Options | Recommendation | Risk (one line) |
|---|---|---|---|
| Minimum account age | 13; 16; 18 | **13.** Persons under 13 use a parent's account. | The government can now raise the consent age to 16 (Article 8(2A)). The ICO checks that age limits work in practice. |
| Lawful basis for account holders aged 13 to 17 | Contract; legitimate interests | **Legitimate interests.** Write the balancing test in the DPIA. | A contract with a minor is voidable. Legitimate interests must give extra weight to children. |
| Training reward ("month free for 3 clips") | On; off | **Off** until you write offer terms with a closing date. Adults only. | CAP rules 8.17 and 3.23. Consent is not "freely given" if refusal causes a penalty. |
| Liability cap (ToS 20.4) | £100 or amount paid (greater); amount paid only; no cap | **Greater of £100 and the amount paid**, with the 20.2 exceptions. | A cap must pass the fairness test. It must not stop a refund of the price. **[U]** for the £100 figure. |
| Refund in the first 14 days | Pro-rata refund; full refund; no refund after start | **Pro-rata refund** for each day. | The "no refund" option fails if you do not get express consent and acknowledgement. Then the customer pays nothing. |
| Face blur for under-18s on the share card | Locked on; on with a switch | **Locked on.** Also: no name and no referral code on junior cards. | Children's code standard 7. The ECB photo guidance is **[U]**. |
| Published ICO address | Home address; business address; a limited company | Decide before A1. | A sole trader working from home gets the home address on a public register. |

## E. Do not do

- Do not switch on the `2026-11-v1` opt-in.
- Do not let anyone except the batter or a parent agree to training use.
- Do not use "as good as a coach", "prevents injury", "ECB approved" or ECB logos.
- Do not write "use at your own risk". Do not exclude liability for injury.
- Do not pay the fee through an agency. Pay on ico.org.uk only.
- Do not build face recognition or body matching.
- Do not build coach or club access to junior videos before an Online Safety Act scope check. Uploaded user videos do not get the "limited functionality" exemption.
- Do not send private messages to junior players. Contact the parents. **[U]** for the ECB text.
- Do not film juniors before the Club Safeguarding Officer approves in writing. Make sure that the person who films has a current ECB DBS check (the ECB does a new check every 12 months).
- Do not cite "DPA 2018 section 9". The rule is now UK GDPR Article 8.

**Still not verified [U]:** all ECB guidance text (photos, helmets, messaging, the LADO route), the NSPCC helpline number, provider contracts, and CMA37 on free licences. Ask the Club Safeguarding Officer about the ECB items.
