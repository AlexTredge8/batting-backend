# Verification log: legal claims in the Batting IQ legal pack

Prepared by AI acting as the verification stage of a legal review. Not legal advice. Date of check: **16 November 2026**.

**What this is.** Every legal claim in the eight files below that matters to a decision, checked against a primary source that was actually opened in this session. The reviewers who wrote the files saw only search extracts. This pass opened the pages themselves.

**Files.** DP = `dp_review.md`; TT = `training_data_terms_v2.md`; CW = `consent_wording_v2.md`; DPIA = `dpia_batting_iq.md`; FEE = `ico_fee_and_registration.md`; CS = `consumer_and_safeguarding_review.md`; TOS = `terms_of_service_v1.md`; SG = `safeguarding_policy_v1.md`.

**Results.**
- **VERIFIED**: the primary source says what the file says.
- **CORRECTED**: the file was wrong or out of date. The correct statement is given, and the file has been changed.
- **NOT FOUND**: the source was reachable but the cited item could not be found.
- **STILL UNVERIFIABLE**: the source was blocked in this session.

**Blocked in this session:** resources.ecb.co.uk and play-cricket.ecb.co.uk (all ECB guidance PDFs), nspcc.org.uk, assets.publishing.service.gov.uk (the CMA37 PDF and the DBT subscription guidance PDF). Not checked at all: provider terms (Supabase, Railway, Lovable, Google), the UK-US data bridge, PECR, copyright (CDPA 1988), Minors' Contracts Act, WhatsApp age, law-firm commentary. Those marks stay UNVERIFIED in the files.

**Totals: 108 claims checked. 84 VERIFIED. 15 CORRECTED. 2 NOT FOUND. 7 STILL UNVERIFIABLE.**

Short URL roots used below: `LGU-GDPR/` = https://www.legislation.gov.uk/eur/2016/679/article/ ; `ICO-G/` = https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/ ; `CODE/` = `ICO-G/`childrens-information/childrens-code-guidance-and-resources/age-appropriate-design-a-code-of-practice-for-online-services/ ; `FEE/` = https://ico.org.uk/for-organisations/data-protection-fee/ ; `CRA/` = https://www.legislation.gov.uk/ukpga/2015/15/section/ ; `CCR/` = https://www.legislation.gov.uk/uksi/2013/3134/regulation/ ; `DMCC/` = https://www.legislation.gov.uk/ukpga/2024/13/section/ ; `CAP/` = https://www.asa.org.uk/type/non_broadcast/code_section/ ; `ECB/` = https://www.ecb.co.uk/about/policies-and-regulations/safeguarding/

## 1. ICO data protection fee

| ID | Claim | Files | Primary source opened | Result | Note |
|---|---|---|---|---|---|
| D1 | A controller must pay unless exempt; Batting IQ fits no exemption | FEE, DP | `FEE/`data-protection-fee/exemptions/ | VERIFIED | Full list: staff admin; advertising, marketing, PR; accounts; not-for-profit; personal or household; public register; judicial; non-automated; elected representatives. |
| D2 | Fees changed on 17 Feb 2025 by amending regulations | FEE, DP | https://www.legislation.gov.uk/uksi/2018/480/regulation/3 ; `FEE/`changes-to-the-data-protection-fee/ | VERIFIED | S.I. 2025/63. |
| D3 | Tier 1 £52, tier 2 £78, tier 3 £3,763 | FEE, DP | as D2 | VERIFIED | The stray "£3,726" figure in FEE was wrong and is removed. |
| D4 | Tier 1 = turnover ≤ £632,000 **or** ≤ 10 staff | FEE, DP | reg 3(2) (as D2) | VERIFIED | Measured on the first day of the fee year (reg 3(4)). |
| D5 | £5 off for direct debit, so £47 | FEE, DP | reg 3(5) (as D2) | VERIFIED | |
| D6 | Fixed penalty £400 / £600 / £4,000; maximum £4,350 | FEE, DP | https://ico.org.uk/about-the-ico/our-information/fixed-penalties-for-failure-to-pay-the-data-protection-charge/ | VERIFIED | Still current. |
| D7 | Register shows name and address; a sole trader can choose a correspondence address | FEE | `FEE/`data-protection-fee/information-we-will-collect-and-publish/ | CORRECTED | Address must be the registered office (company) or principal place of business in the UK (sole trader). A home business means a published home address. |
| D8 | Registration number starts with "Z" | FEE | `FEE/`faqs-data-protection-fee-payment-and-online-registration/ | CORRECTED | Starts with Z, A or C. |
| D9 | Online form: name, address, staff, turnover; about 15 minutes; confirmation next working day; on the register within 7 working days | FEE | `FEE/`register/ | VERIFIED | |
| D10 | Pay by direct debit or card | FEE | `FEE/`data-protection-fee/paying-the-data-protection-fee/ | VERIFIED | For a first registration the ICO sends direct debit details when you contact it. |
| D11 | ICO helpline 0303 123 1113 | FEE, DP, TT | as D10 | VERIFIED | Also the breach advice line. |
| D12 | Third-party "registration" agencies have no official standing | FEE | as D8 | VERIFIED | |
| D13 | ICO may become the "Information Commission" (unknown if or when) | FEE, TT, CW, TOS | https://www.legislation.gov.uk/uksi/2026/1015/made | CORRECTED | Happened on **30 September 2026** (DUAA ss118, 119). Website is still ico.org.uk. |
| D14 | A company needs its own fee; one fee covers one legal entity | FEE | as D8 | VERIFIED | "Separate fees must be paid for each company individually." |
| D15 | A fee self-assessment tool exists | FEE | `FEE/`data-protection-fee-self-assessment/ | VERIFIED | |

## 2. Data protection law (UK GDPR, DPA 2018, Children's code)

| ID | Claim | Files | Primary source opened | Result | Note |
|---|---|---|---|---|---|
| P1 | Age 13 for consent to online services is in **DPA 2018 s9** | DP, TT, CW | https://www.legislation.gov.uk/ukpga/2018/12/section/9 ; `LGU-GDPR/`8 | CORRECTED | s9 was repealed on 31 Dec 2020. The rule is **UK GDPR Article 8(1)**. New Art 8(2A) (29 Apr 2026, Children's Wellbeing and Schools Act 2026) lets the government raise the age to up to 16 by regulations. |
| P2 | Under 13: parent must give or authorise consent; reasonable efforts to verify, using available technology; effort depends on risk | DP, TT | `LGU-GDPR/`8 ; `ICO-G/`childrens-information/children-and-the-uk-gdpr/what-are-the-rules-about-an-iss-and-consent/ | VERIFIED | Art 8(1), (2). |
| P3 | The age rule applies only when consent is the lawful basis | DP | as P2 | VERIFIED | "Where point (a) of Article 6(1) applies". |
| P4 | Preventive or counselling services are excluded (cited to s9) | DP (S3) | `LGU-GDPR/`8 | CORRECTED | Now Art 8(4). |
| P5 | Art 6(1)(f): interests override "in particular where the data subject is a child" | DP, DPIA | `LGU-GDPR/`6 | VERIFIED | |
| P6 | Art 6(1)(b): contract "to which the data subject is party" | DP | `LGU-GDPR/`6 | VERIFIED | |
| P7 | Children and contract: capacity; voidable; voided contract ends the basis; Scotland under 16 mostly lack capacity; no fixed age elsewhere | DP | `CODE/`annex-c-lawful-basis-for-processing/ ; `ICO-G/`childrens-information/children-and-the-uk-gdpr/what-do-we-need-to-consider-when-choosing-a-basis-for-processing-children-s-personal-data/ | VERIFIED | |
| P8 | Children act for themselves "from about age 12" as a rule of thumb | DP | as P7 (second URL) | CORRECTED | In **Scotland** a child of 12+ is **presumed** competent. In England, Wales and NI there is **no set age**. |
| P9 | Consent = freely given, specific, informed, unambiguous, clear affirmative action (Art 4(11)) | DP | `LGU-GDPR/`4 | VERIFIED | |
| P10 | Art 7(1) prove consent; 7(3) withdraw as easily as give; 7(4) no bundling with the service | DP | `LGU-GDPR/`7 | VERIFIED | |
| P11 | ICO: some incentive allowed; must not unfairly penalise refusal | DP, CS | `ICO-G/`lawful-basis/consent/what-is-valid-consent/ | VERIFIED | No fixed limit given. |
| P12 | ICO: separate consent for separate purposes; name the controller | DP | as P11 | VERIFIED | Processors need not be named in the consent request. |
| P13 | Art 30(5): under 250 staff exempt from records unless risky, not occasional, or special category | DP | `LGU-GDPR/`30 ; `ICO-G/`accountability-and-governance/documentation/who-needs-to-document-their-processing-activities/ | VERIFIED | Batting IQ must keep a record. |
| P14 | Art 35(1) and 35(3) DPIA duty and examples | DP | `LGU-GDPR/`35 | VERIFIED | |
| P15 | ICO list: innovative technology (incl. AI) + another criterion needs a DPIA | DP, DPIA | `ICO-G/`accountability-and-governance/data-protection-impact-assessments-dpias/examples-of-processing-likely-to-result-in-high-risk/ | VERIFIED | |
| P16 | ICO list covers "biometric data relating to children" | DP | as P15 | CORRECTED | Biometric item = data used to uniquely identify. Separate item = children's data for profiling or **offering online services directly to children**. |
| P17 | Children's code std 2: in-scope services "inevitably" involve high-risk processing; Annex D template | DP, DPIA | `CODE/`2-data-protection-impact-assessments/ | VERIFIED | |
| P18 | Art 36: consult the ICO only if a high risk remains after measures | DP, DPIA | `LGU-GDPR/`36 ; `ICO-G/`accountability-and-governance/data-protection-impact-assessments-dpias/how-do-we-do-a-dpia/ | VERIFIED | |
| P19 | The code has 15 standards (numbers and names as in DP Issue 4) | DP | `CODE/` | VERIFIED | |
| P20 | Std 3 age bands 0–5, 6–9, 10–12, 13–15, 16–17 | DP | `CODE/`3-age-appropriate-application/ | VERIFIED | |
| P21 | Std 4: bite-sized, just-in-time notices; prompt to talk to an adult | DP, CW | `CODE/`4-transparency/ | VERIFIED | |
| P22 | Std 7: high privacy by default | DP, CS | `CODE/`7-default-settings/ | VERIFIED | |
| P23 | Std 10: geolocation off by default | DP | `CODE/`10-geolocation/ | VERIFIED | |
| P24 | Std 11: tell the child when a parent monitors | DP | `CODE/`11-parental-controls/ | VERIFIED | |
| P25 | Std 12: profiling off by default unless compelling reason | DP | `CODE/`12-profiling/ | VERIFIED | Whether the score is a "compelling reason" stays a judgement. |
| P26 | Std 13: no nudges to give data or weaken privacy | DP, CS | `CODE/`13-nudge-techniques/ | VERIFIED | |
| P27 | Std 1, 5, 6, 8, 9, 15 summaries | DP | `CODE/`1-best-interests-of-the-child/ (and 5, 6, 8, 9, 15 pages) | VERIFIED | |
| P28 | Art 25: "children's higher protection matters" (DUAA s81); commencement unknown | DP | `LGU-GDPR/`25 | VERIFIED | In force **5 Feb 2026** (S.I. 2026/82). |
| P29 | Main DUAA data protection changes commenced 5 Feb 2026 | DP | `LGU-GDPR/`12 (amendment notes) | VERIFIED | |
| P30 | Complaints duty from 19 June 2026 (DPA s164A): acknowledge in 30 days, respond, easy route | DP, TT, CW | https://www.legislation.gov.uk/ukpga/2018/12/section/164A | VERIFIED | Also: tell the complainant the **outcome**. |
| P31 | Reply to rights requests within one month (Art 12(3)), extendable | DP | `LGU-GDPR/`12 ; https://www.legislation.gov.uk/ukpga/2025/18/section/76 | CORRECTED | Period now in **Art 12A**: one month, extendable by two months with notice. |
| P32 | Art 4(14) biometric data: "allow or confirm the unique identification" | DP | `LGU-GDPR/`4 | VERIFIED | Words are in the statute. |
| P33 | Biometric data is special category only when used to uniquely identify | DP | `ICO-G/`lawful-basis/biometric-data-guidance-biometric-recognition/key-data-protection-concepts/ ; `LGU-GDPR/`9 | VERIFIED | Pose data used only for coaching is not special category. |
| P34 | Art 4(4) profiling includes "movements" | DP | `LGU-GDPR/`4 | VERIFIED | |
| P35 | Breach: report within 72 hours unless unlikely risk; log all breaches; tell people if high risk; processor tells controller | DP, DPIA | `LGU-GDPR/`33 ; `LGU-GDPR/`34 ; https://ico.org.uk/for-organisations/advice-for-small-organisations/72-hours-how-to-respond-to-a-personal-data-breach/ | VERIFIED | ICO says the guide is under review after the DUAA. |
| P36 | Art 28(3): processor contract terms | DP | `LGU-GDPR/`28 | VERIFIED | |
| P37 | No DPO needed (Art 37 test) | DP, DPIA | `LGU-GDPR/`37 | VERIFIED | |

## 3. Consumer law

| ID | Claim | Files | Primary source opened | Result | Note |
|---|---|---|---|---|---|
| C1 | CRA s65: no exclusion of liability for death or injury from negligence, by term or notice | CS, TOS | `CRA/`65 | VERIFIED | |
| C2 | CRA s62: fairness test; does not affect ss31, 47, 57, 65 | CS | `CRA/`62 | VERIFIED | s62(8). |
| C3 | Liability that cannot be excluded: s31 (goods), s47 (digital content), s57 (services) | CS, TOS | `CRA/`31 ; `CRA/`47 ; `CRA/`57 | VERIFIED | s31 is goods only; Batting IQ sells none. s47 bars any **restriction** of ss34–36 liability. |
| C4 | CRA s49: services with reasonable care and skill | CS, TOS | `CRA/`49 | VERIFIED | |
| C5 | CRA s57: a limit cannot stop the consumer recovering the price paid | CS, TOS | `CRA/`57 | VERIFIED | s57(3). The ToS cap complies. |
| C6 | CRA ss34–36: digital content quality, fitness, as described | CS, TOS | `CRA/`34 | VERIFIED | |
| C7 | CRA s46 device damage; unclear for free content | CS, TOS | `CRA/`46 ; `CRA/`33 | CORRECTED | s33(8): s46 **does** apply to free content. ToS 19.1 changed. |
| C8 | CRA s33: quality rights only for paid content | CS | `CRA/`33 | VERIFIED | |
| C9 | CRA s2(3): consumer definition | CS | `CRA/`2 | VERIFIED | |
| C10 | CRA s64: core terms exempt only if transparent and prominent | CS | `CRA/`64 | VERIFIED | |
| C11 | CRA s68: written terms must be transparent | CS | `CRA/`68 | VERIFIED | |
| C12 | CRA Sch 2 grey list: para 1 (injury), 11 (change terms), 15 (raise price) | CS | https://www.legislation.gov.uk/ukpga/2015/15/schedule/2 | VERIFIED | Non-exhaustive (CMA guide). |
| C13 | CMA37 revised 22 July 2026 | CS | https://www.gov.uk/government/publications/unfair-contract-terms-cma37 | VERIFIED | |
| C14 | CMA: transparency can be enforced even if a term is not unfair | CS | https://www.gov.uk/guidance/writing-a-fair-contract-for-customers | VERIFIED | |
| C15 | CMA37: free digital content licences must be fair | CS | CMA37 PDF (assets.publishing.service.gov.uk) | STILL UNVERIFIABLE | Host blocked. |
| C16 | CCR reg 13 and Sch 2: information before the contract | CS | `CCR/`13 | VERIFIED | |
| C17 | CCR reg 14: key facts directly before ordering; clear "pay" button | CS, TOS | `CCR/`14 | VERIFIED | Wording: "order with obligation to pay" or equally clear. Otherwise the consumer is not bound. |
| C18 | CCR reg 16: confirmation on a durable medium | CS | `CCR/`16 | VERIFIED | |
| C19 | CCR regs 29–30: 14 days from the contract for services and digital content | CS, TOS | `CCR/`29 ; `CCR/`30 | VERIFIED | |
| C20 | CCR reg 36: services start early only on express request; proportionate payment | CS, TOS | `CCR/`36 | VERIFIED | |
| C21 | CCR reg 37: digital content right lost only with express consent and acknowledgement | CS | `CCR/`37 | VERIFIED | |
| C22 | CCR reg 40: no pre-ticked extra payments | CS | `CCR/`40 | VERIFIED | |
| C23 | DMCC unfair commercial practices in force 6 April 2025 | CS | `DMCC/`225 | VERIFIED | S.I. 2025/272. |
| C24 | Drip pricing is a DMCC "banned practice" | CS | https://www.legislation.gov.uk/ukpga/2024/13/schedule/20 ; `DMCC/`230 | CORRECTED | Not in Sch 20. It is a misleading omission: total price must include unavoidable fees (s230(1)(b), (4)). |
| C25 | Fake or concealed incentivised reviews banned | CS | Sch 20 (as C24) | VERIFIED | Para 13, from 6 April 2025. |
| C26 | Falsely posing as a consumer is banned | CS | Sch 20 (as C24) | VERIFIED | Para 25. |
| C27 | "Free" only if nothing to pay beyond unavoidable costs | CS | Sch 20 (as C24) | VERIFIED | Para 23 (also CAP 3.23). |
| C28 | DMCC subscription rules start **1 Jan 2027**; draft regulations laid 19 Oct 2026 | CS, TOS | https://www.legislation.gov.uk/ukdsi/2026/9780348287912 ; https://www.gov.uk/government/news/pm-starts-roll-out-of-everyday-fixes-on-the-cost-of-living-ending-rip-off-discounts-and-subscription-traps | VERIFIED | Still a **draft** awaiting approval; s254 still "prospective" on 16 Nov 2026. |
| C29 | Reminder every six months for monthly plans | CS | `DMCC/`258 | VERIFIED | |
| C30 | 14-day initial and renewal cooling-off periods | CS, TOS | as C28 | VERIFIED | |
| C31 | Online cancel required by "draft reg 32" | CS | `DMCC/`260 | CORRECTED | Source is Act s260(1) and (4). Draft reg 32 only says "at any time". |
| C32 | Subscription = auto-renewing or trial that converts; non-renewing Season Pass outside | CS | `DMCC/`254 | VERIFIED | |
| C33 | ADR Regulations 2015 govern ADR information | CS, TOS | https://www.legislation.gov.uk/uksi/2015/542/contents ; https://www.legislation.gov.uk/uksi/2026/284/made | CORRECTED | 2015 Regulations **revoked**. ADR now under DMCC Part 4 Ch 4 from 6 April 2026. |

## 4. Advertising (CAP Code, ASA)

| ID | Claim | Files | Primary source opened | Result | Note |
|---|---|---|---|---|---|
| A1 | CAP 8.17: promotions must state significant conditions, incl. closing date; child promotions likely need one | CS | `CAP/`08.html | VERIFIED | 8.17.4.b. |
| A2 | CAP 3.23: "free" rule | CS | `CAP/`03.html | VERIFIED | |
| A3 | CAP 3.1 and 3.7: no misleading; hold evidence before publishing | CS | `CAP/`03.html | VERIFIED | |
| A4 | "Must not dress up objective claims as opinion" (cited to 3.1, 3.7) | CS | `CAP/`03.html | CORRECTED | The opinion rule is 3.6: must not imply opinions are objective claims. |
| A5 | CAP 2.1 and 2.3: ads must be identifiable; no posing as a consumer | CS | `CAP/`02.html | VERIFIED | |
| A6 | CAP Section 5: child = under 16 | CS | `CAP/`05.html | VERIFIED | |
| A7 | CAP guidance on children as brand ambassadors and peer-to-peer marketing | CS | https://www.asa.org.uk/advice-online/brand-ambassadors-and-peer-to-peer-marketing.html | VERIFIED | Existence only; detail not read. |
| A8 | ASA guidance on AI claims | CS | asa.org.uk search and likely URLs | NOT FOUND | CAP 3.7 still applies. |
| A9 | Antipadia ruling: AI app, no evidence of qualified supervision | CS | https://www.asa.org.uk/rulings/antipadia-enterprises-ltd.html | CORRECTED | 4 Nov 2026, upheld: unlicensed **medical** claims, discouraging treatment, suicide imagery. |
| A10 | ASA "pulse check for health apps" | CS | asa.org.uk | NOT FOUND | |

## 5. Safeguarding (ECB, NSPCC) and online safety

| ID | Claim | Files | Primary source opened | Result | Note |
|---|---|---|---|---|---|
| E1 | ECB Safe Hands Policy (March 2026) and Safeguarding Regulations 2026 exist | CS, SG, TOS | `ECB/`safe-hands | VERIFIED | |
| E2 | Safe Hands content: clubs must adopt it; avoid unobserved one-to-one | CS, SG | Safe Hands PDF (resources.ecb.co.uk) | STILL UNVERIFIABLE | Host blocked (403). |
| E3 | ECB DBS check required | CS, SG, TOS | `ECB/`safer-recruitment | VERIFIED | ECB-specific DBS for regulated activity; rechecked every 12 months. |
| E4 | ECB safeguarding training for the role | SG | `ECB/`safeguarding-training | VERIFIED | "Safeguarding Level 2" replaced SSR; valid 3 years. |
| E5 | Role names "club welfare officer" and "ECB Safeguarding Team" | CS, SG, TOS, CW | `ECB/`safeguarding-training ; `ECB/`share-a-concern | CORRECTED | Now **Club Safeguarding Officer**, County Safeguarding Officer, **Cricket Regulator** Safeguarding Team. |
| E6 | Concern route: club, then county, then national team | SG | `ECB/`share-a-concern | VERIFIED | Use the Safeguarding Referral Form; safeguarding@cricketregulator.co.uk. |
| E7 | Concerns about adults go to the LADO via ECB | SG | ECB concerns PDF (resources.ecb.co.uk) | STILL UNVERIFIABLE | Host blocked. |
| E8 | ECB photo and video guidance (approval, notice, no name with image) | CS, SG, CW | play-cricket.ecb.co.uk article linked from `ECB/`kit-bag-resources | STILL UNVERIFIABLE | Host blocked. |
| E9 | ECB: under-18s wear helmets with faceguards against a hard ball | CS, TOS | ECB safety documents (resources.ecb.co.uk) | STILL UNVERIFIABLE | Host blocked. |
| E10 | ECB: no private messaging of juniors | CS, TOS | ECB social media PDF (resources.ecb.co.uk) | STILL UNVERIFIABLE | PDF exists (listed in the Kit Bag); text blocked. |
| E11 | NSPCC helpline 0808 800 5000 | SG | nspcc.org.uk | STILL UNVERIFIABLE | Host blocked. |
| O1 | OSA s3: user-to-user service definition | CS | https://www.legislation.gov.uk/ukpga/2023/50/section/3 | VERIFIED | |
| O2 | OSA Sch 1 para 4: limited functionality exemption covers only comments and reviews on provider content | CS | https://www.legislation.gov.uk/ukpga/2023/50/schedule/1/paragraph/4 | VERIFIED | User videos are never "provider content". |

## Most important corrections

1. **P1 / P4.** Cite **UK GDPR Article 8**, not DPA 2018 section 9. Section 9 was repealed in 2020. Watch Article 8(2A): the age could rise to 16.
2. **D13.** The ICO is now the **Information Commission** (30 Sept 2026). Notices that tell people where to complain now say so.
3. **D7.** The ICO register publishes your **business address**. For a home-based sole trader that is the home address.
4. **C7.** The device-damage remedy (CRA s46) applies to the **free** tier too.
5. **C24, C31, C33.** Drip pricing is a misleading omission (s230), not a banned practice. The online cancel duty is in the Act (s260(4)). The 2015 ADR Regulations are revoked.
6. **E5.** Use the current ECB role names: Club Safeguarding Officer; Cricket Regulator Safeguarding Team.
