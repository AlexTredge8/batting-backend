# Training data rules (for the team and for all code)

Agreed 16 Nov 2026. This applies to every person and every script that touches Batting IQ clips. If in doubt, do not use the clip. Ask Alex.

## 1. The rule

Use only a clip with recorded permission for training, testing, validation, tuning thresholds, coach rating, or showing to anyone outside the team. For an app upload, that means:

- `training_consent = true`, and
- `withdrawn_at is null`, and
- `consent_version` and `consent_timestamp` are filled in, and
- `consent_timestamp` is less than 3 years ago.

"Use" includes downloading the clip, extracting frames or body positions from it, sending it to any outside service (including paid AI during development), and putting it in a rating tool.

## 2. Which clips are allowed

| Source | Allowed? | Covered by |
|---|---|---|
| App uploads that meet section 1 | Yes | The opt-in at upload (wording version `2026-11-v1` onwards) |
| The 18 original calibration clips | Yes | The consent each person gave Alex directly. Keep a note of it (below). |
| Phase 4 session clips filmed by Cam | Yes, within the form's limits | The signed Form A or Form B. Use in coach rating tools only if box 2 was ticked. Delete within 24 months of the session. |
| App uploads made before the opt-in existed | **No** | Nothing. Their rows have `training_consent = false` and no `consent_version`. |
| Anything from YouTube, the web or research-only datasets | **No** | Nothing |

**Note for the 18 original clips.** Keep one private record for each clip:
- the clip file name;
- the person's name;
- the date they agreed;
- how they agreed (in writing, by message, or in person);
- what they agreed to, and any limits.

Keep it with the signed Phase 4 forms, **not in git**: it holds names. If anyone agreed only in person, or only to testing, get a one-line written reply that covers commercial training before using their clip for that. If anyone in the 18 was under 18 when filmed, the record must show that a parent or guardian agreed.

**App uploads made before the opt-in existed.** `phase4_app_uploads.py` downloads app uploads with no consent check. Do not use the clips it has already downloaded for training, testing or coach rating. Delete them from the Mac. Before running it again, change it to take only the ids returned by the query below.

**Under-18 clips** (`consent_given_by = 'parent_guardian'`) are allowed. Until the app emails the guardian to confirm, look at each one before use, and leave it out if anything looks wrong, for example the guardian's email looks like the child's own.

## 3. How scripts must filter

Scripts read Supabase with the service role key. That key ignores row level security, so every script must filter for itself, at the start of every run. Never reuse a saved list that is more than 7 days old.

```sql
select id, storage_path, consent_version, consent_given_by, consent_timestamp
from public.uploads
where training_consent = true
  and withdrawn_at is null
  and consent_version is not null
  and consent_timestamp is not null
  and consent_timestamp > now() - interval '3 years'
order by consent_timestamp;
```

From Python, the same filter over the REST API:
`/rest/v1/uploads?select=id,storage_path,consent_version&training_consent=eq.true&withdrawn_at=is.null&consent_version=not.is.null&consent_timestamp=gt.<today minus 3 years>`

Each run must:
- write a manifest of the upload ids and consent versions it used, kept with the run's outputs;
- never put clips, frames or names into git;
- read the key from an environment variable, never from code.

## 4. Withdrawal

1. **Where people withdraw.** In the app (Account, then "My videos and data"), or by email to the privacy address.
   - For an email request, Alex runs `update public.uploads set training_consent = false where id = '<id>';` within 3 working days. The database stamps `withdrawn_at` itself.
   - A player can withdraw even if their parent or guardian agreed.
2. **From then on**, every new run leaves the clip out, because of the filter.
3. **Every 2 weeks, and before every training run,** list what has gone:
   ```sql
   select upload_id, event, created_at
   from public.upload_consent_events
   where event in ('withdrawn', 'deleted') and created_at > now() - interval '45 days'
   order by created_at;
   ```
   For each id, delete every local copy, and the frames, body positions and coach ratings made from it, on the Mac, in rating tools and in any cloud folder.
4. **Deadline.** All of this is done within **30 days** of the withdrawal: the promise made in the app.
5. **Models already trained** are not retrained just to remove one clip. They do not contain the clip, and the next scheduled retrain leaves it out.
6. **Deletion requests.** Delete the video in the app (or an admin does it), then do step 3 for that id.

## 5. How long to keep things

| What | How long |
|---|---|
| App clips shared for training | Up to 3 years from `consent_timestamp`, or until withdrawn or deleted. Check every 3 months and delete anything older. |
| Phase 4 session clips | 24 months from the session date, as the form says |
| The 18 original clips | As each person agreed. If no time was agreed, 3 years from the date they agreed. |
| Measurements, frames and ratings made from a clip | The same as the clip |
| Consent records (`upload_consent_events`, signed forms, the note for the 18) | As long as the clip, plus 1 year |
| Overall results and trained models that identify no one | May be kept |

## 6. Always

- Blur faces before showing any clip or still outside the team.
- Never use clips for face recognition, marketing or publicity.
- Never send clips to an outside AI service unless its terms say it does not train on our data, and Alex has said yes.
- Keep clips on encrypted devices only. Never share them by WhatsApp, email or public link.
