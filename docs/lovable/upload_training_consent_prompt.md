# Lovable prompt: optional training consent at upload, and "My videos and data"

Two passes. Paste **Pass 1** into Lovable, check it against its tests, then paste **Pass 2**. Each pass is everything between its two lines. The notes at the very end are for Alex and the backend, not for Lovable.

**Before you paste (Alex, 2 minutes).** Replace these two placeholders everywhere in this file:
- `{{PRIVACY_EMAIL}}`: the email address people should write to about their data (for example a privacy@ address on the app's domain).
- `{{ADMIN_EMAIL}}`: the email address Alex signs in to the app with. It is used once, to make that account an admin.

Why this exists: we may only train and test the coaching model on clips we are allowed to use. This adds a clear, optional, unticked opt-in at upload, a parent or guardian route for under-18s, a record of every choice in Supabase, and a page where people can withdraw or delete. The wording matches the session consent forms (`docs/phase4/consent_form_parent_guardian.md`) and the legal drafts (`docs/legal/training_data_terms_draft.md`). The rule every script follows is in `docs/legal/training_data_rules.md`.

---

## Pass 1: the database and the upload screen

Add an optional "help improve Batting IQ" choice to the upload screen, and store that choice with each upload in Supabase. Do not change how the analysis works: the same video goes to the same backend call, with the same result, whatever the user chooses. Do not send any of the new fields to the backend (the Railway API). Use UK spelling. Keep the current fonts, colours and layout everywhere else.

**1. Database migration.** Create one Supabase migration with the SQL below, in this order. The `uploads` table already exists with `id`, `name`, `file_url` and `created_at`.
- If `uploads.id` is not a `uuid`, change `upload_id uuid` in section D to the same type as `uploads.id`.
- If `public.user_roles` and `public.has_role(uuid, app_role)` already exist, skip section A, but still run its last statement (making `{{ADMIN_EMAIL}}` an admin).

```sql
-- A. Admin role (Lovable's standard pattern). Skip if it already exists.
do $$ begin
  create type public.app_role as enum ('admin', 'user');
exception when duplicate_object then null; end $$;

create table if not exists public.user_roles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  role public.app_role not null,
  unique (user_id, role)
);
alter table public.user_roles enable row level security;
drop policy if exists "Users can read their own roles" on public.user_roles;
create policy "Users can read their own roles" on public.user_roles
  for select to authenticated using (user_id = auth.uid());

create or replace function public.has_role(_user_id uuid, _role public.app_role)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.user_roles where user_id = _user_id and role = _role)
$$;

insert into public.user_roles (user_id, role)
select id, 'admin' from auth.users where email = '{{ADMIN_EMAIL}}'
on conflict (user_id, role) do nothing;

-- B. New columns on uploads. Older rows get training_consent = false and are never used for training.
alter table public.uploads
  add column if not exists user_id uuid references auth.users(id) on delete cascade,
  add column if not exists storage_path text,
  add column if not exists uploader_under_18 boolean,
  add column if not exists training_consent boolean not null default false,
  add column if not exists consent_version text,
  add column if not exists consent_timestamp timestamptz,
  add column if not exists consent_given_by text,
  add column if not exists guardian_name text,
  add column if not exists guardian_email text,
  add column if not exists withdrawn_at timestamptz;

alter table public.uploads
  add constraint uploads_consent_given_by_check
    check (consent_given_by is null or consent_given_by in ('self', 'parent_guardian')),
  add constraint uploads_consent_complete_check
    check (training_consent = false
           or (consent_version is not null and consent_timestamp is not null
               and consent_given_by is not null and withdrawn_at is null)),
  add constraint uploads_guardian_details_check
    check (consent_given_by is distinct from 'parent_guardian'
           or (length(trim(coalesce(guardian_name, ''))) >= 2
               and guardian_email ~* '^[^@\s]+@[^@\s]+\.[^@\s]+$')),
  add constraint uploads_minor_needs_guardian_check
    check (not (training_consent and uploader_under_18 is true
                and consent_given_by is distinct from 'parent_guardian'));

-- Where older rows have a Supabase storage URL, work out the file's path in the bucket.
update public.uploads
set storage_path = substring(file_url from '/storage/v1/object/(?:public|sign|authenticated)/[^/]+/([^?]+)')
where storage_path is null and file_url is not null;

create index if not exists uploads_user_created_idx on public.uploads (user_id, created_at desc);
create index if not exists uploads_training_idx on public.uploads (consent_timestamp) where training_consent;

-- C. Guard: the database, not the browser, sets times and protects the consent record.
create or replace function public.uploads_consent_guard()
returns trigger language plpgsql set search_path = public as $$
declare
  caller_is_admin boolean := coalesce(public.has_role(auth.uid(), 'admin'), false);
begin
  if tg_op = 'INSERT' then
    if auth.uid() is not null and not caller_is_admin then
      new.user_id := auth.uid();
    end if;
    if new.training_consent then
      new.consent_timestamp := now();
      new.withdrawn_at := null;
      if new.consent_given_by = 'self' then
        new.guardian_name := null;
        new.guardian_email := null;
      end if;
    else
      new.consent_version := null;
      new.consent_timestamp := null;
      new.consent_given_by := null;
      new.guardian_name := null;
      new.guardian_email := null;
      new.withdrawn_at := null;
    end if;
    return new;
  end if;

  -- UPDATE
  if auth.uid() is not null and not caller_is_admin then
    if new.user_id is distinct from old.user_id
       or new.file_url is distinct from old.file_url
       or new.storage_path is distinct from old.storage_path
       or new.uploader_under_18 is distinct from old.uploader_under_18
       or new.created_at is distinct from old.created_at then
      raise exception 'These details of an upload cannot be changed.';
    end if;
  end if;

  if old.training_consent and not new.training_consent then
    -- Withdrawal: stamp the time and keep the record of what was agreed.
    new.withdrawn_at := now();
    new.consent_version := old.consent_version;
    new.consent_timestamp := old.consent_timestamp;
    new.consent_given_by := old.consent_given_by;
    new.guardian_name := old.guardian_name;
    new.guardian_email := old.guardian_email;
  elsif not old.training_consent and new.training_consent then
    -- Fresh consent from the account page: only for a batter recorded as 18 or over.
    if old.uploader_under_18 is distinct from false or new.consent_given_by is distinct from 'self' then
      raise exception 'A parent or guardian must agree. Upload the video again to ask them.';
    end if;
    new.consent_timestamp := now();
    new.withdrawn_at := null;
    new.guardian_name := null;
    new.guardian_email := null;
  else
    -- No change of choice: the consent record cannot be edited.
    new.consent_version := old.consent_version;
    new.consent_timestamp := old.consent_timestamp;
    new.consent_given_by := old.consent_given_by;
    new.guardian_name := old.guardian_name;
    new.guardian_email := old.guardian_email;
    new.withdrawn_at := old.withdrawn_at;
  end if;
  return new;
end $$;

drop trigger if exists uploads_consent_guard on public.uploads;
create trigger uploads_consent_guard before insert or update on public.uploads
  for each row execute function public.uploads_consent_guard();

-- D. A permanent log of every consent event (no guardian details in it).
create table if not exists public.upload_consent_events (
  id bigint generated always as identity primary key,
  upload_id uuid not null,      -- no foreign key on purpose: the record outlives a deleted video
  user_id uuid,
  event text not null check (event in ('given', 'withdrawn', 'deleted')),
  consent_version text,
  consent_given_by text,
  created_at timestamptz not null default now()
);
alter table public.upload_consent_events enable row level security;
drop policy if exists "Owners and admins can read consent events" on public.upload_consent_events;
create policy "Owners and admins can read consent events" on public.upload_consent_events
  for select to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(), 'admin'));
-- No insert, update or delete policies: only the trigger below writes here.

create or replace function public.uploads_log_consent_event()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  if tg_op = 'INSERT' and new.training_consent then
    insert into public.upload_consent_events (upload_id, user_id, event, consent_version, consent_given_by)
    values (new.id, new.user_id, 'given', new.consent_version, new.consent_given_by);
  elsif tg_op = 'UPDATE' and old.training_consent is distinct from new.training_consent then
    insert into public.upload_consent_events (upload_id, user_id, event, consent_version, consent_given_by)
    values (new.id, new.user_id,
            case when new.training_consent then 'given' else 'withdrawn' end,
            new.consent_version, new.consent_given_by);
  elsif tg_op = 'DELETE' and old.consent_version is not null then
    insert into public.upload_consent_events (upload_id, user_id, event, consent_version, consent_given_by)
    values (old.id, old.user_id, 'deleted', old.consent_version, old.consent_given_by);
  end if;
  return coalesce(new, old);
end $$;

drop trigger if exists uploads_log_consent_event on public.uploads;
create trigger uploads_log_consent_event after insert or update or delete on public.uploads
  for each row execute function public.uploads_log_consent_event();

-- E. Row level security: only the owner and admins can read an upload.
alter table public.uploads enable row level security;
do $$ declare p record; begin
  for p in select policyname from pg_policies where schemaname = 'public' and tablename = 'uploads' loop
    execute format('drop policy %I on public.uploads', p.policyname);
  end loop;
end $$;

create policy "Owners and admins can read uploads" on public.uploads
  for select to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(), 'admin'));
create policy "Signed-in users can add their own uploads" on public.uploads
  for insert to authenticated
  with check (user_id = auth.uid());
create policy "Owners can update their uploads" on public.uploads
  for update to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy "Admins can update uploads" on public.uploads
  for update to authenticated
  using (public.has_role(auth.uid(), 'admin')) with check (public.has_role(auth.uid(), 'admin'));
create policy "Owners and admins can delete uploads" on public.uploads
  for delete to authenticated
  using (user_id = auth.uid() or public.has_role(auth.uid(), 'admin'));
```

**Signed-out uploads.** Check whether the upload screen works today without signing in. If it does, keep that working: add the policy below, make sure the signed-out flow does not read the row back after inserting it (no `.select()` after `.insert()`), and hide the whole training section from signed-out users (they could not withdraw later, so they cannot opt in). If uploads already need a sign-in, do not add this policy.

```sql
create policy "Signed-out uploads, never for training" on public.uploads
  for insert to anon
  with check (user_id is null and training_consent = false);
```

Any admin page that lists uploads keeps working for admins only. Do not add an admin page in this pass.

**2. Wording and settings.** Create `src/lib/trainingConsent.ts` and hardcode these exactly. Do not reword them. If any of this wording changes later, the version must change too.

```ts
export const TRAINING_CONSENT_VERSION = "2026-11-v1";
export const PRIVACY_EMAIL = "{{PRIVACY_EMAIL}}";
export const TRAINING_REMOVAL_DAYS = 30;

export const CONSENT_COPY = {
  sectionTitle: "Help improve Batting IQ (optional)",
  ageQuestion: "Is the batter in this video 18 or over?",
  ageYes: "Yes, 18 or over",
  ageNo: "No, under 18",
  selfCheckbox: "Help improve Batting IQ: let us use this video to train and test our coaching AI",
  selfHelper: "Optional. Your analysis is exactly the same either way. If the batter isn't you, only tick this if they have agreed.",
  whatLink: "What does this mean?",
  whatPoints: [
    "We keep your video private, in secure storage. Only the Batting IQ team and the qualified coaches who help us can see it.",
    "We use it only to improve how Batting IQ analyses batting: to train our coaching AI and to test that its scores and tips are right.",
    "Qualified cricket coaches may watch it and rate the technique.",
    "If we ever show it to anyone outside the Batting IQ team, we blur the face first.",
    "We never sell it. We never publish it, or use it in adverts, marketing or social media, unless we ask you separately and you say yes.",
    "We don't use it to recognise who anyone is. There is no face recognition.",
    "We keep it for up to 3 years, then delete it.",
    "You can change your mind at any time in Account, under \"My videos and data\". We then stop using it for anything new and remove it from our training sets within 30 days. Training already done can't be undone, but it doesn't contain or show your video.",
    "Saying no changes nothing else in the app.",
  ],
  privacyLink: "Read our privacy notice",
  guardianTitle: "A parent or guardian must agree",
  guardianBody: "The batter is under 18, so we can only use this video to improve Batting IQ if a parent or guardian agrees. This is optional. If they don't, just upload as normal: the analysis is exactly the same.",
  guardianPlayerNote: "Players: ask your parent or guardian to read this with you. You can change your mind at any time, even if they agreed.",
  guardianReveal: "Show the form for a parent or guardian",
  guardianNameLabel: "Parent or guardian's full name",
  guardianEmailLabel: "Parent or guardian's email",
  guardianEmailHelper: "We'll only use this to contact them about this permission.",
  guardianCheckbox: "I am this player's parent or guardian. I have read \"What does this mean?\" and I agree that Batting IQ may use this video to train and test its coaching AI.",
  incentiveBanner: "Share 3 drives for training and get a month of Premium free",
  incentiveSmallPrint: "Optional, for players aged 18 and over. Your analysis is the same either way, and you keep the reward if you change your mind later.",
};
```

Create `src/config/features.ts` (or add to it if it exists):

```ts
export const FEATURES = {
  trainingIncentiveBanner: false, // OFF until Alex decides to run the offer
};
```

**3. Knowing the batter's age.** Create `getProfileAge(profile)` in `src/lib/trainingConsent.ts`.
- Look on the signed-in user's profile for `date_of_birth`, `dob` or `birth_date` (a date), or `age` (a number). Work out the age in whole years on today's date in the Europe/London time zone. Return `null` if none of these exist.
- Do not add a date-of-birth field anywhere. We ask a yes/no question instead, because we need nothing more.

On the upload screen:
- If the profile age is under 18, treat the batter as under 18. Do not show the age question; go straight to the parent or guardian version (step 5).
- Otherwise (an adult or unknown age), show the age question, because an adult may be filming a child. Nothing is selected to start with.

**4. The upload screen: adult version.** Add a section titled `sectionTitle` under the file picker and any existing fields, above the Upload button. Use the app's normal card style, with a muted outline, not a highlight colour.

- The age question: `ageQuestion`, then two radio buttons, `ageYes` and `ageNo`, side by side on wide screens and stacked on phones. Neither is selected to start with. Answering is optional.
- Until it is answered, show nothing else in the section. Do not show a hint telling them to answer.
- If they choose `ageYes`, show:
  - a checkbox, **unticked**, labelled `selfCheckbox`, with `selfHelper` under it in small muted text;
  - a link-style button `whatLink` with a chevron. It expands in place (an accordion, not a pop-up) to show `whatPoints` as a bulleted list, then `privacyLink` linking to the app's privacy page (`/privacy`) if one exists. If there is no privacy page, leave the link out. Set `aria-expanded` on the button. It starts closed every time.
- If they choose `ageNo`, show the parent or guardian version (step 5).
- If they change their answer, clear everything below it: untick the checkbox, empty the guardian fields and close the expandable.

**5. The upload screen: parent or guardian version.** In place of the checkbox, show:
- `guardianTitle` as a small heading, then `guardianBody`, then `guardianPlayerNote` in small muted text.
- The same `whatLink` expandable as in step 4.
- A text button, `guardianReveal`. It reveals the form below. The form is hidden to start with, so the section does not press young players to agree.
- The form:
  - `guardianNameLabel`: text input, `autocomplete="off"`.
  - `guardianEmailLabel`: `type="email"`, `autocomplete="off"`, with `guardianEmailHelper` under it.
  - A checkbox, **unticked**, labelled `guardianCheckbox`.
- Consent counts only when all three are complete:
  - a name of at least 2 characters (after trimming);
  - an email that looks valid (`x@y.z`). If the profile age is under 18, it must also not be the signed-in user's own email.
  - the tick.
- If the name or email is filled in but the box is not ticked, there is no consent. Upload normally and do not save the name or email.
- If the box is ticked but the name or email is missing or invalid, pressing Upload does not upload. Instead:
  - show these messages under the field, in the error colour, linked with `aria-describedby`: "Please enter the parent or guardian's full name.", "Please enter a valid email address." or "Please enter your parent or guardian's email, not your own.";
  - move the focus to the first problem;
  - under the messages, show a text button, **"Upload without sharing for training"**. It unticks the box, clears the fields and uploads normally.

**6. Never nudge.** These rules apply to everyone, whatever their age:
- Every upload starts with nothing selected and nothing ticked. Never remember a previous choice.
- The opt-in never blocks or slows the upload.
- Do not use a pop-up, a reminder, a confirmation, a badge or a highlight colour to encourage opting in.
- Do not change any other wording or any button on the screen depending on the choice.

**7. Incentive banner (OFF by default).** Only when `FEATURES.trainingIncentiveBanner` is `true`, show a slim, muted banner at the top of the training section. It shows `incentiveBanner` in normal text and `incentiveSmallPrint` in small muted text.

Show the banner only when all of these are true:
- the user is signed in;
- `getProfileAge` gives 18 or over;
- the age question is not answered `ageNo`.

Never show it to anyone under 18 or of unknown age, and never on the account page. It must not tick anything or change anything else. Nothing in the app gives out the reward; Alex handles that by hand for now.

**8. Saving the choice.** Find where the upload flow inserts the `uploads` row after the file has gone into Supabase Storage. Add these fields to that same insert. If several videos are uploaded together, each row gets the same choice.

| column | value |
|---|---|
| `user_id` | the signed-in user's id |
| `storage_path` | the exact path the file was stored under in the bucket (the same path passed to `storage.upload`) |
| `uploader_under_18` | `true` if the profile age is under 18 or the answer was `ageNo`; `false` if the answer was `ageYes`; `null` if unanswered |
| `training_consent` | `true` only if the adult box was ticked, or the guardian form was complete and ticked; otherwise `false` |
| `consent_version` | `TRAINING_CONSENT_VERSION` when consent is true, else `null` |
| `consent_given_by` | `"self"` (adult box) or `"parent_guardian"` (guardian form) when consent is true, else `null` |
| `guardian_name`, `guardian_email` | trimmed values, only when `consent_given_by` is `"parent_guardian"`, else `null` |
| `consent_timestamp` | send `new Date().toISOString()` when consent is true, else `null` (the database replaces it with its own time) |

Keep `name`, `file_url` and everything else exactly as today.

- Do not send any of these fields to the analysis API.
- Start the analysis exactly as today, whether or not consent was given.
- **If the insert fails only because of the new fields** (for example a check constraint): retry once with `training_consent: false` and the five consent fields set to `null`, and let the analysis carry on. Then show a toast: "Your video is being analysed, but we couldn't save your choice to share it for training, so we won't use it for training." Never stop or delay the analysis because of the training choice.
- After a successful upload, reset the section to its starting state.

**9. Mobile.**
- Design for a 375 px wide phone first.
- The whole checkbox label is tappable, and every tap target is at least 44 px high.
- The radio buttons stack. The guardian fields are full width and stacked.
- The expandable opens in place and pushes the content down. Nothing overlaps the Upload button, and the page scrolls normally.
- When the phone keyboard opens for the guardian fields, the field in use stays visible.

**10. Tests for Pass 1.** Check each of these in the app and in the Supabase table editor.
1. Adult profile (date of birth 30 years ago), answer `ageYes`, leave the box unticked, upload. The analysis runs as normal. The row has `training_consent = false`, all consent fields `null`, `uploader_under_18 = false`, `user_id` set and `storage_path` set.
2. Same, but tick the box. The row has `training_consent = true`, `consent_version = "2026-11-v1"`, `consent_given_by = "self"`, `consent_timestamp` set by the database, and no guardian fields. `upload_consent_events` has one `given` row.
3. Profile with no date of birth, age question left unanswered. Only the question shows (no checkbox). The upload works, and the row has `uploader_under_18 = null` and `training_consent = false`.
4. Profile aged 15. No age question. The guardian title and text show, and the form stays hidden until `guardianReveal` is pressed. With the form left empty, the upload works and `training_consent = false`.
5. Profile aged 15, form completed with "Jane Smith", "jane@example.com" and the tick. The row has `training_consent = true`, `consent_given_by = "parent_guardian"`, `uploader_under_18 = true` and the guardian name and email.
6. Profile aged 15, guardian email the same as the user's own. The error "Please enter your parent or guardian's email, not your own." shows, and nothing uploads. "Upload without sharing for training" then uploads with `training_consent = false`.
7. Adult profile, answer `ageNo`, tick the guardian box with no name. The name error shows and focus moves to the name field. Nothing uploads until it is fixed or skipped.
8. Answer `ageYes`, tick the box, then change to `ageNo`. The tick is gone and the guardian version shows. Change back: the adult box is unticked.
9. Upload twice in a row. The second upload starts with nothing selected or ticked.
10. With `trainingIncentiveBanner` false (the default), no banner shows anywhere. Set it to true: the banner shows for the 30-year-old profile, not for the 15-year-old, not for a profile with no date of birth, and it disappears when `ageNo` is chosen. Set it back to false.
11. In the SQL editor, as a test user who is not an admin, run `update uploads set training_consent = true, consent_given_by = 'self', consent_version = '2026-11-v1' where id = '<an under-18 row of theirs that has no consent>'`. It is refused with "A parent or guardian must agree".
12. Signed in as user B, user A's uploads cannot be read (the query returns no rows). Signed in as the admin, all rows can be read.
13. On a 375 px phone, the section, the expandable and the guardian form fit with no sideways scrolling, and the Upload button stays reachable.

**Out of scope for this pass:** the account page (Pass 2), any change to the backend call or the analysis, emails to parents or guardians, an admin page, and giving out the incentive.

---

## Pass 2: "My videos and data" on the account page

Pass 1 is in place: the new `uploads` columns, the triggers, `upload_consent_events`, `src/lib/trainingConsent.ts` and `src/config/features.ts` exist. This pass adds one section to the account page and one Supabase Edge Function. Do not change the upload screen, the analysis or the results page. Use UK spelling.

**1. The section.** On the account (profile or settings) page, add a card titled **"My videos and data"**, below the existing profile details. Under the title, in muted text: "These are the videos you've uploaded. You choose whether we may use each one to improve Batting IQ. Your analysis is the same either way."

Load the signed-in user's uploads:
- `select id, name, created_at, training_consent, consent_timestamp, consent_given_by, guardian_name, withdrawn_at, uploader_under_18 from uploads where user_id = <me> order by created_at desc`;
- 20 at a time, with a **"Show more"** button while more remain.

**2. Each video.** One row per upload: a card on phones, a list row on wider screens.
- **Title:** `name` if it is set, else "Untitled video".
- **Date:** `created_at` in en-GB, Europe/London, for example "16 Nov 2026, 14:05".
- **A switch**, labelled **"Use to improve Batting IQ"**, on when `training_consent` is true.
- **Status line** under the switch, in small muted text:
  - On, given by self: "Shared for training since {consent_timestamp as 16 Nov 2026}."
  - On, given by a parent or guardian: "Shared for training since {date}, with agreement from {guardian_name}."
  - Off, with `withdrawn_at`: "You withdrew this on {withdrawn_at date}. It will be out of our training sets by {withdrawn_at + 30 days}."
  - Off, never shared: "Not shared for training."
- **"Delete this video"**: a text button in the destructive colour, with a bin icon, under the status line.

**3. Turning the switch off (withdrawing).**
- Open a dialog titled "Stop using this video for training?", with the body "We'll stop using it for anything new and remove it from our training sets within 30 days. Training already done can't be undone, but it doesn't contain or show your video. Your analysis isn't affected." and two buttons, **"Stop using it"** and **"Cancel"**.
- On "Stop using it", run `update uploads set training_consent = false where id = <id>`. The database stamps `withdrawn_at` itself, so do not send it.
- Wait for success, then show the new status line and the toast "Done. We'll stop using this video for training."
- On "Cancel", leave the switch on.
- Anyone can withdraw, including a player whose parent or guardian agreed.

**4. Turning the switch on (sharing a video that is not shared).**
- **Allowed only when `uploader_under_18 === false`.** Open a dialog titled "Use this video to improve Batting IQ?" It shows the `whatPoints` list, then the `privacyLink`, then two buttons, **"Yes, use this video"** and **"Cancel"**. On yes, run `update uploads set training_consent = true, consent_version = TRAINING_CONSENT_VERSION, consent_given_by = 'self' where id = <id>`. Wait for success, then show the toast "Thanks. This video will help improve Batting IQ."
- **When `uploader_under_18` is `true` or `null`,** the switch is disabled when off. Show this helper instead of the status line: "To share this video for training, upload it again and answer the age question. For players under 18, a parent or guardian must agree."
- Do not show the incentive banner on this page, ever.

**5. Deleting a video.**
- "Delete this video" opens a dialog titled "Delete this video?" with the body "This permanently deletes the video from Batting IQ, and it can't be undone. If you shared it for training, it will be out of our training sets within 30 days. Your past analysis of it stays in your history." and two buttons, **"Delete video"** (destructive) and **"Cancel"**.
- On confirm, call the Edge Function `delete-upload` (step 6) with `{ upload_id }`.
- While it runs, the button shows a spinner and the row is dimmed.
- On success, remove the row from the list and show the toast "Video deleted."
- On failure, keep the row and show the toast "We couldn't delete this video. Please try again, or email {PRIVACY_EMAIL}."

**6. Edge Function `delete-upload`.** Create it in Supabase. It runs on the server with the service role key, which must never reach the browser.
- Accept `POST { upload_id }` with the user's `Authorization` header. If there is no signed-in user, return 401.
- Using a client made with the anon key and the user's token (so row level security applies), read `id, user_id, storage_path, file_url` for that id. If no row comes back, return 404. That means it is not theirs, or it is already gone. Admins can read any row, so admins can delete any row.
- Work out the path: `storage_path`, or else take it from `file_url` with the same pattern as the migration (`/storage/v1/object/(public|sign|authenticated)/<bucket>/<path>`). If there is still no path, return 422 and delete nothing.
- With a service-role client, remove the file from the bucket the upload screen uses. Use the same bucket name constant as the upload code. If removal fails for any reason other than "not found", return 500 and keep the row.
- Then delete the `uploads` row with the service-role client. The Pass 1 trigger logs a `deleted` event by itself.
  - If another table has a foreign key to `uploads.id`, set that column to `null` first, so the user's past analyses stay.
  - If the row delete fails, return 500.
- Return `200 { deleted: true }`.

**7. Footer of the section** (small muted text):
- "Videos you uploaded before this page was added aren't listed here, and we never use them for training. To have them deleted, email {PRIVACY_EMAIL}."
- "To get a copy of your data or close your account, email {PRIVACY_EMAIL}." Make the email address a `mailto:` link, and add a link to `/privacy` if that page exists.

**8. Empty, loading and error states.**
- **Loading:** three skeleton rows.
- **Empty:** "No videos yet. When you upload a drive, it shows here, and you can choose whether we may use it to improve Batting IQ." and a button, **"Upload a video"**, that goes to the upload screen.
- **Load error:** "We couldn't load your videos. Check your connection and try again." and a **"Try again"** button. Do not show the footer links twice.
- **Switch error** (the update fails, or the database refuses it): put the switch back where it was, and show the toast "We couldn't save that change. Please try again." If the database message contains "parent or guardian", show that message instead.
- **Signed out:** do not show the section.

**9. Mobile.**
- Design for a 375 px wide phone first. Each video is a full-width card: the title and date on top, the switch row under them (label left, switch right, at least 44 px tall), then the status line, then "Delete this video".
- Long titles wrap; nothing is cut off. The dialogs are bottom sheets on phones and centred dialogs on wider screens. Buttons are full width in bottom sheets.

**10. Tests for Pass 2.**
1. A user with no uploads sees the empty state and the "Upload a video" button.
2. A user with 25 uploads sees 20, then all 25 after "Show more", newest first.
3. A video shared by self shows the switch on and "Shared for training since …". Turn it off and confirm. The row now has `training_consent = false` and `withdrawn_at` set, the consent version and time are kept, the status line reads "You withdrew this on {today}. It will be out of our training sets by {today + 30 days}.", and `upload_consent_events` has a `withdrawn` row.
4. Turn it off, then press Cancel: nothing changes.
5. The same video (adult, `uploader_under_18 = false`) can be turned back on through the dialog. Afterwards `withdrawn_at` is `null`, `consent_timestamp` is new, and a new `given` event is logged.
6. A video uploaded with parent or guardian agreement shows "… with agreement from Jane Smith." The player can turn it off. Once off, the switch is disabled and shows the "upload it again" helper.
7. A video with `uploader_under_18 = null` and no consent shows the switch disabled, with the helper.
8. Delete a video. The file is gone from the storage bucket, the row is gone, a `deleted` event is logged if it had ever been shared, and the toast shows. The analysis of that video is still in the history.
9. Make the function fail (for example, rename the bucket constant in a test branch). The row stays and the error toast shows.
10. User B calls `delete-upload` with user A's upload id and gets 404. Nothing is deleted.
11. With the network turned off, the load error shows. "Try again" works once the network is back.
12. On a 375 px phone there is no sideways scrolling, the switches are easy to tap, and the dialogs open as bottom sheets.

**Out of scope for this pass:** downloading or playing videos from this page, deleting analyses, deleting the account, admin tools, guardian confirmation emails and the incentive.

---

## Notes for Alex and the backend (not for Lovable)

- **Is the upload bucket private?** The explanation promises the video is "stored privately". In Supabase, open Storage, then the upload bucket. If it is marked **Public**, anyone with a file's link can watch it. Making it private may break the analysis call if the backend downloads videos by public URL, so treat that as a separate, tested change (signed URLs) before the opt-in goes live. Checked 16 Nov 2026: plain public links to `battingiq-uploads` return HTTP 400, so the bucket is private today. Keep it that way, and still confirm in the dashboard before launch. (`phase4_app_uploads.py`'s manifest mode only works on a public bucket, which is why it failed.)
- **Training scripts** read `uploads` with the service role key, which ignores row level security. So every script must filter exactly as in `docs/legal/training_data_rules.md`. `phase4_app_uploads.py` currently downloads app uploads with no consent filter. Those clips were uploaded before the opt-in existed, so they must not be used for training, testing or coach rating.
- **Storyboard frames** that the backend saves under `analysis-media/` in the media bucket show the batter too. Deleting a video in the app does not remove them. Either delete them with the video (a backend change), or say so in the privacy notice. The draft notice says so for now.
- **Guardian confirmation.** Today a junior could type any name and email. Before the club channel opens, add an email to the guardian ("Please confirm you agreed"), and use under-18 clips only once it has been confirmed. Until then, treat under-18 clips with care (see the rules).
- **Incentive.** If `trainingIncentiveBanner` is ever switched on, give out the Premium month by hand. People who withdraw later keep it. The terms say so.
- **Launch.** Do not switch on Pass 1 until a solicitor has reviewed `docs/legal/training_data_terms_draft.md` and the privacy page carries the training section. Upload screen wording version `2026-11-v1` must match the published notice.
