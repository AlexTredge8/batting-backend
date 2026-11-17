# Lovable prompt: "Your drill plan" (a 3-week plan under the priority fix)

Five pastes, plus two short templates. Paste them into Lovable one at a time, in this order, and check each before the next:

1. **Pass 0a** now (one line on the pricing card).
2. **Pass 1** and **Pass 2**: the drill data (57 drills, in two parts because it is long).
3. **Pass 3**: the plan logic, saving and the "Your drill plan" section.
4. **Pass 0b** only when the plan is live (after Alex's sign-off and the tests).
5. The two short follow-up templates at the end, whenever Alex signs off drills or a demo video is ready.

Each pass is everything between its two lines. The note for Alex at the very end is not for Lovable.

Sources: drills from `docs/phase6/drill_library.json` (this prompt's data is made from it; if the JSON changes, change the data here too). Rules and copy from `docs/phase6/drill_plan_rules.md`. It builds on `docs/lovable/results_tips_and_sharing_prompt.md` Passes 1 and 2 (the priority fix card, `getPriorityFix`, "Film it right", "Your progress" and the share card), and on `docs/lovable/upload_training_consent_prompt.md` (`getProfileAge`, `user_roles`). Legal points: `docs/legal/ALEX_DECISIONS.md` and `docs/legal/consumer_and_safeguarding_review.md` Issue 4.

---

## Pass 0a: the pricing line (paste now)

Search the whole project for the text "Personalised drill plan with coaching videos" (it is on the pricing card, and may also be on the landing page or in a features list). Change every copy of it to **"Drills to fix your priority fault"**. Change nothing else: not the price, the layout, the other lines or any other page. Keep UK spelling.

---

## Pass 0b: the pricing line again (paste only after the plan is live)

Search the whole project for the text "Drills to fix your priority fault" on the pricing card (and anywhere else Pass 0a changed it). Change every copy of it to **"Personalised drill plan"**. Do not mention coaching videos anywhere until a demo video exists for every drill. Change nothing else.

---

## Pass 1: drill data, part 1

Create two new files. Copy the code below **exactly**: do not reword, shorten, reorder or "improve" any text, and keep UK spelling. Do not change any page yet. Do not call the backend.

**1. `src/data/drillTypes.ts`:**

```ts
export type DrillFaultId =
  | "F00" | "F01" | "F02" | "F03" | "F04" | "F05" | "F06" | "F07" | "F08" | "F09"
  | "F10" | "F11" | "F12" | "F13" | "F15" | "F16" | "F17" | "F18" | "F19";

export type DrillLevel = "beginner" | "club" | "advanced";

// pending = not yet signed off by Alex; ok = Alex signed it off; changed = Alex asked for a change (not yet made); removed = do not use.
export type Signoff = "pending" | "ok" | "changed" | "removed";

export interface Drill {
  id: string;            // e.g. "F09-C": fault, then B (beginner), C (club) or A (advanced)
  fault: DrillFaultId;
  level: DrillLevel;
  name: string;
  purpose: string;       // one line: what it fixes and why
  equipment: string[];
  setup: string;
  steps: string[];       // 3 to 6 short steps
  sets: number;
  reps: number;
  unit: string;          // "balls", "throws", "reps", "strides", "calls", "slow reps"
  timeMinutes: number;   // about how long one round of the sets takes
  successCheck: string;  // starts "You know it's working when"
  progression: string;   // the next step, used as "Make it harder"
  safety: string;        // this drill's own safety line
  soloOk: boolean;
  videoUrl: string | null; // null until the demo video is uploaded
  signoff: Signoff;
}

export const DRILL_LEVELS: DrillLevel[] = ["beginner", "club", "advanced"];

export const DRILL_LEVEL_LABELS: Record<DrillLevel, { label: string; description: string }> = {
  beginner: { label: "Beginner", description: "New to cricket, or new to batting with a proper ball." },
  club: { label: "Club", description: "I play for a school or club team." },
  advanced: { label: "Advanced", description: "I play first-team adult cricket, or county or academy age-group cricket." },
};

export const DRILL_SAFETY =
  "Warm up first. Make sure there is space around you and nobody stands within reach of your swing. Use a tennis ball or soft ball for partner drills, with the feeder standing to the side, not in line with the shot. With a hard ball, always wear a helmet with a faceguard and proper protective kit. Stop if anything hurts.";

export const DRILL_SAFETY_JUNIOR =
  "Players under 18: do these drills with an adult supervising, and always wear a helmet with a faceguard when batting against a hard ball, in nets and in matches.";
```

If `src/lib/coaching.ts` already exports `DRILL_SAFETY` or `DRILL_SAFETY_JUNIOR`, make `coaching.ts` re-export these two from `src/data/drillTypes.ts` instead (`export { DRILL_SAFETY, DRILL_SAFETY_JUNIOR } from "@/data/drillTypes";`), so there is only one copy of each text. If its old text is different, this new text wins.

**2. `src/data/drillLibraryPart1.ts`** (30 drills: F01, F02, F17, F13, F03, F19, F15, F04, F05, F06):

```ts
import type { Drill } from "./drillTypes";

export const DRILLS_PART_1: Drill[] = [
  { id: "F01-B", fault: "F01", level: "beginner", name: "Still head, ball drop", purpose: "Keeps your head still and your eyes level before the ball comes, because a moving head gives you a blurred view of the ball.", equipment: ["Bat", "Tennis ball", "Partner"], setup: "Take your stance. Your partner stands 3 metres away, a little to your off side, and holds a tennis ball at shoulder height.", steps: ["Look at the ball in your partner's hand.", "Keep your head still and your eyes level.", "Your partner drops the ball when they choose.", "Keep your head still until the ball bounces.", "Play a soft forward defence, so the ball stops near your feet."], sets: 2, reps: 10, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when your partner sees no head movement before the bounce on 8 of 10 balls.", progression: "Your partner waits a different time before each drop, so you cannot guess when the ball comes.", safety: "Use a tennis ball. Play soft defence shots only, so the ball stays near your feet.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F01-C", fault: "F01", level: "club", name: "Run-up watch", purpose: "Trains a still head while the bowler runs in, because that is when most heads start to move.", equipment: ["Bat", "Tennis balls", "Partner", "Phone, propped up"], setup: "Your partner stands 8 metres away, to one side of the line of the shot. Prop a phone 2 metres behind your partner at chest height, pointing at you.", steps: ["Start the phone recording.", "Your partner takes five walking steps in and throws a tennis ball underarm.", "Keep your head still and your eyes level during the steps.", "Play a straight drive or a forward defence.", "After 6 balls, watch the clip. Check your head against something fixed behind you, such as a fence post."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when your head stays level with the same fixed point behind you until the ball leaves your partner's hand.", progression: "Your partner jogs in and throws a little faster, so the ball arrives sooner.", safety: "Use a tennis ball. The thrower stands 6 to 8 metres away, to one side of the line of the shot, and watches every ball. Put the phone where the ball and the bat cannot reach it.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F01-A", fault: "F01", level: "advanced", name: "Trigger, then still", purpose: "Lets you keep your trigger movement but finish it before the ball is released, so your head is still at the moment that matters.", equipment: ["Bat", "Tennis balls", "Partner", "Phone, propped up"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. Prop the phone behind your partner, out of the line of the ball.", steps: ["Make your normal trigger movement as your partner's arm comes up.", "Finish the trigger before the ball leaves the hand.", "Keep your head still and your eyes level as the ball is released.", "Play the shot that suits the ball, or leave it.", "After 8 balls, watch 3 clips. Check that your head stops before the release."], sets: 2, reps: 8, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the clips show your trigger finished and your head still at release, and the ball seems to arrive later.", progression: "Your partner varies the pace and the length, so you must still finish the trigger early against quicker balls.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Put the phone where the ball and the bat cannot reach it.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F02-B", fault: "F02", level: "beginner", name: "Ready-position hops", purpose: "Sets an even, relaxed stance, so you can move forward or back with equal ease.", equipment: ["Bat"], setup: "Stand in an open space in your normal stance, with your feet about shoulder-width apart.", steps: ["Take your stance.", "Do two small hops on the spot.", "Land with your weight even on both feet and your knees soft.", "Hold still for 3 seconds.", "Check that you could step forward or back without a wobble."], sets: 2, reps: 10, unit: "reps", timeMinutes: 6, successCheck: "You know it's working when you land in the same balanced stance every time, with no shuffle of the feet.", progression: "After the hold, step straight into a shadow forward defence, then a shadow back-foot defence.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F02-C", fault: "F02", level: "club", name: "Forward or back call", purpose: "Tests your stance under a call, because a balanced stance lets you go either way quickly.", equipment: ["Bat", "Partner, or a phone playing calls you recorded"], setup: "Take your stance. Your partner stands 3 metres away, out of reach of your bat. Or record yourself saying \"forward\" and \"back\" in a random order, with gaps, and play it back.", steps: ["Hold your stance with your weight even.", "On \"forward\", play a shadow front-foot drive.", "On \"back\", play a shadow back-foot defence.", "Return to your stance and hold it for 2 seconds.", "Wait for the next call."], sets: 2, reps: 10, unit: "calls", timeMinutes: 8, successCheck: "You know it's working when you move as quickly forward as back, and you never need a small step before you go.", progression: "Your partner calls later, just as they pretend to let go of the ball.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F02-A", fault: "F02", level: "advanced", name: "Freeze on release", purpose: "Checks your balance at the moment the ball is released, when balance matters most.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot.", steps: ["Your partner runs in and either throws, or fakes the throw and calls \"freeze\".", "On a throw, play the right shot.", "On \"freeze\", stop still at once.", "On a freeze, lift one foot a little and check that you stay balanced.", "Mix 8 throws and 4 freezes in a random order."], sets: 1, reps: 12, unit: "balls", timeMinutes: 10, successCheck: "You know it's working when you can lift either foot on every freeze without falling to one side.", progression: "Your partner throws a little faster and fakes later, so your balance must hold under more pace.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F17-B", fault: "F17", level: "beginner", name: "Line up on off stump", purpose: "Lines up your head, hands and feet over off stump as the ball is released, so you start on the line of the ball.", equipment: ["Bat", "Stump or cone", "Partner"], setup: "Put a stump or a cone on your off-stump line at the crease. Your partner stands 10 metres away, where a bowler lets go of the ball.", steps: ["Take your stance.", "Set your feet first, then your hands, then your head.", "Your partner pretends to bowl.", "As their arm comes over, line up your head, hands and feet over off stump.", "Your partner calls \"yes\" or \"no\" after each rep."], sets: 1, reps: 10, unit: "reps", timeMinutes: 6, successCheck: "You know it's working when your partner calls \"yes\" on 8 of 10 reps.", progression: "Your partner throws a soft underarm tennis ball after the check, and you play a forward defence.", safety: "No ball is needed. If you add a ball, use a tennis ball thrown underarm from the side of the pitch.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F17-C", fault: "F17", level: "club", name: "Phone line-up check", purpose: "Lets you check your line-up on your own, with the phone as the bowler's eye.", equipment: ["Bat", "Phone, propped up", "Cone or stump"], setup: "Prop the phone at chest height 10 metres away, straight down the pitch. Put a cone on your off-stump line at the crease.", steps: ["Start the phone recording.", "Take your stance.", "Say \"now\" when an imaginary bowler lets go.", "Hold your position for 2 seconds after \"now\".", "Do 10 reps, then watch the clip.", "Pause on each \"now\" and check your head, hands and feet against the cone."], sets: 1, reps: 10, unit: "reps", timeMinutes: 8, successCheck: "You know it's working when your head, hands and feet stack over the cone in the paused clip on most reps.", progression: "Add a shadow drive after each hold, and check that your head stays on the line.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F17-A", fault: "F17", level: "advanced", name: "Line up and leave", purpose: "Joins the line-up to judging the ball, so you know where off stump is and leave well.", equipment: ["Bat", "Tennis balls", "Partner", "Cone"], setup: "Put a cone on your off-stump line at the crease. Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They mix balls on the stumps with balls wide of off stump.", steps: ["Line up your head, hands and feet over off stump as your partner's arm comes over.", "Drive the balls that are full and on the stumps.", "Leave the balls that are wide of off stump.", "Hold your line-up for one second on every leave.", "Your partner calls \"yes\" if you were lined up at release."], sets: 2, reps: 8, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you get 8 \"yes\" calls in 10 balls and leave the wide ones with confidence.", progression: "Your partner adds balls just outside off stump, so the leave is a closer decision.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F13-B", fault: "F13", level: "beginner", name: "Lift on \"now\"", purpose: "Starts your backlift earlier, so the swing has time to flow and is not rushed.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot.", steps: ["Take your stance.", "Your partner says \"now\" as their throwing arm starts to swing.", "Start to lift your hands on \"now\".", "Play a straight drive or a forward defence.", "Check that your bat was up before the ball bounced."], sets: 1, reps: 10, unit: "throws", timeMinutes: 8, successCheck: "You know it's working when your bat is up and ready before the ball bounces, with no rush.", progression: "Your partner stops saying \"now\", and you start the lift on the swing of their arm alone.", safety: "Use a tennis ball. The thrower stands 6 to 8 metres away, to one side of the line of the shot, and watches every ball.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F13-C", fault: "F13", level: "club", name: "Lift with the gather", purpose: "Times your backlift to the bowler's action, not to the ball, so it starts at the same moment every time.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner stands 10 metres away, to one side of the line of the shot, and bowls a tennis ball with a short, gentle action.", steps: ["Do 3 shadow reps with no ball first.", "Watch your partner's bowling action.", "Start to lift your hands as their front arm goes up.", "Have the bat up by the time their bowling arm comes over.", "Play the shot that suits the ball."], sets: 1, reps: 10, unit: "balls", timeMinutes: 10, successCheck: "You know it's working when your backlift starts at the same point in the action every ball, and the shot feels unhurried.", progression: "Your partner bowls a little quicker, and you keep the same early start.", safety: "Use a tennis ball. The bowler bowls gently from at least 10 metres and moves out of the line of the shot after letting go.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F13-A", fault: "F13", level: "advanced", name: "Early lift, quicker ball", purpose: "Keeps an early, unhurried backlift against quicker balls, where late hands cost the most.", equipment: ["Bat", "Tennis balls", "Partner", "Phone, propped up"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. Prop the phone side-on to you, out of the line of the ball.", steps: ["Start the lift as your partner's arm comes up.", "Have the bat up by the time the ball bounces.", "Swing in one movement into the shot.", "Drive the full balls and defend the rest.", "After every 6 balls, watch one clip. Check that the bat was up before the bounce."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the clips show the bat up before the bounce, even on the quicker balls.", progression: "Your partner adds a quicker ball every third throw, with no warning.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Put the phone where the ball and the bat cannot reach it.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F03-B", fault: "F03", level: "beginner", name: "Mirror backlift", purpose: "Teaches a backlift that goes up and back, close to your body, with the top hand in charge.", equipment: ["Bat", "Mirror or a dark window"], setup: "Stand side-on to a mirror, about 2 metres away, so you can see yourself in your stance.", steps: ["Take your stance.", "Lift your hands slowly up and back, close to your body.", "Keep your top hand in charge.", "Check in the mirror that your hands stay near your back hip.", "Lower the bat and repeat."], sets: 2, reps: 10, unit: "reps", timeMinutes: 6, successCheck: "You know it's working when your hands stay close to your body on every lift. The bat pointing towards the slips is fine.", progression: "After each lift, swing through a slow shadow drive, and check that your hands come down close to your body too.", safety: "Swing slowly. Keep the bat at least one bat length away from the mirror or window.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F03-C", fault: "F03", level: "club", name: "Top-hand tee drives", purpose: "Builds a strong top hand, which keeps the backlift close to your body and the bat straight.", equipment: ["Bat (a light or short one is best)", "Tennis balls", "Tee or tall cone"], setup: "Put a tennis ball on a tee or a tall cone next to where your front foot lands. Face a net or open space.", steps: ["Hold the bat in your top hand only. Put your bottom hand behind your back.", "Lift the bat up and back, close to your body.", "Step in and drive the ball gently off the tee.", "Do 8 balls with the top hand only.", "Then do 8 balls with both hands, with the top hand still in charge."], sets: 2, reps: 8, unit: "balls", timeMinutes: 10, successCheck: "You know it's working when the two-handed drives feel led by the top hand and the ball goes straight.", progression: "Do the same with soft drop feeds instead of the tee.", safety: "Hit into a net or towards open space, away from people, windows and cars. Use a light bat. Stop if your wrist or forearm hurts.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F03-A", fault: "F03", level: "advanced", name: "Net-side lift", purpose: "Shows you at once if your hands push out, because the bat touches the netting.", equipment: ["Bat", "Net lane", "Tennis balls", "Partner"], setup: "In a net lane, take guard so the off-side netting is about one metre in front of your toes. Your partner throws underarm from 8 metres, from the leg side of the lane.", steps: ["Do 5 slow shadow backlifts. The bat must not touch the netting.", "Your partner throws tennis balls on middle stump.", "Lift up and back, close to your body.", "Drive straight back down the lane.", "If the bat touches the netting, do 2 slow shadow lifts before the next ball."], sets: 2, reps: 8, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the bat never touches the netting and your drives go straight back down the lane.", progression: "Your partner throws overarm with one bounce, a little quicker.", safety: "Use a tennis ball. The thrower stands on the leg side of the lane, at least 8 metres away.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F19-B", fault: "F19", level: "beginner", name: "Full or short?", purpose: "Trains you to pick the length early, so you drive only balls that are pitched up.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot. They throw some full and some short, in a random order.", steps: ["Watch where the ball bounces.", "Say \"full\" or \"short\" out loud as it bounces.", "Drive the full balls.", "Go back to the short balls, or leave them."], sets: 1, reps: 12, unit: "throws", timeMinutes: 8, successCheck: "You know it's working when you choose the right shot on 10 of 12 balls.", progression: "Your partner adds balls of a middle length, so the choice is harder.", safety: "Use a tennis ball. The thrower stands 6 to 8 metres away, to one side of the line of the shot, and watches every ball.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F19-C", fault: "F19", level: "club", name: "Cone on a length", purpose: "Gives you a clear marker for a ball you can drive, so you learn which lengths to drive.", equipment: ["Bat", "Tennis balls", "Partner", "Cone or shoe"], setup: "Put a cone about 2 metres in front of the batting crease, at the side of the pitch so the ball cannot hit it. Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot.", steps: ["Watch where the ball bounces, compared with the cone.", "If it bounces between the cone and you, drive it.", "If it bounces before the cone, go back and defend, or leave it.", "Your partner calls \"right\" or \"wrong\" after each ball."], sets: 1, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when your partner calls \"right\" on 10 of 12 balls.", progression: "Move the cone 30 centimetres closer to you, so only the fuller balls count as balls to drive.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F19-A", fault: "F19", level: "advanced", name: "Drive, defend or leave", purpose: "Joins length and line choices, so you drive the right ball and a short one does not catch you out.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They mix full, short and wide balls.", steps: ["Drive the full balls on the stumps.", "Go back and defend the short balls.", "Leave the wide balls outside off stump.", "Score 1 point for each right choice.", "Play 12 balls and keep your score."], sets: 1, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you score 10 or more out of 12, two sessions in a row.", progression: "Your partner throws a little quicker, so you have less time to choose.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F15-B", fault: "F15", level: "beginner", name: "Stride along the line", purpose: "Teaches you to move forward towards the ball, not across the crease, so your head stays in line.", equipment: ["Bat", "Spare bat or a stump", "Tennis balls and a partner (optional)"], setup: "Lay a spare bat or a stump on the ground from your front toe, pointing straight down the pitch.", steps: ["Take your stance.", "Stride forward alongside the bat on the ground, not across it.", "Hold the stride for 2 seconds.", "Do 10 strides.", "If you have a partner, add 10 soft drop feeds and drive each one."], sets: 2, reps: 10, unit: "strides", timeMinutes: 8, successCheck: "You know it's working when your front foot lands next to the line every time and your head stays above it.", progression: "Do the same with underarm throws from 8 metres.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing. For drop feeds, use a tennis ball. The feeder kneels on your leg side, out of the path of the bat.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F15-C", fault: "F15", level: "club", name: "Back-foot cone", purpose: "Stops your first movement going across the crease, which pulls your head out of line.", equipment: ["Bat", "Cone or shoe", "Tennis balls", "Partner"], setup: "Put a cone just outside your back foot, on the off side. Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot.", steps: ["Take your stance.", "Keep your back foot away from the cone.", "Step forward towards each ball.", "Drive the ball straight back.", "If you touch the cone, do 2 slow shadow strides before the next ball."], sets: 2, reps: 8, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you do not touch the cone and the ball goes back past your partner, not to the leg side.", progression: "Your partner aims at middle and leg stump, where a move across hurts most.", safety: "Use a tennis ball. The thrower stands 6 to 8 metres away, to one side of the line of the shot, and watches every ball.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F15-A", fault: "F15", level: "advanced", name: "Straight-ball throws", purpose: "Tests your movement against straight balls, which are the ones that punish a step across.", equipment: ["Bat", "Tennis balls", "Partner", "Two cones"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They aim at middle and leg stump. Put two cones 3 metres apart, straight down the pitch and 15 metres from you, as a target.", steps: ["Move forward towards the ball, not across.", "Keep your head over the line of the ball.", "Drive the ball back between the cones.", "Score 1 point for each drive between the cones.", "Play 12 balls."], sets: 1, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you score 8 or more out of 12 and nothing goes off the inside edge to the leg side.", progression: "Your partner adds the odd ball outside off stump, and you keep the same movement.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Nobody stands near the target cones.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F04-B", fault: "F04", level: "beginner", name: "Head first", purpose: "Teaches your head and front shoulder to lead you towards the ball, so your front foot follows.", equipment: ["Bat", "Tennis ball", "Cone or tee"], setup: "Put a tennis ball on a cone or a tee, one long stride in front of your front foot.", steps: ["Take your stance and look at the ball on the cone.", "Move your head and front shoulder towards the ball first.", "Let your front foot follow.", "Stop in a shadow drive with your head over the ball. Do not hit the ball.", "Return to your stance."], sets: 1, reps: 10, unit: "slow reps", timeMinutes: 6, successCheck: "You know it's working when your head arrives over the ball before your front foot lands.", progression: "Drive the ball gently off the cone, still leading with your head.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F04-C", fault: "F04", level: "club", name: "Lead to the drop", purpose: "Joins the head-first movement to a real ball, so it holds up when the ball moves.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Drop feed: your partner kneels on your leg side, out of the path of the bat, and drops a tennis ball so it bounces in front of your front pad.", steps: ["Watch the ball in your partner's hand.", "As it drops, move your head and front shoulder towards it.", "Let your front foot follow.", "Drive the ball along the ground.", "Your partner says \"head\" if your foot moved first."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when your partner does not need to say \"head\" on most balls.", progression: "Your partner drops the ball a little further away, so your head must travel further.", safety: "Use a tennis ball. The feeder kneels on your leg side, out of the path of the bat, and leans back after each drop.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F04-A", fault: "F04", level: "advanced", name: "Shoulder to the pitch", purpose: "Keeps your head and shoulder leading against throws that make you reach forward.", equipment: ["Bat", "Tennis balls", "Partner", "Phone, propped up"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They pitch the ball full, on and just outside off stump. Prop the phone side-on, out of the line of the ball.", steps: ["Point your front shoulder at the ball as it bounces.", "Let your head follow the shoulder towards the ball.", "Land your front foot after your head has started to move.", "Drive along the ground.", "After 6 balls, watch one clip. Your head should be over your front knee at contact."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the clip shows your head over your front knee at contact on most drives.", progression: "Your partner adds the odd fuller, wider ball that makes you reach further.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Put the phone where the ball and the bat cannot reach it.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F05-B", fault: "F05", level: "beginner", name: "Step to the marker", purpose: "Lands your front foot on the line of the ball, or just inside it, so the bat has room to come through.", equipment: ["Bat", "Cone or shoe", "Tennis balls and a partner (optional)"], setup: "Put a cone or a shoe where a full ball would bounce, just outside off stump.", steps: ["Take your stance.", "Stride so your front foot lands next to the cone, on its line or just inside it.", "Hold for 2 seconds and check the gap between your foot and the cone.", "Do 10 strides.", "If you have a partner, add 10 soft drop feeds that bounce near the cone."], sets: 2, reps: 10, unit: "strides", timeMinutes: 8, successCheck: "You know it's working when your front foot lands next to the cone, not past it or well inside it, on 8 of 10 strides.", progression: "Move the cone to a different line for each set: off stump, then middle stump.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing. For drop feeds, use a tennis ball. The feeder kneels on your leg side, out of the path of the bat.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F05-C", fault: "F05", level: "club", name: "Moving marker", purpose: "Teaches your front foot to go to the line of the ball wherever it is, not to the same place every time.", equipment: ["Bat", "Cone", "Tennis balls", "Partner"], setup: "Your partner puts the cone on a different line each time: off stump, outside off stump or middle stump. Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot. They aim to bounce the ball near the cone.", steps: ["Look at where the cone is.", "Take your stance.", "Stride so your front foot lands next to the line of the cone.", "Drive the ball.", "Your partner moves the cone for the next ball."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when your front foot matches the line of the cone on most balls and the ball comes off the middle of the bat.", progression: "Your partner throws without the cone, and you read the line from the ball alone.", safety: "Use a tennis ball. The thrower stands 6 to 8 metres away, to one side of the line of the shot, and watches every ball. Your partner moves the cone only when you stand still.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F05-A", fault: "F05", level: "advanced", name: "Read the line", purpose: "Moves your front foot to the line of real throws, so the bat swings through on the line.", equipment: ["Bat", "Tennis balls", "Partner", "Phone, propped up"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They throw on different lines. Prop the phone behind your partner at chest height, out of the line of the ball.", steps: ["Watch the ball from the hand.", "Step your front foot to the line of the ball, or just inside it.", "Drive along the ground.", "After 6 balls, watch the clip.", "Check your front foot against the line of the ball at contact."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the clip shows your front foot on or just inside the line of the ball on 8 of 12 balls.", progression: "Your partner throws a little quicker, so you must read the line earlier.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Put the phone where the ball and the bat cannot reach it.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F06-B", fault: "F06", level: "beginner", name: "Bent-knee hold", purpose: "Teaches your front knee to bend and take your weight, so the weight does not stay on your back foot.", equipment: ["Bat"], setup: "Stand in an open space in your stance.", steps: ["Shadow a front-foot drive.", "Bend your front knee as you land, so it is over your toes.", "Let your back heel come off the ground.", "Hold the finish for 3 seconds.", "Return to your stance."], sets: 1, reps: 10, unit: "reps", timeMinutes: 6, successCheck: "You know it's working when you can hold each finish for 3 seconds with your front knee bent and no wobble.", progression: "Do the same off a tennis ball on a tee.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F06-C", fault: "F06", level: "club", name: "Long-stride tee drive", purpose: "Makes you bend your front knee to reach the ball, so your weight goes into the shot.", equipment: ["Bat", "Tennis balls", "Tee or tall cone"], setup: "Put a tennis ball on a tee one long stride in front of your stance, next to where your front foot will land. Face a net or open space.", steps: ["Take your stance.", "Take a long stride towards the tee.", "Bend your front knee so it is over your toes.", "Drive the ball along the ground.", "Hold the finish for 2 seconds."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when the ball stays low and your front knee is bent over your toes in the hold.", progression: "Lower the tee, or use a smaller cone, so you must bend more.", safety: "Hit into a net or towards open space, away from people, windows and cars.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F06-A", fault: "F06", level: "advanced", name: "Low-ball throws", purpose: "Trains a bent front knee against full balls that stay low, where a stiff leg makes you hit the ball in the air.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They pitch the ball full, so it stays below knee height.", steps: ["Stride towards the ball.", "Bend your front knee to get your head down to the ball.", "Drive along the ground.", "Hold the finish for 2 seconds.", "Your partner calls \"air\" if the ball goes up."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when your partner calls \"air\" on no more than 2 balls in 12.", progression: "Your partner throws a little faster and fuller.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
];
```

**Check:** the project builds, and `DRILLS_PART_1.length` is 30. Nothing on screen changes.

---

## Pass 2: drill data, part 2

Pass 1 is in place. Create two more files. Copy the code **exactly**, as in Pass 1. Do not change any page yet.

**1. `src/data/drillLibraryPart2.ts`** (27 drills: F18, F16, F07, F08, F09, F10, F11, F12, F00):

```ts
import type { Drill } from "./drillTypes";

export const DRILLS_PART_2: Drill[] = [
  { id: "F18-B", fault: "F18", level: "beginner", name: "Back-foot push", purpose: "Teaches your back foot to push you forward, so your weight drives into the shot.", equipment: ["Bat"], setup: "Stand in an open space in your stance.", steps: ["Shadow a front-foot drive.", "Push your back foot against the ground as you stride.", "Let your weight move forward.", "Finish with your back heel up and only the toe on the ground.", "Hold for 2 seconds."], sets: 1, reps: 10, unit: "reps", timeMinutes: 6, successCheck: "You know it's working when every finish has your back heel up, without you lifting it on purpose.", progression: "Do the same off a tennis ball on a tee.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F18-C", fault: "F18", level: "club", name: "Toe-drag finish", purpose: "Gives you a clear sign that your back foot pushed: the back toe drags forward a little.", equipment: ["Bat", "Tennis balls", "Partner", "Chalk or tape"], setup: "Mark a line on the ground at the tip of your back toe with chalk or tape. Drop feed: your partner kneels on your leg side, out of the path of the bat, and drops a tennis ball so it bounces in front of your front pad.", steps: ["Take your stance with your back toe on the line.", "Push with your back foot as you stride.", "Drive the drop-fed ball.", "Let your back toe drag forward past the line.", "Check the gap after each ball."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when your back toe finishes in front of the line on most balls.", progression: "Use underarm throws from 8 metres instead of drop feeds.", safety: "Use a tennis ball. The feeder kneels on your leg side, out of the path of the bat, and leans back after each drop.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F18-A", fault: "F18", level: "advanced", name: "Drive and walk through", purpose: "Keeps your back foot driving through the shot against throws, so you carry your weight into the ball.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They pitch the ball full.", steps: ["Push off your back foot as you stride.", "Drive the ball along the ground.", "Let your back foot come through into one step forward after the shot.", "Stop and stand tall after the step.", "Walk back to your crease."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the step through happens on its own, without you thinking about it.", progression: "Your partner throws a little faster, so the push must come sooner.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Check that the space in front of you is clear before you step through.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F16-B", fault: "F16", level: "beginner", name: "Nose over toes", purpose: "Moves your weight towards the ball, so your head and chest are over your front foot when you hit it.", equipment: ["Bat", "Partner, or a phone propped up"], setup: "Stand in an open space in your stance. Your partner stands side-on to you, 3 metres away, out of reach of your bat. Or prop a phone side-on and film.", steps: ["Shadow a front-foot drive.", "Freeze at the moment you would hit the ball.", "Your partner checks that your nose is over your front toes.", "Hold for 2 seconds.", "Return to your stance."], sets: 1, reps: 10, unit: "reps", timeMinutes: 6, successCheck: "You know it's working when your nose is over your front toes on 8 of 10 freezes.", progression: "Do the same off a tennis ball on a tee, and freeze after contact.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F16-C", fault: "F16", level: "club", name: "Chest over the cone", purpose: "Puts your head and chest over the ball at contact, with a marker for where that is.", equipment: ["Bat", "Tennis balls", "Partner", "Cone"], setup: "Put a cone just in front of where your front foot lands. Drop feed: your partner kneels on your leg side, out of the path of the bat, and drops a tennis ball so it bounces in front of your front pad. The ball bounces next to the cone.", steps: ["Stride towards the cone.", "Get your head and chest over the cone as you hit the ball.", "Drive along the ground.", "Hold the finish for 2 seconds.", "Your partner says \"back\" if your head stayed behind the cone."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when your partner does not need to say \"back\" on most balls.", progression: "Use underarm throws from 8 metres that bounce near the cone.", safety: "Use a tennis ball. The feeder kneels on your leg side, out of the path of the bat, and leans back after each drop.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F16-A", fault: "F16", level: "advanced", name: "Freeze after contact", purpose: "Checks where your weight is at contact against throws, when it is hardest to keep it moving forward.", equipment: ["Bat", "Tennis balls", "Partner", "Phone, propped up"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. Prop the phone side-on to you, out of the line of the ball.", steps: ["Drive the ball along the ground.", "Freeze straight after contact.", "Hold the freeze for 2 seconds.", "After 6 balls, watch the clip.", "Check that your head and chest are over your front foot."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the clip shows your head over your front foot at contact on most balls.", progression: "Your partner mixes the lengths, so you must still get forward to the full ones.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Put the phone where the ball and the bat cannot reach it.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F07-B", fault: "F07", level: "beginner", name: "Side-on to contact", purpose: "Keeps your front shoulder pointing down the line of the shot until contact, so it does not pull the bat across.", equipment: ["Bat", "Tennis balls", "Tee or tall cone", "Partner, or a phone propped up"], setup: "Put a tennis ball on a tee next to where your front foot lands. Face a net or open space. Your partner stands 4 metres behind you, like a wicketkeeper standing back. Or prop a phone there and film.", steps: ["Pick a target straight down the pitch.", "As you stride, point your front shoulder at the target.", "Drive the ball off the tee.", "Keep the shoulder pointing at the target until the ball has gone.", "Your partner calls \"open\" if your chest turned to the target before contact."], sets: 1, reps: 10, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when your partner calls \"open\" on no more than 2 balls in 10.", progression: "Use soft drop feeds instead of the tee.", safety: "Hit into a net or towards open space, away from people, windows and cars. Your partner stands well out of reach of the bat.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F07-C", fault: "F07", level: "club", name: "Chin to shoulder", purpose: "Uses your chin and front shoulder as a check: if they stay close, the shoulder has not opened early.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Drop feed: your partner kneels on your leg side, out of the path of the bat, and drops a tennis ball so it bounces in front of your front pad.", steps: ["As you stride, bring your chin close to your front shoulder.", "Keep it there as you swing.", "Drive the drop-fed ball.", "Let the shoulder turn only after the ball has gone.", "Hold the finish for 2 seconds."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when your drives go straight and you feel the shoulder turn after contact, not before.", progression: "Use underarm throws from 8 metres.", safety: "Use a tennis ball. The feeder kneels on your leg side, out of the path of the bat, and leans back after each drop.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F07-A", fault: "F07", level: "advanced", name: "Off-side gate", purpose: "Shows you what an early-opening shoulder does: the ball goes squarer than you meant.", equipment: ["Bat", "Tennis balls", "Partner", "Two cones"], setup: "Put two cones 3 metres apart at mid-off, about 15 metres away. Your partner throws overarm, with one bounce, from 10 to 12 metres, standing on the leg side. They throw on and just outside off stump.", steps: ["Point your front shoulder at the gate as you stride.", "Drive the ball through the gate.", "Keep the shoulder closed until contact.", "Score 1 point for each drive through the gate.", "Play 12 balls."], sets: 1, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you score 8 or more out of 12, and the misses go straighter, not squarer.", progression: "Move the gate to extra cover for balls wider of off stump, and keep the shoulder closed.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Nobody stands near the gate.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F08-B", fault: "F08", level: "beginner", name: "Through the gate", purpose: "Teaches the bat to come down straight along the line of the ball, so the ball goes where you aim.", equipment: ["Bat", "Tennis balls", "Tee or tall cone", "Two small cones"], setup: "Put two cones a little wider than a ball apart, one bat length in front of your front foot. Put a tennis ball on a tee between you and the gate.", steps: ["Check your stance: head over your toes, eyes in line with off stump.", "Take your stance.", "Drive the ball off the tee straight through the gate.", "Hold the finish for 2 seconds.", "Reset the ball and repeat."], sets: 1, reps: 10, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when the ball goes through the gate on 7 of 10 drives without touching a cone.", progression: "Make the gate narrower, or move it further away.", safety: "Hit into a net or towards open space, away from people, windows and cars.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F08-C", fault: "F08", level: "club", name: "Straight back past the stumps", purpose: "Joins the straight bat to a moving ball, aiming back past the stumps at the other end.", equipment: ["Bat", "Tennis balls", "Partner", "Stumps or two cones"], setup: "Put stumps or two cones 1 metre apart at the far end. Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot.", steps: ["Check your stance: head over your toes, eyes in line with off stump.", "Watch the ball from the hand.", "Bring the bat down straight along the line of the ball.", "Drive back between the cones.", "Score 1 point for each drive between them."], sets: 2, reps: 8, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you score 6 or more out of 8, and the misses are close to the cones, not wide to the leg side.", progression: "Your partner throws on off stump, and you still drive straight.", safety: "Use a tennis ball. The thrower stands 6 to 8 metres away, to one side of the line of the shot, and watches every ball. Nobody stands near the cones.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F08-A", fault: "F08", level: "advanced", name: "Narrow channel", purpose: "Tests a straight bat path against throws on different lines, aimed into a narrow channel.", equipment: ["Bat", "Tennis balls", "Partner", "Four cones"], setup: "Make a channel 3 metres wide with cones, from 10 metres in front of you to 20 metres, straight down the ground. Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot.", steps: ["Check your stance before each ball.", "Bring the bat down along the line of the ball.", "Drive into the channel.", "Score 2 points along the ground and 1 point in the air.", "Play 12 balls."], sets: 1, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you score 16 or more out of 24.", progression: "Make the channel 2 metres wide.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Nobody stands in or near the channel.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F09-B", fault: "F09", level: "beginner", name: "One-two rhythm", purpose: "Makes the backlift and the swing one smooth movement, with no stop at the top.", equipment: ["Bat", "Tennis balls and a partner (optional)"], setup: "Stand in an open space in your stance.", steps: ["Say \"one\" as your hands lift.", "Say \"two\" as you swing through.", "Leave no gap between \"one\" and \"two\".", "Do 10 shadow drives.", "If you have a partner, do 10 more with soft drop feeds."], sets: 2, reps: 10, unit: "reps", timeMinutes: 8, successCheck: "You know it's working when \"one-two\" sounds like one word and the swing feels smooth.", progression: "Do the same off a tennis ball on a tee, with the count out loud.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing. For drop feeds, use a tennis ball. The feeder kneels on your leg side, out of the path of the bat.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F09-C", fault: "F09", level: "club", name: "Metronome drives", purpose: "Uses a steady beat to remove the stop between the backlift and the swing.", equipment: ["Bat", "Tennis balls", "Tee or tall cone", "Phone with a free metronome app"], setup: "Set a metronome to 60 beats a minute. Put a tennis ball on a tee next to where your front foot lands. Face a net or open space.", steps: ["Lift your hands on one beat.", "Swing and hit the ball on the next beat.", "Do not wait at the top.", "Reset the ball and wait for the next beat."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when you hit on the beat every time, with no extra wait at the top.", progression: "Set the metronome to 70 beats a minute.", safety: "Hit into a net or towards open space, away from people, windows and cars.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F09-A", fault: "F09", level: "advanced", name: "Smooth at speed", purpose: "Keeps your swing smooth when the ball is quicker, which is when a hitch often shows up.", equipment: ["Bat", "Tennis balls", "Partner", "Phone, propped up"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot. They vary the pace. Prop the phone side-on, out of the line of the ball.", steps: ["Start the lift early, as your partner's arm comes up.", "Swing into the shot in one smooth movement.", "A short settle of the front foot is fine. Do not stop and start the swing.", "After 6 balls, watch one clip.", "Check for a stop at the top."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the clip shows no stop at the top, on slow and quick balls alike.", progression: "Your partner adds a slower ball every few throws, and you keep the same smooth swing without a wait.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Put the phone where the ball and the bat cannot reach it.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F10-B", fault: "F10", level: "beginner", name: "Eyes on the tee", purpose: "Keeps your head over the ball at contact, so you meet it close to your front pad and keep it on the ground.", equipment: ["Bat", "Tennis balls", "Tee or tall cone"], setup: "Put a tennis ball on a tee next to your front pad. Face a net or open space.", steps: ["Take your stance.", "Drive the ball off the tee.", "Keep looking at the tee for one second after you hit it.", "Then look up to see where the ball went.", "Reset and repeat."], sets: 1, reps: 10, unit: "balls", timeMinutes: 6, successCheck: "You know it's working when you still see the tee after contact and the ball stays low.", progression: "Move the tee half a stride further forward, and still keep your head over it.", safety: "Hit into a net or towards open space, away from people, windows and cars.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F10-C", fault: "F10", level: "club", name: "Ball under the eyes", purpose: "Joins a still head to a moving ball, met under your eyes next to the front pad.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Drop feed: your partner kneels on your leg side, out of the path of the bat, and drops a tennis ball so it bounces in front of your front pad. The ball bounces next to your front pad.", steps: ["Watch the ball from your partner's hand.", "Stride and meet the ball under your eyes, next to your front pad.", "Keep your head still at contact.", "Look at the contact spot for one second after the shot.", "Drive along the ground."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when you can point to the spot where you hit the ball, and the ball stays low.", progression: "Use underarm throws from 8 metres.", safety: "Use a tennis ball. The feeder kneels on your leg side, out of the path of the bat, and leans back after each drop.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F10-A", fault: "F10", level: "advanced", name: "Read the number", purpose: "Trains you to watch the ball right onto the bat, which keeps your head over it.", equipment: ["Bat", "Tennis balls with a large number (1 to 4) on each", "Partner"], setup: "Write a large number from 1 to 4 on each tennis ball with a marker pen. Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot.", steps: ["Watch the ball from the hand.", "Call the number out loud as the ball comes.", "Drive the ball.", "Keep your head over the ball at contact.", "Your partner tells you if the number was right."], sets: 2, reps: 6, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you call most numbers right and your head is still over the ball at contact.", progression: "Use smaller numbers, or your partner throws a little faster.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F11-B", fault: "F11", level: "beginner", name: "Hold the pose", purpose: "Teaches you to finish every shot balanced, which shows your body stayed in control through the shot.", equipment: ["Bat", "Tennis balls", "Tee, or a partner to drop feed"], setup: "Put a tennis ball on a tee next to where your front foot lands, or ask a partner to drop feed. Face a net or open space.", steps: ["Drive the ball.", "Hold your finish for a count of two.", "Keep your head over your front knee.", "If you wobble, play the next ball at half pace.", "Reset and repeat."], sets: 1, reps: 10, unit: "balls", timeMinutes: 6, successCheck: "You know it's working when you hold 8 of 10 finishes for a count of two without a step.", progression: "Hold each finish for a count of three.", safety: "Hit into a net or towards open space, away from people, windows and cars. For drop feeds, use a tennis ball. The feeder kneels on your leg side, out of the path of the bat.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F11-C", fault: "F11", level: "club", name: "One-leg finish", purpose: "Makes balance at the finish harder, so you hold it on your front leg alone.", equipment: ["Bat", "Tennis balls", "Tee or tall cone"], setup: "Put a tennis ball on a tee next to where your front foot lands. Face a net or open space. Use a flat surface that is not slippery.", steps: ["Drive the ball off the tee.", "Hold the finish.", "Lift your back foot a little off the ground.", "Balance on your front leg for 2 seconds.", "Put the back foot down and reset."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when you can balance on your front leg for 2 seconds after most drives.", progression: "Do the same with soft drop feeds from a partner.", safety: "Hit into a net or towards open space, away from people, windows and cars.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F11-A", fault: "F11", level: "advanced", name: "Finish score", purpose: "Keeps your finish balanced as the ball gets quicker, with a simple score.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot.", steps: ["Drive the ball.", "Hold your finish for a count of three.", "Your partner gives 1 point for a still finish.", "Your partner gives 0 points for a step or a wobble.", "Play 12 balls, then 12 more a little quicker."], sets: 2, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you score 10 or more out of 12 in each set.", progression: "Your partner varies the length, so you must stay balanced on every ball.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F12-B", fault: "F12", level: "beginner", name: "Through and hold", purpose: "Lets your hands keep going towards the target, with the bat under control at the end.", equipment: ["Bat"], setup: "Stand in an open space in your stance. Pick a target straight down the pitch.", steps: ["Shadow a drive.", "Let your hands keep going towards the target after the contact point.", "Finish with the bat under control. A short or a full finish are both fine.", "Hold the finish for 2 seconds.", "Return to your stance."], sets: 1, reps: 10, unit: "reps", timeMinutes: 6, successCheck: "You know it's working when every finish looks the same and the bat is still at the end.", progression: "Do the same off a tennis ball on a tee.", safety: "Clear a space. Make sure nobody is within two bat lengths of your swing.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F12-C", fault: "F12", level: "club", name: "Hands to the target", purpose: "Joins a full, controlled follow-through to a real hit, aimed at a target.", equipment: ["Bat", "Tennis balls", "Tee or tall cone", "Cone for a target"], setup: "Put a target cone 15 metres away, straight down the pitch. Put a tennis ball on a tee next to where your front foot lands.", steps: ["Drive the ball towards the target cone.", "Let your hands follow the ball towards the cone.", "Finish with the bat under control.", "Hold for 2 seconds.", "Reset and repeat."], sets: 2, reps: 8, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when your hands point at the target in every finish and the ball goes towards it.", progression: "Move the target to mid-off, then to mid-on, and follow the ball each time.", safety: "Hit into a net or towards open space, away from people, windows and cars. Nobody stands near the target.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F12-A", fault: "F12", level: "advanced", name: "Same finish, any effort", purpose: "Keeps your follow-through controlled whether you hit softly or hard.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot.", steps: ["Drive the first 4 balls at half effort.", "Drive the next 4 at three-quarter effort.", "Drive the last 4 at full effort.", "Make every finish look the same.", "Hold each finish for 2 seconds."], sets: 1, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when the full-effort finishes look the same as the half-effort ones.", progression: "Your partner calls the effort level just as the ball is released.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F00-B", fault: "F00", level: "beginner", name: "Keep it grooved", purpose: "Keeps a good drive the same, ball after ball, so it holds up when it matters.", equipment: ["Bat", "Tennis balls", "Tee, or a partner to drop feed"], setup: "Put a tennis ball on a tee next to where your front foot lands, or ask a partner to drop feed. Face a net or open space.", steps: ["Drive 20 balls at three-quarter pace.", "Hold every finish for a count of two.", "Use the same set-up for every ball.", "At the end of the week, film one clip and analyse it."], sets: 1, reps: 20, unit: "balls", timeMinutes: 8, successCheck: "You know it's working when every drive looks and feels the same.", progression: "Use underarm throws from 8 metres.", safety: "Hit into a net or towards open space, away from people, windows and cars. For drop feeds, use a tennis ball. The feeder kneels on your leg side, out of the path of the bat.", soloOk: true, videoUrl: null, signoff: "pending" },
  { id: "F00-C", fault: "F00", level: "club", name: "Groove and vary", purpose: "Keeps the same good drive when the line changes: off side, straight and on side.", equipment: ["Bat", "Tennis balls", "Partner"], setup: "Your partner throws tennis balls underarm from 6 to 8 metres, standing to one side of the line of the shot. They throw on off stump, middle stump and leg stump.", steps: ["Drive the off-stump balls towards mid-off.", "Drive the middle-stump balls straight.", "Drive the leg-stump balls towards mid-on.", "Hold every finish for a count of two.", "Keep your head in the same position each time."], sets: 2, reps: 9, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when each drive goes where you aim and all the finishes look the same.", progression: "Your partner throws overarm, with one bounce, from 10 to 12 metres.", safety: "Use a tennis ball. The thrower stands 6 to 8 metres away, to one side of the line of the shot, and watches every ball.", soloOk: false, videoUrl: null, signoff: "pending" },
  { id: "F00-A", fault: "F00", level: "advanced", name: "Pressure grooving", purpose: "Tests your drive under a little pressure, with a score, so it stays the same when it counts.", equipment: ["Bat", "Tennis balls", "Partner", "Four cones"], setup: "Make two gates with cones, one at mid-off and one at mid-on, about 15 metres away. Your partner throws tennis balls overarm, with one bounce, from 10 to 12 metres, standing to one side of the line of the shot.", steps: ["Your partner calls \"off\" or \"on\" as the ball is released.", "Drive through the gate they called.", "Score 2 points along the ground with a held finish.", "Score 1 point through the gate in the air.", "Play 12 balls and try to beat your last score."], sets: 1, reps: 12, unit: "throws", timeMinutes: 10, successCheck: "You know it's working when you score 16 or more out of 24 and hold every finish.", progression: "Your partner calls later, or throws a little faster.", safety: "Use a tennis ball. The thrower stands at least 10 metres away, to one side of the line of the shot, or behind a net screen. Nobody stands near the gates.", soloOk: false, videoUrl: null, signoff: "pending" },
];
```

**2. `src/data/drillLibrary.ts`:**

```ts
import type { Drill, DrillFaultId, DrillLevel } from "./drillTypes";
import { DRILLS_PART_1 } from "./drillLibraryPart1";
import { DRILLS_PART_2 } from "./drillLibraryPart2";

export * from "./drillTypes";

export const DRILLS: Drill[] = [...DRILLS_PART_1, ...DRILLS_PART_2];

export const DRILL_FAULT_ORDER: DrillFaultId[] = [
  "F01", "F02", "F17", "F13", "F03", "F19", "F15", "F04", "F05", "F06",
  "F18", "F16", "F07", "F08", "F09", "F10", "F11", "F12", "F00",
];

export function getDrill(fault: DrillFaultId, level: DrillLevel): Drill | undefined {
  return DRILLS.find((d) => d.fault === fault && d.level === level);
}

export function getDrillById(id: string): Drill | undefined {
  return DRILLS.find((d) => d.id === id);
}

// Development check only: never throws, never shows anything to users.
export function checkDrillLibrary(): string[] {
  const problems: string[] = [];
  if (DRILLS.length !== 57) problems.push(`expected 57 drills, found ${DRILLS.length}`);
  const ids = new Set<string>();
  for (const d of DRILLS) {
    if (ids.has(d.id)) problems.push(`duplicate id ${d.id}`);
    ids.add(d.id);
    if (d.steps.length < 3 || d.steps.length > 6) problems.push(`${d.id}: ${d.steps.length} steps`);
  }
  for (const f of DRILL_FAULT_ORDER) {
    for (const l of ["beginner", "club", "advanced"] as DrillLevel[]) {
      if (!getDrill(f, l)) problems.push(`missing ${f} ${l}`);
    }
  }
  return problems;
}

if (import.meta.env.DEV) {
  const p = checkDrillLibrary();
  if (p.length) console.warn("Drill library problems:", p);
}
```

**Check:** the project builds; in development the console shows no "Drill library problems" warning; `getDrill("F09", "club")?.name` is "Metronome drives"; `getDrill("F00", "advanced")?.name` is "Pressure grooving". Nothing on screen changes.

---

## Pass 3: the plan logic, saving and the "Your drill plan" section

Passes 1 and 2 are in place (`src/data/drillLibrary.ts`). So are the earlier results-page passes: `src/lib/coaching.ts` (`FAULTS`, `RULE_TO_FAULT`), `getPriorityFix` in `src/lib/priorityFix.ts`, the priority fix card, the "Film it right" sheet, the "Your progress" strip and the share card. If any of these is missing, stop and tell me; do not build a copy.

This pass changes the results page (`src/pages/AnalysisResults.tsx`) and adds new files. Do not change the upload flow, the login, the API call or any other page. Do not add any call to the backend (the Railway API). Use UK spelling. Keep the current fonts, colours and layout everywhere else.

### 1. Release switch

In `src/config/features.ts` (create it if it does not exist, and keep anything already in it), add:

```ts
drillPlan: "preview" as "off" | "preview" | "live",
```

- `"off"`: the plan section never shows.
- `"preview"`: the plan section shows only (a) to admins, meaning the signed-in user has a row in `user_roles` with `role = 'admin'` (users can read their own roles; if the table does not exist, nobody is an admin), or (b) when the page address has `?drills=preview`. Everyone else sees the results page exactly as before. Drills with `signoff` other than `"ok"` show the "Draft" badge.
- `"live"`: the section shows to everyone, and only drills with `signoff = "ok"` are used.

### 2. Age band

Create `src/lib/ageBand.ts` (if it already exists from the results prompt, reuse it and check it matches this):

```ts
export type AgeBand = "adult" | "under_18" | "unknown";
```

`getAgeBand(profile, uploadRow?)`:
1. `getProfileAge(profile)` from `src/lib/trainingConsent.ts`: a number under 18 gives `"under_18"`; 18 or more gives `"adult"`. If that file does not exist, use the same rule inline: look for `date_of_birth`, `dob` or `birth_date` (a date) or `age` (a number) on the profile, and work out whole years on today's date in Europe/London.
2. Else, if the analysis row links to its upload row (an `upload_id` column or similar) and that row has `uploader_under_18`: `true` gives `"under_18"`, `false` gives `"adult"`. One extra read of that one row, with the existing client, is fine.
3. Else `"unknown"`.

Treat `"unknown"` the same as `"under_18"` everywhere in this pass.

### 3. The plan logic: `src/lib/drillPlan.ts`

Keep every function here pure (no network, no storage, no React), so it is easy to test.

**Types:**

```ts
import type { Drill, DrillFaultId, DrillLevel } from "@/data/drillLibrary";
import type { AgeBand } from "@/lib/ageBand";

export type LevelSource = "self" | "profile" | "score_band" | "score" | "default";
export type SessionKey = "w1s1" | "w1s2" | "w1s3" | "w2s1" | "w2s2" | "w2s3" | "w3s1" | "w3s2" | "w3s3";
export const SESSION_KEYS: SessionKey[] = ["w1s1","w1s2","w1s3","w2s1","w2s2","w2s3","w3s1","w3s2","w3s3"];

export interface PlanSlot { drill: Drill; role: "main" | "second"; makeItHarder: boolean; oneSetOnly: boolean; }
export interface PlanSession { key: SessionKey; week: 1 | 2 | 3; index: 1 | 2 | 3; checkIn: boolean; minutes: number; slots: PlanSlot[]; }
export interface PlanWeek { week: 1 | 2 | 3; main: PlanSlot; second: PlanSlot | null; sessions: PlanSession[]; }
export type DrillPlan =
  | { kind: "plan"; faultId: DrillFaultId; faultName: string; secondId: DrillFaultId | null; secondName: string | null;
      isClean: boolean; level: DrillLevel; levelSource: LevelSource; ageBand: AgeBand; weeks: PlanWeek[] }
  | { kind: "noPlan"; reason: "none" | "unmapped" | "checking" };
```

**`normaliseFault(id)`:** returns `"F07"` for `"F14"`; returns the id for F00 to F19; else `null`.

**`resolveLevel(saved, profile, result)` returns `{ level, levelSource }`.** Stop at the first that gives an answer:
1. `saved?.levelSource === "self"` and `saved.level` is a valid level: that level, `"self"`.
2. A profile field `batting_level`, `playing_level` or `level` whose value, in lower case, is `beginner`, `club` or `advanced`: that level, `"profile"`.
3. `result.score_band`: "Fundamentals" or "Work Needed" gives `beginner`; "Developing", "Good" or "Excellent" gives `club` (compare ignoring case); `"score_band"`.
4. `result.battingiq_score` is a number: under 55 gives `beginner`, else `club`; `"score"`.
5. `beginner`, `"default"`.

The score never gives `advanced`. Only the player (step 1) or a profile field (step 2) can.

**`getSecondFault(result, priorityId)`:** go through `result.tips` in the order sent. Return the first `normaliseFault(item.id)` that is not null, not `"F00"` and not `priorityId`. If `tips` is missing, or nothing qualifies, return `null`. Never use the pillars for this.

**`pickDrill(fault, level, mode)`:**
- Usable drills: in `"live"`, drills of that fault with `signoff === "ok"`; in `"preview"`, drills of that fault with `signoff !== "removed"`.
- Try levels in this order and return the first usable drill, else `null`: for `beginner`: beginner, club, advanced; for `club`: club, beginner, advanced; for `advanced`: advanced, club, beginner.

**`nextLevel(level)`:** beginner gives club; club gives advanced; advanced gives advanced.
**`levelBelow(level)`:** advanced gives club; club gives beginner; beginner gives `null`.

**`buildDrillPlan({ fix, result, level, levelSource, ageBand, mode })`**, where `fix` is the output of `getPriorityFix(result)`:
1. If `fix` is `null`: return `null` (the caller hides the section).
2. If `fix.state === "none"`: `{ kind: "noPlan", reason: "none" }`. If `"unmapped"`: `{ kind: "noPlan", reason: "unmapped" }`.
3. `P = normaliseFault(fix.id)` (F00 when `fix.state === "clean"`). `L1 = level`, `L2 = nextLevel(L1)`.
4. `S = P === "F00" ? null : getSecondFault(result, P)`. If S is not null and `pickDrill(S, "beginner", mode)`, `pickDrill(S, "club", mode)` and `pickDrill(S, "advanced", mode)` are all null, set S to null.
5. If `pickDrill(P, L1, mode)` is null: `{ kind: "noPlan", reason: "checking" }`.
6. Choose each week's main and second drill:
   - **Fault plan (P is not F00):**
     - Week 1: main `pickDrill(P, L1)`; second none.
     - Week 2: main `pickDrill(P, L2)`; second `S ? pickDrill(S, L1) : pickDrill(P, L1)`.
     - Week 3: main `pickDrill(P, L2)`; second `S ? pickDrill(S, L2) : pickDrill("F00", L1)`.
   - **F00 plan:**
     - Week 1: main `pickDrill("F00", L1)`; second none.
     - Week 2: main `pickDrill("F00", L2)`; second: the F00 drill at `levelBelow(main.level)`, if it exists and is usable in this mode; else none.
     - Week 3: main `pickDrill("F00", L2)`; second none.
7. Clean-up, in this order, for each week:
   - If the main drill is null, use the previous week's main drill.
   - If the second drill has the same id as the main drill: for a fault plan, use `pickDrill("F00", L1)`; if that is also the same id, or this is an F00 plan, use none.
   - `makeItHarder` is true for a main drill whose id was the main drill of an earlier week. It is always false for second drills.
8. Sessions. Each week has sessions 1, 2 and 3, keys `w{week}s{index}`. Every session has the week's main slot, then (if any) its second slot. Except **week 3, session 3**, the check-in: `checkIn = true`, and it has only the main drill with `oneSetOnly = true`.
   - `minutes`: check-in 18; a session with a second drill 20; otherwise 15.
9. `faultName` and `secondName` come from `FAULTS` in `src/lib/coaching.ts` (for F00, "Nothing major to fix"). `isClean = P === "F00"`.

**Text helpers (also in this file):**
- `repsText(d)`: `d.sets === 1 ? "{reps} {unit}" : "{sets} sets of {reps} {unit}"` (for example "2 sets of 10 balls").
- `repsOneSet(d)`: `"{reps} {unit}"`.

### 4. The copy: `src/lib/drillPlanCopy.ts`

Create `PLAN_COPY` with exactly these keys and texts. Braces are filled in by the app.

```ts
export const PLAN_COPY = {
  sectionTitle: "Your drill plan",
  sectionIntroFault: "Three weeks of short sessions to work on your priority fix: {fault}.",
  sectionIntroClean: "Nothing major to fix, so this plan helps you keep your drive the same, ball after ball.",
  openPlan: "Open my plan",
  closePlan: "Hide my plan",
  summaryLine: "3 sessions a week for 3 weeks, 15 to 20 minutes each. Week 1: {drill}.",
  levelLine: "Level: {level}",
  levelGuessed: "We've guessed your level from your score. Pick the one that fits you best.",
  levelChange: "Change",
  levelDialogTitle: "Which sounds most like you?",
  levelSave: "Use this level",
  progressLine: "{done} of 9 sessions done",
  week1Title: "Week 1: learn the fix",
  week1Aim: "Learn one change for {fault}. Go slowly and get it right.",
  week1AimClean: "Groove your drive with one simple drill.",
  week2Title: "Week 2: build on it",
  week2AimSecond: "Make the fix harder, and start work on {second}.",
  week2AimNoSecond: "Make the fix harder, and keep last week's drill going as a short recap.",
  week2AimClean: "Make the drill harder, and keep last week's drill going.",
  week3Title: "Week 3: put it together",
  week3Aim: "Join the drills up, then film a new clip to see how it's going.",
  sessionLabel: "Session {n} · {minutes} minutes",
  sessionNote1: "Go slowly. Get every step right before you add speed.",
  sessionNote2: "Normal pace. Do the check at the end of the drill card.",
  sessionNote3: "Normal pace. Do the check again: is it easier than in session 1?",
  checkInLabel: "Check-in · 18 minutes",
  checkInNote: "One set of your main drill, then film a new clip.",
  warmUp: "Warm-up (3 minutes): jog or skip on the spot for 1 minute. Swing your arms in big circles, 10 each way. Do 5 slow shadow drives, then 5 at normal pace.",
  mainInstruction: "Main drill (10 minutes): do {reps}. If you finish early, do the last set again.",
  checkInMainInstruction: "Main drill (5 minutes): do one set of {repsOneSet}.",
  secondInstruction: "Second drill (5 minutes): do one set of {repsOneSet}.",
  finish: "Finish (2 minutes): play 5 slow shadow drives with today's cue. Hold each finish for a count of two.",
  filmBlock: "Film and re-analyse (10 minutes): film 3 drives, following the five \"Film it right\" rules. Upload the best one. Your new result shows your progress against this one.",
  makeItHarder: "Make it harder: {progression}",
  soloBadge: "Can do alone",
  soloBadgeJunior: "No partner needed (an adult still supervises)",
  partnerBadge: "Needs a partner",
  draftBadge: "Draft: waiting for a coach to check",
  checkedBadge: "Checked by a qualified coach",
  youNeed: "You need: {equipment}",
  setupLabel: "Set-up",
  stepsLabel: "Steps",
  repsLabel: "Reps",
  aboutMinutes: "About {minutes} minutes",
  checkLabel: "Check",
  nextStepLabel: "Next step",
  videoComingSoon: "Demo video coming soon",
  netsAdult: "If you have a net session this week, take your cue into it. With a hard ball, always wear a helmet with a faceguard and proper protective kit.",
  netsJunior: "If you have a net session this week, take your cue into it, with an adult supervising. Always wear a helmet with a faceguard against a hard ball.",
  reanalyseTitle: "Ready to check your progress?",
  reanalyseBody: "Film a new clip of the same shot and analyse it. We'll show your new result next to this one.",
  reanalyseButton: "Analyse a new clip",
  filmItRightLink: "How to film it right",
  newerAnalysis: "You've analysed a newer clip since this plan. Your latest plan is on that result.",
  newerAnalysisLink: "See my latest result",
  seePlanLink: "See your 3-week drill plan",
  noPlanNone: "We need a clearer clip to build your drill plan.",
  noPlanUnmapped: "We can't match this fix to our drills yet, so there is no plan for this clip. Try another clip.",
  noPlanChecking: "Drills for this fix are being checked by our coach. Check back soon.",
  printButton: "Print or save as PDF",
  shareButton: "Share plan",
  copyButton: "Copy plan",
  copiedToast: "Plan copied",
  copyFailedToast: "Couldn't copy the plan. Use Print or save as PDF instead.",
  printTitle: "My 3-week drill plan",
  printWorkingOn: "Working on: {fault}",
  printWorkingOnClean: "Working on: keeping my drive the same",
  footer: "Drills are suggestions based on a computer analysis of one clip. They can be wrong, and they don't know about your body or any injury. If something doesn't feel right, stop and check with a coach.",
} as const;
```

### 5. Saving the checklist and the level

**Plan state** (one per analysis):

```ts
export interface DrillPlanState {
  version: 1;
  analysisId: string;
  level?: DrillLevel;          // only when the player picked it
  levelSource?: "self";
  done: SessionKey[];
  createdAt: string;           // ISO time of the first save; never changed
  updatedAt: string;           // ISO time of the latest save
}
```

`analysisId` is the id of the analysis row the results page shows. If the page has no row id (for example, straight after an upload), use `result.job_id`.

**Server copy (reuse the existing analyses table; no new table).** Find the table that stores the user's analyses (the one the history list and the "Your progress" strip read, for example `analyses`). If it exists and has a column that holds the owner (for example `user_id`), create one migration. Change the table name, the id type and the owner column to match the real ones:

```sql
alter table public.analyses add column if not exists drill_plan_state jsonb;

create or replace function public.save_drill_plan_state(p_analysis_id uuid, p_state jsonb)
returns boolean language plpgsql security definer set search_path = public as $$
declare n integer;
begin
  if auth.uid() is null then return false; end if;
  if p_state is null or pg_column_size(p_state) > 4096 then return false; end if;
  update public.analyses set drill_plan_state = p_state
   where id = p_analysis_id and user_id = auth.uid();
  get diagnostics n = row_count;
  return n = 1;
end $$;

revoke all on function public.save_drill_plan_state(uuid, jsonb) from public, anon;
grant execute on function public.save_drill_plan_state(uuid, jsonb) to authenticated;
```

Do not add or change any row level security policy on that table; the function is the only way to write the column. Add `drill_plan_state` to the select the results page already uses to load the analysis row. If there is no such table, or no owner column, or the page does not load a row, skip the migration and use only the browser copy.

**Browser copy.** `localStorage` key `biq_drill_plan_v1_{analysisId}`. Wrap every read and write in try/catch. If storage fails, the plan still works for this visit.

**`src/lib/drillPlanStorage.ts`:**
- `loadPlanState(analysisId, serverState)`: read the browser copy; compare with the server copy; return the one with the later `updatedAt` (or whichever exists). If they differ, write the winner to the other place (server via `save_drill_plan_state`, browser via `localStorage`).
- `savePlanState(state)`: set `updatedAt` to now (and `createdAt` if it is the first save); write the browser copy first, then call `supabase.rpc("save_drill_plan_state", { p_analysis_id, p_state })` if the server copy is in use. If the call fails or returns false, do nothing visible; the next save tries again.
- Save straight away on every tick, untick and level change. No save button.

### 6. Where it goes on the results page

Put the plan section **directly under the priority fix card**, above the storyboard. Show it only when: the release switch allows it (section 1), and `buildDrillPlan` returns something (not `null`).

**Collapsed by default.** The section starts as a summary card (the app's normal card style):
- Title `sectionTitle`, with a small "Dumbbell" lucide icon.
- `sectionIntroFault` (with the fault name) or `sectionIntroClean`.
- `summaryLine` with the week 1 main drill's name.
- `progressLine` and a thin progress bar, only if at least one session is ticked.
- A full-width button `openPlan`. When open, the button reads `closePlan`.
- It starts **open** if at least one session is ticked, or if the address has `#drill-plan`.

Give the section the id `drill-plan` so the priority card link can scroll to it.

**"No plan" states** (`kind: "noPlan"`): a small muted card with `sectionTitle`, the matching text (`noPlanNone`, `noPlanUnmapped` or `noPlanChecking`) and, for `none` and `unmapped`, the link `filmItRightLink`, which opens the existing "Film it right" sheet. No button, no weeks.

### 7. The open plan

Top to bottom:

1. **Level line.** `levelLine` with the level's label (Beginner, Club or Advanced from `DRILL_LEVEL_LABELS`), and a text button `levelChange`. If `levelSource` is `score_band`, `score` or `default`, show `levelGuessed` under it in small muted text. `levelChange` opens a sheet (bottom sheet on phones, dialog on wider screens) titled `levelDialogTitle`, with three radio options (label in bold, description under it), the current level selected, and a button `levelSave`. Saving stores `level` and `levelSource: "self"`, rebuilds the plan at once and keeps the ticks.
2. **Junior note.** For `under_18` or `unknown`: a highlighted note (soft warning tint, a "ShieldCheck" icon) with `DRILL_SAFETY_JUNIOR`.
3. **Progress.** `progressLine` and the progress bar.
4. **Re-analyse card or newer-analysis banner** (section 9), when it applies.
5. **Week tabs.** A segmented control: "Week 1", "Week 2", "Week 3". The current week is selected on load: the first week with an unticked session; week 3 if all 9 are ticked. A small tick icon shows on a tab when all 3 of its sessions are ticked.
6. **The week panel:**
   - The week title (`week1Title`, `week2Title`, `week3Title`) and its aim: week 1 `week1Aim` (or `week1AimClean`); week 2 `week2AimSecond` if there is a second fault, `week2AimClean` for F00, else `week2AimNoSecond`; week 3 `week3Aim`.
   - **Three session rows.** Each row: a checkbox (the whole row is the tap target, at least 44 px high), then `sessionLabel` (or `checkInLabel` for the check-in), then its note in small muted text (`sessionNote1` to `sessionNote3`, or `checkInNote`), then a compact line of its blocks: "Warm-up 3 min · {main drill name} 10 min · {second drill name} 5 min · Finish 2 min" (leave out the second drill when there is none; for the check-in: "Warm-up 3 min · {main drill name} 5 min · Film and re-analyse 10 min"). Each drill name in that line is a link that scrolls to its card below. Ticking saves at once (section 5).
   - **"How each session works"**: a collapsed row that opens to show, in order: `warmUp`; `mainInstruction` with `repsText(main)`; `secondInstruction` with `repsOneSet(second)` (if there is one); `finish`.
   - **Drill cards** for the week's main drill, then its second drill. The main drill card starts open; the second starts closed.
   - **Week 3 only:** a "Check-in" box after the drill cards: `checkInMainInstruction` with `repsOneSet(main)`, `filmBlock`, the link `filmItRightLink` (opens the existing sheet) and a button `reanalyseButton` (goes to the app's upload page: the same route the main "upload" or "analyse" button uses).
   - **Weeks 2 and 3 only:** the nets note at the end: `netsAdult` for adults, `netsJunior` for `under_18` and `unknown`.
7. **Footer.** `footer` in small muted text, then three buttons in a row (stacked on phones): `printButton`, `shareButton` (only if `navigator.share` exists), `copyButton`.

### 8. The drill card

A rounded card with a header that opens and closes it (`aria-expanded`). Header: the drill name (bold), and badges: the level label; `soloBadge` (adult) or `soloBadgeJunior` (`under_18`, `unknown`) when `soloOk`, else `partnerBadge`; `draftBadge` (muted outline) when `signoff !== "ok"`, or `checkedBadge` (success colour, small tick) when `signoff === "ok"`. Under the header, always visible: the `purpose`.

When open, in this order:
1. If `makeItHarder`: a highlighted line, `makeItHarder` with the drill's `progression`.
2. **Video area.** A 16:9 rounded box, full card width.
   - `videoUrl` is null: a muted panel with a "Video" lucide icon and `videoComingSoon`.
   - `videoUrl` ends in `.mp4`, `.webm` or `.mov` (ignore any `?` query): a `<video controls playsInline preload="metadata">` with that source. No autoplay, no loop.
   - `videoUrl` is a YouTube link (`youtube.com/watch?v=`, `youtu.be/` or `youtube.com/shorts/`): an `<iframe>` to `https://www.youtube-nocookie.com/embed/{id}`, with `title` set to the drill name, `loading="lazy"` and `allowfullscreen`.
   - Any other URL: a link button "Watch the demo" that opens it in a new tab (`rel="noopener noreferrer"`).
3. `youNeed` with the equipment joined by ", ".
4. `setupLabel` (small bold), then `setup`.
5. `stepsLabel`, then the steps as a numbered list.
6. `repsLabel`: `repsText(drill)`, then `aboutMinutes` with `timeMinutes`.
7. `checkLabel`: `successCheck`.
8. `nextStepLabel`: `progression` (leave this out when `makeItHarder` is shown at the top).
9. Safety, small muted text, three lines in this order: the drill's own `safety`; `DRILL_SAFETY`; `DRILL_SAFETY_JUNIOR` only for `under_18` and `unknown`.

### 9. Re-analyse and newer analyses

- Use the list of the user's analyses that the "Your progress" strip already loads (no new query). If any analysis was created after this one, show `newerAnalysis` as a banner at the top of the open plan, with a link `newerAnalysisLink` to the newest one's results page. Do not show the re-analyse card in this case.
- Else, if all 9 sessions are ticked, or the plan state's `createdAt` is 21 or more days ago: show the re-analyse card: `reanalyseTitle` (bold), `reanalyseBody`, the button `reanalyseButton` and the link `filmItRightLink`.
- Never say a fault is "fixed" or "gone".

### 10. The priority fix card

When the plan section shows with `kind: "plan"`, change the drill box on the priority fix card (only this box) to:
- **"Drill to try: {week 1 main drill name}"**, then that drill's `purpose`, then the link `seePlanLink`. The link opens the plan section and scrolls smoothly to `#drill-plan`.
- Keep the safety lines in the box: `DRILL_SAFETY`, plus `DRILL_SAFETY_JUNIOR` for `under_18` and `unknown`.

When the plan section is hidden, or shows a "no plan" state, the card keeps its own drill box, exactly as before.

### 11. Print and share

- `printButton` opens a full-screen dialog with the print view (id `drill-plan-print`) and a "Print" and a "Close" button (class `no-print`). "Print" calls `window.print()`. The print view: `printTitle`, then `printWorkingOn` (or `printWorkingOnClean` for F00), then the level line, the junior note (if it applies), and all three weeks open. Each session row has an empty square box to tick on paper. Every drill card is in full (all of section 8 except the video area and the badges). Then the nets note, the check-in box text and the `footer`. No name, no score, no photo, no analysis link.
- Print styles (in the global stylesheet):

```css
@media print {
  body * { visibility: hidden; }
  #drill-plan-print, #drill-plan-print * { visibility: visible; }
  #drill-plan-print { position: absolute; left: 0; top: 0; width: 100%; }
  #drill-plan-print .print-week + .print-week { break-before: page; }
  #drill-plan-print .print-drill { break-inside: avoid; }
  .no-print { display: none !important; }
}
```

- `shareButton` calls `navigator.share({ title: PLAN_COPY.printTitle, text })`. `copyButton` calls `navigator.clipboard.writeText(text)` and shows the toast `copiedToast` (on failure, `copyFailedToast`). If the user cancels sharing (`AbortError`), do nothing; on any other error, copy instead.
- The text (`text`), plain, one item a line:

```
My 3-week Batting IQ drill plan
Working on: {fault name, or "keeping my drive the same" for F00}
Week 1: {week 1 drill names, joined with " + "}
Week 2: {week 2 drill names, joined with " + "}
Week 3: {week 3 drill names, joined with " + "}, then film a new clip
3 sessions a week, 15 to 20 minutes each. Warm up first.
{DRILL_SAFETY}
{DRILL_SAFETY_JUNIOR, only for under_18 or unknown}
{PLAN_COPY.footer}
{window.location.origin}
```

Never put a name, score, photo, analysis link or referral code in the print view or the text, for any age.

### 12. Mobile

Design for a 375 px wide phone first.
- The summary card, the week tabs and the buttons are full width. The tabs fit on one line ("Week 1", "Week 2", "Week 3").
- Session rows and checkboxes are at least 44 px high. Drill names wrap; nothing is cut off.
- The video box keeps its 16:9 shape at full card width.
- The level sheet and the print dialog scroll if their content is taller than the screen.
- Footer buttons stack on phones and sit in one row from 640 px up.
- On wider screens (1024 px and up) the plan stays in the same single column as the rest of the results page, at most 720 px wide.

### 13. Tests

Use the Pass 1 example result from the results prompt (`priority_fix.rule_id = "F4"`, `score_band = "Good"`, no `tips`, handedness disagreement) as an admin with `FEATURES.drillPlan = "preview"`, profile age unknown. This must show:
- Under the priority card, a collapsed **Your drill plan** card: "Three weeks of short sessions to work on your priority fix: Pause or hitch in the swing." and "Week 1: Metronome drives." The priority card's drill box reads "Drill to try: Metronome drives", its purpose, "See your 3-week drill plan" and the safety lines, including the under-18 line (age unknown).
- "Open my plan": "Level: Club" with the guessed-level line; the junior note; "0 of 9 sessions done"; Week 1 selected.
- Week 1: Metronome drives (F09-C) in all three sessions, 15 minutes each; the card shows "Demo video coming soon", "2 sets of 8 balls", "About 8 minutes", the "Draft" badge, and "No partner needed (an adult still supervises)" (because the age is unknown).
- Week 2: Smooth at speed (F09-A) then Metronome drives as a recap, 20 minutes; the aim "Make the fix harder, and keep last week's drill going as a short recap."; the junior nets note.
- Week 3: Smooth at speed with "Make it harder: Your partner adds a slower ball every few throws, and you keep the same smooth swing without a wait.", then Groove and vary (F00-C); session 3 is the 18-minute check-in with "Analyse a new clip".

And these variants:
1. Change the level to Beginner: week 1 One-two rhythm; week 2 Metronome drives + One-two rhythm; week 3 Metronome drives with "Make it harder: Set the metronome to 70 beats a minute." + Keep it grooved. The ticks already made stay. The guessed-level line goes away.
2. Add `"tips": [{"id": "F13"}, {"id": "F09"}, {"id": "F11"}]`: the second fault is F13. Week 2: Smooth at speed + Lift with the gather (F13-C), aim "Make the fix harder, and start work on Hands late: no rhythm from the backlift."; week 3: Smooth at speed + Early lift, quicker ball (F13-A).
3. All four pillars measured with no faults, `priority_fix: null`: "Nothing major to fix, so this plan helps you keep your drive the same, ball after ball." Week 1 Groove and vary; week 2 Pressure grooving + Groove and vary; week 3 Pressure grooving with "Make it harder", no second drill, 15-minute sessions and the check-in.
4. All four pillars `assessed: false`: the small card "We need a clearer clip to build your drill plan." with "How to film it right". The priority card is as before.
5. A profile aged 15: the junior note, the junior line on every card and the junior nets note. A profile aged 30: none of these; `netsAdult`; "Can do alone".
6. Tick week 1 sessions 1 and 2, then reload: both still ticked, "2 of 9 sessions done", the plan starts open on Week 1. In the analyses table, `drill_plan_state.done` is `["w1s1","w1s2"]` (if the server copy is in use). Tick all three week 1 sessions: Week 2 is selected on the next load.
7. With `localStorage` blocked (private window): the plan works and ticks save to the server copy; with no server copy, the ticks last until the page is closed, with no error.
8. Not an admin, no `?drills=preview`: no plan section, and the priority card's drill box is "Drill to try: One-two rhythm", as before. Add `?drills=preview`: the section shows.
9. `FEATURES.drillPlan = "live"` while every drill is `pending`: "Drills for this fix are being checked by our coach. Check back soon.", and the priority card keeps its own drill box. Set F09-C to `signoff: "ok"`: week 1 shows Metronome drives with "Checked by a qualified coach". Weeks 2 and 3 fall back to F09-C as well (F09-A is not signed off), and week 2 then has no second drill.
10. Set `videoUrl` on F09-C to an `.mp4` address: a video player replaces "Demo video coming soon". Set it to a YouTube link: a youtube-nocookie player.
11. "Print or save as PDF": the print preview shows only the plan, with empty tick boxes, no name and no score. On a phone, "Share plan" opens the share sheet with the plain text; on a desktop without sharing, "Copy plan" copies it and shows "Plan copied".
12. When a newer analysis exists: the banner "You've analysed a newer clip since this plan. Your latest plan is on that result." with "See my latest result", and no re-analyse card.

**Out of scope for this pass:** new backend calls, a new table, push or email reminders, editing drills in the app, uploading videos in the app, coach access to plans, and comparing plans between analyses. Keep the current fonts, colours and layout everywhere else.

---

## Follow-up template: Alex's sign-off (paste after Alex fills in the sign-off sheet)

> In `src/data/drillLibraryPart1.ts` and `src/data/drillLibraryPart2.ts`, change only the `signoff` field of these drills, and nothing else: {for example: F01-B, F01-C, F01-A: "ok"; F07-A: "removed"}. Do not change any text.

If Alex asked for a wording change, edit `docs/phase6/drill_library.json` first, then give Lovable the drill's new text the same way ("replace the `steps` of F05-C with exactly: ...") and set `signoff: "ok"` only once the change is in.

## Follow-up template: a demo video (paste when a video is uploaded)

> In `src/data/drillLibraryPart1.ts` or `src/data/drillLibraryPart2.ts`, set `videoUrl` of drill {F09-C} to "{https://... .mp4 or YouTube link}". Change nothing else.

Also set `video_slot.status` to `"live"` and `video_slot.url` in `docs/phase6/drill_library.json`. Use only videos of an adult batter, with nobody else recognisable and no club badges.

---

## Note for Alex (not for Lovable)

- **Order:** Pass 0a today. Passes 1 to 3 when you are ready to test. The plan stays hidden from users (`preview`) until you switch it to `live`.
- **Before `live`:** Alex signs off the drills (`docs/phase6/drill_signoff.md`), then set `signoff` with the follow-up template, run the Pass 3 tests, and then paste Pass 0b. This keeps to the rule "nothing goes live untested".
- **The priority card's "Drill to try"** (results prompt Pass 1) is each fault's beginner drill from the library, so Alex's sign-off covers it too.
- **Claims:** keep "coaching videos" off the pricing card until every drill has a demo. Never say the drills prevent injury or are "as good as a coach".
