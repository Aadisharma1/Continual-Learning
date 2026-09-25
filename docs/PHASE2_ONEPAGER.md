# Phase 2 — Subliminal Safety Erosion (one-pager)

**Where we are.** Phase 1 is solid: aligned checkpoints, real EWC baseline, full GSM8K, XSTest split properly, HarmBench, and every trace screened by Llama-Guard-3. The result: ten rounds of self-improvement on benign math traces erodes refusal, and the safety filter never fires once. What we can't say is *why* — a reviewer will just say "that's fine-tuning forgetting, Qi et al. 2024."

**What puzzles me in our own data.** Three things. (1) The drift keeps climbing after the model stops learning math — by round 4 math is plateaued, so the "each gradient step nudges it toward unsafe" story should fade with it. It doesn't. (2) The damage is lopsided: AdvBench got 10 points *better* while XSTest got 17 worse. That's not general decay, that's one behavior losing margin — complying when it should refuse. (3) STaR doesn't train on the model's output, it trains on the slice a filter let through. Refusal is never in that slice, so it never gets reinforced, while the filter keeps pointing training at "always answer." A filter applied a hundred times starts looking like a preference.

**What I think is happening.** Training on your own filtered outputs round after round is a ratchet pushing the model toward "always comply." The teacher's style is "helpful assistant that answers," which is harmless by every text measure — but refusal is the opposite of that style, and it gets re-injected every round while refusal gets nothing. This doesn't need an evil teacher like Cloud et al.'s setup. It needs a teacher, a filter, and enough rounds.

**Three testable predictions.**
1. *The damage has a sign.* Project each round's update onto the refusal direction: my story says negative every round; forgetting says it flips randomly like a random walk. A sign test across ten rounds settles it — and we can run it on logs we already have.
2. *The damage sits in a few traces.* Ordinary forgetting smears damage evenly, so the worst 10% of traces carry about 10% of it. I predict much more. If I'm right, we can point at the harmful traces; if I'm wrong, no data filter can help and we say so.
3. *The safety buffer you need grows with the horizon.* Bianchi's 3% works for one shot because it's one tug-of-war. Our loop applies the same pressure every round against fixed safety data, so the buffer needed should grow with rounds. Measuring that curve at 1, 3, 10, and 20 rounds is the cleanest separation of our story from Bianchi's, and nobody's measured it.

**Experiments.** A five-condition factorial (self vs. human data × 1 vs. 10 rounds ± buffer), a teacher swap (own vs. cross-family vs. human-gold traces at matched accuracy, plus one deliberately-misaligned teacher arm), three style controls (shuffle, paraphrase, entropy-matched human data) to rule out "it was just narrow data," and the filter test at matched rejection rates: our mechanism filter vs. random vs. oracle. If random works as well, the method is fine and my mechanism is wrong — and that goes in the paper.

**Decisions I want from you.** Make the buffer-scaling result the centerpiece or keep the trace-concentration result? Run the misaligned-teacher arm in the main text or the appendix? And if no drift shows up on the Instruct models within three rounds — extend the horizon, or lead with the teacher-transmission result, which is guaranteed by the literature and still on theme?

All machinery is built: exact Fisher, the Gate R filter, activation probes, judge-based eval. Phase 2 is mostly these runs and reporting whatever they say.
