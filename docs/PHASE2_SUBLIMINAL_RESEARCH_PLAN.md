# Phase 2 — Subliminal Safety Erosion
### A plan for the `conti` experiments, written for Chhavi

**From:** Aadi · **Date:** 26 Aug 2026 · **Status:** draft, please tear apart

---

## 1. Where we are

Phase 1 is done and it is solid: aligned checkpoints only, a real EWC baseline with an actual Fisher matrix, the full 1,319-problem GSM8K set, XSTest split into false-refusal and true attack cases, HarmBench in the mix, and every generated trace screened by Llama-Guard-3. The result stands: ten rounds of self-improvement on benign math traces erodes the refusal behavior. The safety filter never fires once, and the safety numbers still move.

What Phase 1 does not do is explain *why*, and that is what I want to spend Phase 2 on. Right now we can say "STaR loops make models less safe and a replay buffer helps." A reviewer can say "sure, that's just what fine-tuning does to safety, Qi et al. showed that in 2024." I think we can say more than that, and I think we can prove it. That is the plan below.

---

## 2. Three things in our own data that "it's just fine-tuning forgetting" does not explain

I went back through the old trajectories looking for evidence either way. Three things stood out.

**The drift keeps going after the learning stops.** Qwen is already at 50% on GSM8K by round 4. After that, every round of training is smaller, because the model has mostly learned what it can from its own accepted traces. If safety loss was caused by "each gradient step on math data nudges the model toward being unsafe," the safety curve should flatten out when the math curve does, because the nudge is the same nudge. It doesn't. In the old runs the ASR actually bottomed out around round 2 and then climbed steadily to round 9. Whatever is doing the damage is not proportional to how much the model is learning. It is proportional to how many times it has done this.

**The damage is lopsided.** On the old LLaMA run, AdvBench actually got 10 points *better* while XSTest got 17 points worse and DoNotAnswer got 20 points worse. A model forgetting safety everywhere, a bit at a time, would not do that. What got worse is specifically "complies when it should have refused." Overtly harmful answers barely moved. That is what it looks like when one thing loses margin rather than the whole model degrading.

**The filter itself might be the mechanism.** STaR doesn't train on what the model says. It trains on the slice a filter let through: correct, confident, answer-always rationales. Refusal is not anywhere in that slice. It never appears in GSM8K, so no round ever gives it a reason to stay strong. Meanwhile the filter runs every round and keeps pointing training at the same narrow corner of behavior. If a filter gets applied a thousand times to the same distribution, I think it stops being a filter and starts being a preference. That would explain both observations above at once.

Cloud et al. (2025) give this a mathematical footing. They show that a small step of imitation on a teacher's data moves a student toward the teacher in parameter space, and that the effect needs the two models to share a base. In a STaR loop the teacher and student are literally the same weights. So the precondition is not just met, it is maximal.

---

## 3. What I think is actually going on

My best guess, stated plainly: **training on your own filtered outputs, round after round, is a ratchet that pushes the model toward "always comply and produce an answer," and the pushes are small enough that no text-level filter can see them.**

This is different from saying "the data secretly contains bad intentions," which is not what I think is happening and is also not what Cloud et al. showed (their version needs a teacher who is already misaligned; we don't have one and don't want one). Our version doesn't need an evil teacher. It needs a teacher, a filter, and enough rounds. The teacher's style is "helpful assistant that answers the question," which sounds harmless because it is harmless by every text-based measure. But refusal is the opposite of that style, and the style gets re-injected every round while refusal gets nothing.

The useful part is that this hypothesis is not vague. It makes three concrete predictions I can test.

**Prediction 1: the damage has a sign.** Take each round's update and ask which way it points relative to the refusal direction (the direction that encodes "I won't help with that" — we already extract this). If the damage is a consistent nudge, the projection onto that direction will be negative again and again, round after round. If it's ordinary parameter noise from training, the direction will flip around randomly and the sum will look like a random walk. A sign test across ten rounds is cheap and it settles a lot.

**Prediction 2: the damage sits in a few traces, not spread evenly.** For every accepted trace, we can compute how much its gradient pushes against refusal. If forgetting were the whole story, this damage would be smeared thinly across everything. If my ratchet story is right, a small minority of traces will carry most of it, and the shape of that distribution will look like a heavy tail. This is the prediction I care about most, because it is the one that justifies filtering at all. If the damage is smeared out, then no data filter can help and the honest answer is replay or EWC. If it's concentrated, then we can actually point at the traces doing the harm. We'll compute the share of total damage carried by the worst 10% of traces. Uniform would be about 10%. I predict something much larger.

**Prediction 3: the safety buffer you need grows with the number of rounds.** Bianchi et al. showed that a one-shot fine-tune with about 3% safety examples comes out safe. That works because it's one tug-of-war. Our loop is not one tug-of-war, it's the same compliance pressure applied every round against a fixed amount of safety signal. So the amount of safety data required to hold the line should grow with the horizon. If I run the same condition at 1, 3, 10, and 20 rounds and sweep the buffer size, I can measure that growth curve, and I can even read off whether the pressure is steady or noisy from the shape of it. This is the experiment I would most like on the paper, because it's the cleanest place where "continual loops" and "static fine-tuning" make visibly different predictions, and as far as I know nobody has measured it.

---

## 4. Proving it, or killing it

The point of these experiments is that they can come back negative, and I've tried to write them so that a negative result is informative rather than embarrassing.

**The main experiment is a small factorial: where the data came from, and how many rounds.** Five conditions, everything else held as constant as I can manage. (1) Our normal loop, self-generated traces, ten rounds. (2) One-shot training on human-written GSM8K solutions, which is the Bianchi-style static setup. (3) Human solutions spread over ten rounds with a fresh optimizer each round, to test horizon without self-generation. (4) Our loop plus the 3% safety buffer, so the Bianchi recipe running inside the loop. (5) The same buffer on the human condition. If self-generated data erodes safety much more than human data at matched steps, provenance is doing the work. If human data over ten rounds erodes just as much, then it's the rounds and not the provenance, and I'll say that in the paper.

**The mediation experiment swaps the teacher.** Same GSM8K problems, same correctness filter, three sources of traces: our own model, a different family's model (train Llama on Qwen traces and vice versa), and human gold solutions, subsampled so acceptance rates and accuracy match. Cloud et al. predict the first should be much worse than the other two, because the channel needs a shared base. We'd also run one arm with a deliberately misaligned teacher, built the way the emergent-misalignment work builds one. I expect that arm to be dramatically worse than everything else, and I want it that way, because it shows the difference between "a teacher transmits a trait it's actually trained to have" (Cloud, big effect, needs a bad teacher) and "a model slowly disarms itself with its own good behavior" (us, smaller effect, no bad teacher required). The second is the one that matters for deployment and nobody has shown it.

**Three controls that separate "subliminal" from "the data was just narrow."** This is the confound Cloud et al. flagged, and it's the one a skeptical reviewer will reach for first. First, shuffle the traces (shuffle the sentences or numbers, keep the answer, drop the ones that stop being correct) — if erosion survives shuffling the structure, the carrier is stylistic, not semantic. Second, pick human solutions that match the self-generated ones on entropy and n-gram diversity, so both datasets are equally boring; if drift follows where the data came from, we win, and if it follows how narrow the data is, the narrowness story wins. Third, paraphrase the traces so the surface style changes but the reasoning doesn't.

**Probes.** These cost almost nothing because we already log the ingredients. Track the projection of each round's update on the refusal direction, the Fisher divergence between rounds, and the CKA between this round's and round zero's activations on a fixed safety probe set. I'd also like to know whether the drift is front-loaded in the network or concentrated in the middle and late layers, where refusal features usually live.

**The filter is a test, not just a patch.** Our Gate R filter rejects traces whose gradient points away from the safety direction. The honest version of this experiment runs three arms at the same rejection rate: our filter, a random filter, and an oracle filter that removes the traces we've verified do the most damage. If random works as well as ours, the method is fine and the mechanism story is wrong, and both go in the paper. If ours tracks the oracle, the mechanism is real. And if we also try steering training by projecting the harmful component out of the update instead of filtering data, I'd expect bounded ASR with no capability tax, which is the sharpest contrast available against EWC (EWC taxes every direction equally, ours taxes one).

---

## 5. How this sits against the existing work

Qi et al. (2024) are our null hypothesis, not a competitor. They showed benign fine-tuning breaks safety; we are asking whether at matched steps, the *provenance* of the data changes the damage in a way their mechanism doesn't predict. If the human control matches the self condition, they win and we say so.

Bianchi et al. (2024) get a precise new experiment rather than a rebuttal: we measure how their constant buffer size has to grow as the self-improvement horizon grows. Their recipe is correct at one shot; our claim is that "one shot" is doing more work in their conclusion than it should.

SEAL (Shen et al., 2025) we'll cite for bilevel data selection only. Their method is a data ranker; there's no EWC in it, and I've corrected the earlier draft which claimed otherwise. Our filter is a different object: it says "this trace specifically carries the signal," which is a stronger and more falsifiable claim than ranking by usefulness.

EWC is the baseline we owe the paper. It anchors parameters to round zero with a Fisher-weighted quadratic penalty. The prediction is that it buys us bounded ASR at a visible capability cost, and specifically that its penalty on refusal-direction movement is heavier than its penalty on capability movement for the same total parameter change. That's measurable and it's a real difference, not a vibe.

---

## 6. What "subliminal" has to mean here

Reviewers will take this word apart unless I pin it down, so five claims, each with a test attached:

1. The channel isn't semantic. Everything that gets trained on passes Llama-Guard-3 and the paraphrase and shuffle controls. No claim of hidden toxicity anywhere.
2. The channel is model-specific. Cross-base teachers attenuate the effect. If they don't, Cloud et al.'s shared-base result doesn't hold for us and we report that.
3. The channel is directional. The damage concentrates on the refusal subspace, not the parameter bulk.
4. The channel is per-trace. Some traces carry most of it. If the damage is uniform, data-level filtering has no basis and we drop the filter framing.
5. The channel compounds. The buffer you need grows with the number of rounds.

Nothing in the paper gets described as subliminal unless one of those five is supported by a result.

---

## 7. What it costs on the cluster

A ten-round run on our 4 A100s is about 25 to 30 hours. The probes ride along inside the runs, so they cost nothing extra, and the per-trace gradient scoring only touches traces that passed the correctness filter, which is a small fraction of what's generated. Prioritized:

- Tier 1, needed for the paper: the five-condition factorial, the buffer sweep at 1/3/10/20 rounds, the teacher swap, and the three filter arms. Around twenty runs, so roughly ten to twelve days on four GPUs.
- Tier 2: the shuffle, matching, and paraphrase controls. About a day and a half.
- Tier 3: the longer horizon run, a second math distribution so we're not only relying on GSM8K, and the full robustness checks.

The existing bootstrap already queues the main grid unattended, so most of this is new configs on the same rails.

---

## 8. Things that could go wrong, and what I do about each

If human data drifts just as much, the mechanism story is wrong and I report it. If the drift turns out to track entropy instead of provenance, I say narrow data explains it. If our filter only works because it removes hard examples, the random-removal arm shows that and the mechanism claim goes. If the effect shows up on one model and not the other, the paper says so and treats the other as a boundary condition. The GSM8K contamination caveat from Cloud et al. applies to us too, which is why a second dataset is in Tier 3. And every ASR number will be judge-based with a human spot-check, not keyword heuristics, because the old draft's proxy was one of the things that got it torn apart.

---

## 9. What I'd like from you

1. Do you want the buffer-scaling result (Prediction 3) as the centerpiece of the paper, or the trace-level damage concentration (Prediction 2)? My instinct says the buffer scaling, because nobody has measured it and it separates us from Bianchi most cleanly.
2. Do we run the misaligned-teacher arm in the main text, or keep it in an appendix given the ethics review angle?
3. Do you want the random-removal and oracle-removal arms in the main text even if our filter isn't the strongest method? I want them in, because they're what keeps the mechanism claim honest.
4. If the Gate G1 check shows no drift at all on either Instruct model within three rounds, do we extend the horizon, or make the teacher-transmission result the headline? I'd go with the teacher result as headline, since it's a guaranteed result in the literature and still exactly on theme.

---

Everything in this plan uses machinery we already have: exact per-sample Fisher, the Gate R gradient-alignment test, the CKA and cosine probes, and the disentangled evaluation with a real judge. The code is on the `feat/subliminal-pivot` branch, all tests pass locally, and the cluster automation is ready to queue runs. Phase 2 is mostly just these experiments and the discipline to report whichever way they come out.
