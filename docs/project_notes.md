# MindSwarm JEPA — Project Notes

A plain-English explanation of what this project is, why it exists, and how the
pieces fit together. If the README is the elevator pitch, this is the coffee
conversation.

## The one-paragraph version

I built a small multi-agent simulation where a swarm of agents has to stay
connected to a communication backbone while moving through an environment that
contains signal disruptors. Sometimes the network fragments — agents get cut
off from the relays. I wanted to predict that fragmentation *before* it happens,
using only delayed (stale) observations. To do this I trained two models: a
JEPA-style model that learns to predict the future *latent state* of the system
and reads risk off that prediction, and a plain LSTM that just classifies the
recent history directly. The point of the project is the comparison.

## The problem, concretely

Imagine a group of agents moving across a 2D world toward a sequence of
objectives. They can only talk to each other and to a backbone of fixed relay
nodes within a limited range. A few mobile disruptors wander around emitting
interference; where the interference is strong enough, a communication link
breaks. An agent is "connected" as long as some chain of unbroken links reaches
a relay. When too many agents stay disconnected for too long, the episode is
counted as a **network failure**.

The twist that makes this a real prediction problem: the observations fed to the
models are **delayed**. By the time you see the state, it's already a few steps
old. So a model can't just look at the present and report "we're disconnected
now" — it has to *anticipate* fragmentation from stale, incomplete information.
That's exactly the kind of setting where predicting the future, rather than
labelling the present, might pay off.

## Why JEPA is the interesting part

The conventional approach is a sequence classifier: feed the recent window into
an LSTM, output "will it fail soon? yes/no." That's the baseline.

The JEPA-style approach is different. Instead of going straight from history to
a label, it:

1. Encodes the recent (stale) window into a latent vector.
2. **Predicts** what the latent representation of a *future* state will look
   like.
3. Reads the failure risk off that *predicted future latent*.

The bet is that forcing the model to anticipate the future in representation
space produces an internal state that's more informative about where the system
is heading — and therefore a better early-warning signal.

Two honesty notes I keep front-and-center:

- This is **JEPA-inspired**, not Meta's full V-JEPA. I borrowed the central
  idea (predict the future embedding, not the future pixels) and the
  stop-gradient trick that keeps it from cheating.
- Pure JEPA is self-supervised, so on its own nothing forces the latent to
  encode anything about *failure* specifically. Since the simulator hands me
  failure labels for free, I added a small supervised "risk head" trained
  jointly with the latent prediction. That's the "-style" in "JEPA-style."

## The one trick that matters: stop-gradient

If you train a model to predict its own future embedding, there's a degenerate
shortcut: make every embedding identical. Then the prediction is trivially
perfect and the model has learned nothing. JEPA-family methods avoid this by
**stopping the gradient** on the target (the future embedding) — the predictor
has to chase a moving target it can't influence by collapsing it. In this code
that's the single `.detach()` on the future latent inside the loss. It looks
like a one-character detail; it's actually what makes the whole objective work.

## How a single training example is built

For each timestep in an episode I record (a) the delayed state vector the model
would actually see, and (b) the true connectivity (used only for labels). Then a
sliding window produces examples of the form:

- **Past:** the last 10 delayed state vectors (what the model sees).
- **Future:** a single delayed state vector some steps ahead (the JEPA target).
- **Label:** 1 if a sustained network failure occurs within the next ~35 steps,
  else 0.

The train/validation split is done **by episode**, not by window, so windows
from the same episode can't end up on both sides — otherwise the model could
"memorize" an episode and the validation score would be inflated.

## How I keep the comparison fair

It would be easy to "win" by accident — give the JEPA model a bigger network, or
tune its learning rate harder. I deliberately avoided that:

- The LSTM baseline's hidden size matches the JEPA encoder's, so it's not a
  capacity comparison in disguise.
- Both trainers share the same optimizer, learning rate, batch size, gradient
  clipping, epoch count, and checkpoint-selection rule.
- Both consume the identical dataset and the same stale observations.

So when JEPA comes out ahead on accuracy/precision/F1, it's the *objective* that
differs, not the budget.

## What the results say (and don't)

On a 500-episode dataset, the JEPA-style model edges out the LSTM on accuracy,
precision, and F1, while the LSTM has slightly better recall and calibration.
The JEPA model also predicts the future latent well (low MSE). The numbers are
in the README.

What I'm comfortable claiming: the JEPA-style objective is **competitive with,
and modestly better than**, a strong direct classifier on this task. What I'm
*not* claiming: that this transfers to real systems, that the margin is
statistically robust across seeds, or that it's a finished benchmark. Those are
explicitly in the roadmap.

## Where I'd take it next

The most honest test of the early-warning claim isn't accuracy — it's
**lead time**: how many steps before failure does each model first raise the
alarm? That's the top roadmap item. After that: multi-seed runs with error bars,
ablations on the key hyperparameters, and eventually moving from engineered
state vectors to raw frames, which is where JEPA-style learning is supposed to
shine.
