// dfrag_bandit.h -- node-side timing selection, four policies.
//
// The review requires D-FRAG's learning claim to be tested against
// non-learning baselines under matched load. A rolling ACK-success
// curve on its own cannot establish learning: a fixed policy under
// improving channel conditions produces a similar-looking curve.
//
// Select the policy at COMPILE TIME with DFRAG_POLICY below, then flash
// all nodes with that setting and run one full session. Repeat per
// policy. Compile-time rather than runtime because runtime selection
// would need a new downlink message type parsed by both the gateway and
// the server, and because a compile-time constant cannot desync
// mid-run.
//
//   DFRAG_FIXED  always action 0. The control: no adaptation at all.
//   DFRAG_RANDOM uniform over actions. Tests whether EXP3's advantage
//                is real learning or just the jitter that any
//                randomized policy provides.
//   DFRAG_RR     deterministic rotation. Tests the same thing without
//                randomness.
//   DFRAG_EXP3   the proposed learner.
//
// Only EXP3 consumes the reward, so update() is a no-op for the other
// three. Call it unconditionally; the policy decides what to do.

#ifndef DFRAG_BANDIT_H
#define DFRAG_BANDIT_H

#include <Arduino.h>
#include <math.h>

#define DFRAG_FIXED   0
#define DFRAG_RANDOM  1
#define DFRAG_RR      2
#define DFRAG_EXP3    3

// ---- SET THIS PER CAMPAIGN RUN, then reflash all nodes ----
#ifndef DFRAG_POLICY
#define DFRAG_POLICY DFRAG_EXP3
#endif

namespace dfrag {

struct TimingConfig {
    uint32_t mean_delay_ms;
    uint32_t jitter_range_ms;
};

// Four duty-cycle-legal timing configurations. Kept identical across
// policies so the comparison is matched-load: the policies differ only
// in HOW they pick, never in WHAT is available to pick.
static const uint8_t NUM_ACTIONS = 4;
static const TimingConfig CONFIGS[NUM_ACTIONS] = {
    {4000,  500},
    {5000, 1000},
    {6000, 1500},
    {7000, 2000},
};

static const float GAMMA = 0.1f;               // EXP3 exploration rate
static const float OVERFLOW_THRESHOLD = 1e6f;

class Exp3Bandit {
public:
    Exp3Bandit() {
        for (uint8_t i = 0; i < NUM_ACTIONS; i++) weights_[i] = 1.0f;
    }

    uint8_t select_action() {
#if DFRAG_POLICY == DFRAG_FIXED
        last_action_ = 0;
        last_prob_ = 1.0f;
        return 0;

#elif DFRAG_POLICY == DFRAG_RANDOM
        last_action_ = (uint8_t)random(0, NUM_ACTIONS);
        last_prob_ = 1.0f / NUM_ACTIONS;
        return last_action_;

#elif DFRAG_POLICY == DFRAG_RR
        last_action_ = rr_next_;
        rr_next_ = (uint8_t)((rr_next_ + 1) % NUM_ACTIONS);
        last_prob_ = 1.0f / NUM_ACTIONS;
        return last_action_;

#else   // DFRAG_EXP3
        float W = 0.0f;
        for (uint8_t i = 0; i < NUM_ACTIONS; i++) W += weights_[i];

        float probs[NUM_ACTIONS];
        for (uint8_t i = 0; i < NUM_ACTIONS; i++) {
            probs[i] = (1.0f - GAMMA) * (weights_[i] / W) + GAMMA / NUM_ACTIONS;
        }

        float r = (float)random(0, 10000) / 10000.0f;
        float cum = 0.0f;
        uint8_t chosen = NUM_ACTIONS - 1;   // fallback for float rounding
        for (uint8_t i = 0; i < NUM_ACTIONS; i++) {
            cum += probs[i];
            if (r <= cum) { chosen = i; break; }
        }

        last_action_ = chosen;
        last_prob_ = probs[chosen];
        return chosen;
#endif
    }

    // Call ONLY when a real response arrived. A timeout carries no
    // information about the chosen action -- scoring silence as failure
    // injects noise unrelated to the decision. The caller enforces this;
    // see waitForAck()'s AckResult in the transmitter firmware.
    void update(float reward) {
#if DFRAG_POLICY == DFRAG_EXP3
        float r_hat = reward / last_prob_;
        weights_[last_action_] *= expf(GAMMA * r_hat / NUM_ACTIONS);

        float W = 0.0f;
        for (uint8_t i = 0; i < NUM_ACTIONS; i++) W += weights_[i];
        if (W > OVERFLOW_THRESHOLD) {
            for (uint8_t i = 0; i < NUM_ACTIONS; i++) weights_[i] /= W;
        }
#else
        (void)reward;   // non-learning policies ignore the reward
#endif
    }

    TimingConfig config_for(uint8_t action) const {
        return CONFIGS[action < NUM_ACTIONS ? action : 0];
    }

    // Printed at startup so every serial log records which policy
    // produced it. Without this, a mis-flashed board is invisible until
    // the data is analysed.
    static const char* policy_name() {
#if DFRAG_POLICY == DFRAG_FIXED
        return "FIXED";
#elif DFRAG_POLICY == DFRAG_RANDOM
        return "RANDOM";
#elif DFRAG_POLICY == DFRAG_RR
        return "ROUND-ROBIN";
#else
        return "EXP3";
#endif
    }

private:
    float weights_[NUM_ACTIONS];
    uint8_t last_action_ = 0;
    float last_prob_ = 1.0f / NUM_ACTIONS;
    uint8_t rr_next_ = 0;
};

}  // namespace dfrag

#endif  // DFRAG_BANDIT_H
