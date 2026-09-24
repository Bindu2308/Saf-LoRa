// interferer/main.cpp  (flash to the spare ESP32+SX1262 board)
//
// CONTROLLED INTERFERENCE SOURCE for HMM validation.
//
// Purpose: your HMM estimates whether each node-gateway link is in a
// GOOD or BAD interference state, but nothing in the deployment has
// ever told you what the true state was, so the estimates have never
// been checked. This board creates a KNOWN interference schedule and
// logs every transition, so belief can be scored against ground truth.
//
// Placement: physically close to Gateway 1 ONLY, not equidistant. The
// asymmetry is the experiment -- one gateway's links should degrade
// while the other's stay clean. If both degrade together, the run
// measures nothing about discrimination.
//
// Schedule: pseudo-random ON/OFF with a fixed seed, so the pattern is
// reproducible across runs but not periodic (a periodic jammer lets the
// HMM's transition prior do the work without observing anything).
//
// Output: one serial line per transition, e.g.
//     INTERFERER,<millis>,ON
//     INTERFERER,<millis>,OFF
// Capture with:
//     python -m serial.tools.miniterm /dev/ttyUSB0 115200 | tee interferer.log
// or in Arduino IDE, copy the Serial Monitor contents afterward.
//
// Align with server timestamps by noting the wall-clock time at the
// STARTED line -- millis() is boot-relative, not absolute.

#include <RadioLib.h>

#define LORA_CS   5
#define LORA_DIO1 26
#define LORA_RST  14
#define LORA_BUSY 27

SX1262 radio = new Module(LORA_CS, LORA_DIO1, LORA_RST, LORA_BUSY);

// Same PHY as the deployment, so the interference actually lands in
// band and is seen by the gateways' demodulator.
const float FREQ_MHZ      = 868.0;
const float BANDWIDTH_KHZ = 125.0;
const uint8_t SPREAD_FACTOR = 7;
const uint8_t CODING_RATE  = 5;
const uint8_t PREAMBLE_LEN = 8;
const uint8_t PACKET_SIZE  = 15;    // matches the deployment's implicit-header length

// Deliberately NOT the deployment's sync word (0x12). A different sync
// word means the gateways will not demodulate these as valid packets --
// they raise the noise floor and cause collisions, which is the point.
// Using 0x12 would inject well-formed garbage into the fragment table
// instead, which is a different (and much messier) experiment.
const uint8_t SYNC_WORD = 0x34;

const uint8_t TX_POWER_DBM = 14;

// Burst structure while ON: short packets back to back, so an ON window
// is genuinely occupied rather than a single brief hit.
const uint32_t BURST_GAP_MS = 40;

// ON/OFF window bounds, milliseconds. Windows are drawn uniformly from
// these ranges. ON windows are shorter than OFF so the channel is not
// saturated -- the HMM needs to see recovery as well as degradation.
const uint32_t ON_MIN_MS  = 8000;
const uint32_t ON_MAX_MS  = 25000;
const uint32_t OFF_MIN_MS = 20000;
const uint32_t OFF_MAX_MS = 60000;

const uint32_t SEED = 20260914;   // fixed: schedule is reproducible

bool interfering = false;
uint32_t windowEndsAt = 0;
uint8_t payload[PACKET_SIZE];

uint32_t randRange(uint32_t lo, uint32_t hi) {
    return lo + (uint32_t)random(0, (long)(hi - lo + 1));
}

void logTransition(const char* state) {
    Serial.print("INTERFERER,");
    Serial.print(millis());
    Serial.print(",");
    Serial.println(state);
}

void beginWindow() {
    if (interfering) {
        windowEndsAt = millis() + randRange(ON_MIN_MS, ON_MAX_MS);
        logTransition("ON");
    } else {
        windowEndsAt = millis() + randRange(OFF_MIN_MS, OFF_MAX_MS);
        logTransition("OFF");
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    randomSeed(SEED);
    for (uint8_t i = 0; i < PACKET_SIZE; i++) payload[i] = (uint8_t)random(0, 256);

    int state = radio.begin();
    if (state != RADIOLIB_ERR_NONE) {
        Serial.print("INTERFERER,0,RADIO_INIT_FAILED,code=");
        Serial.println(state);
        while (true) { delay(1000); }
    }

    radio.setFrequency(FREQ_MHZ);
    radio.setBandwidth(BANDWIDTH_KHZ);
    radio.setSpreadingFactor(SPREAD_FACTOR);
    radio.setCodingRate(CODING_RATE);
    radio.setSyncWord(SYNC_WORD);
    radio.setPreambleLength(PREAMBLE_LEN);
    radio.implicitHeader(PACKET_SIZE);
    radio.setCRC(true);
    radio.setOutputPower(TX_POWER_DBM);

    Serial.println("INTERFERER,0,STARTED");
    Serial.print("INTERFERER,0,CONFIG,freq=");
    Serial.print(FREQ_MHZ);
    Serial.print(",sf=");
    Serial.print(SPREAD_FACTOR);
    Serial.print(",power=");
    Serial.print(TX_POWER_DBM);
    Serial.print(",seed=");
    Serial.println(SEED);
    Serial.println("INTERFERER,0,NOTE,record wall-clock time now to align with server log");

    interfering = false;
    beginWindow();
}

void loop() {
    if ((int32_t)(millis() - windowEndsAt) >= 0) {
        interfering = !interfering;
        beginWindow();
    }

    if (interfering) {
        radio.transmit(payload, PACKET_SIZE);
        // Re-randomize so successive bursts are not identical, which
        // would let a receiver lock onto a repeating pattern.
        for (uint8_t i = 0; i < PACKET_SIZE; i++) payload[i] = (uint8_t)random(0, 256);
        delay(BURST_GAP_MS);
    } else {
        delay(50);
    }
}
