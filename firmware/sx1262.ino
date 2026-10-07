// transmitter_sx1262/main.cpp  (flash to each of your 7 ESP32+SX1262 nodes)
// Use NODE_ID 4 through 10, one per board.
//
// NEW in this version:
//   1. MDS erasure coding (mds_encoder.h) -- message now splits into
//      K=3 data chunks, encoded into N=5 fragments. Any 3 of the 5
//      reaching a gateway are enough to reconstruct (verified end-to-end
//      against the real RPi4 decoder: C++ encode -> Python decode,
//      byte-for-byte correct, multiple fragment-loss patterns tested).
//   2. D-FRAG timing bandit (dfrag_bandit.h) -- an EXP3 no-regret learner
//      selects among 4 legal timing configs each cycle, using downlink
//      ACK success as reward (verified in isolation: correctly learns to
//      favor a better-performing action over ~2000 simulated rounds).
//
// *** IMPORTANT, READ BEFORE FLASHING ***
// The ACK-listening code below is REAL and will run, but there is NO
// gateway/receiver code yet that actually SENDS an ACK back down to this
// node -- that relay (RPi4 -> gateway -> node) has not been built. Until
// it is, every telegram will "time out" waiting for an ACK that never
// arrives, meaning:
//   - The bandit will only ever see reward=0, so it won't actually learn
//     anything yet (it'll keep exploring roughly uniformly) -- this is
//     SAFE (no crash, no wrong behavior), just not yet a working
//     learning system.
//   - Each telegram cycle will be slightly slower (waits out the full
//     ACK_TIMEOUT_MS before giving up).
// This is fine to flash now to validate MDS end-to-end; ask for the
// gateway+receiver ACK relay to make D-FRAG's learning actually real.

#include <RadioLib.h>
#include "mds_encoder.h"
#include "dfrag_bandit.h"

#define NODE_ID 4   // change to 5, 6, 7, 8, 9, 10 for the other boards

#define LORA_CS   5
#define LORA_DIO1 26
#define LORA_RST  14
#define LORA_BUSY 27

SX1262 radio = new Module(LORA_CS, LORA_DIO1, LORA_RST, LORA_BUSY);

const char* MESSAGE = "Hello isac-lora";
const uint8_t MESSAGE_LEN = 15;
const uint8_t FRAGMENT_PAYLOAD_SIZE = 8;
const uint8_t FRAGMENT_WIRE_SIZE = 15;

const uint32_t ACK_TIMEOUT_MS = 800;
const uint8_t ACK_WIRE_SIZE = 7;
const uint8_t ACK_MAGIC = 0xAA;

uint32_t telegramId = 0;
dfrag::Exp3Bandit bandit;

void buildFragmentHeader(uint8_t* out, uint8_t nodeId, uint32_t telegramId,
                          uint8_t fragmentId, uint8_t totalFragments) {
    out[0] = nodeId;
    out[1] = (telegramId >> 24) & 0xFF;
    out[2] = (telegramId >> 16) & 0xFF;
    out[3] = (telegramId >> 8) & 0xFF;
    out[4] = telegramId & 0xFF;
    out[5] = fragmentId;
    out[6] = totalFragments;
}

bool waitForAck(uint32_t thisTelegramId) {
    uint32_t start = millis();
    uint8_t buf[ACK_WIRE_SIZE];

    while (millis() - start < ACK_TIMEOUT_MS) {
        int state = radio.receive(buf, ACK_WIRE_SIZE);
        if (state == RADIOLIB_ERR_NONE) {
            if (buf[0] != ACK_MAGIC || buf[1] != NODE_ID) continue;
            uint32_t ackTelegramId = ((uint32_t)buf[2] << 24) | ((uint32_t)buf[3] << 16) |
                                      ((uint32_t)buf[4] << 8) | buf[5];
            if (ackTelegramId != thisTelegramId) continue;
            return buf[6] == 1;
        }
    }
    return false;
}

void sendTelegram() {
    uint8_t chunk_data[mds::K][FRAGMENT_PAYLOAD_SIZE] = {0};
    for (int i = 0; i < mds::K; i++) {
        for (uint8_t b = 0; b < FRAGMENT_PAYLOAD_SIZE; b++) {
            size_t idx = i * FRAGMENT_PAYLOAD_SIZE + b;
            chunk_data[i][b] = (idx < MESSAGE_LEN) ? (uint8_t)MESSAGE[idx] : 0;
        }
    }
    const uint8_t* chunks[mds::K];
    for (int i = 0; i < mds::K; i++) chunks[i] = chunk_data[i];

    uint8_t frag_data[mds::N][FRAGMENT_PAYLOAD_SIZE];
    uint8_t* fragments[mds::N];
    for (int i = 0; i < mds::N; i++) fragments[i] = frag_data[i];
    mds::mds_encode(chunks, fragments, FRAGMENT_PAYLOAD_SIZE);

    Serial.print("Sending telegram ");
    Serial.print(telegramId);
    Serial.print(" as ");
    Serial.print(mds::N);
    Serial.print(" MDS fragments (k=");
    Serial.print(mds::K);
    Serial.println(")");

    for (uint8_t fragId = 0; fragId < mds::N; fragId++) {
        uint8_t packet[FRAGMENT_WIRE_SIZE];
        buildFragmentHeader(packet, NODE_ID, telegramId, fragId, mds::N);
        memcpy(packet + 7, frag_data[fragId], FRAGMENT_PAYLOAD_SIZE);

        int state = radio.transmit(packet, FRAGMENT_WIRE_SIZE);
        if (state == RADIOLIB_ERR_NONE) {
            Serial.print("  fragment ");
            Serial.print(fragId);
            Serial.println(" sent OK");
        } else {
            Serial.print("  fragment ");
            Serial.print(fragId);
            Serial.print(" FAILED, code ");
            Serial.println(state);
        }
        delay(50);
    }

    bool acked = waitForAck(telegramId);
    bandit.update(acked ? 1.0f : 0.0f);
    Serial.print("  ACK result: ");
    Serial.println(acked ? "SUCCESS" : "timeout/no ACK (expected until gateway relay is built)");

    telegramId++;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.print("ISAC-LoRa node ");
    Serial.print(NODE_ID);
    Serial.println(" (SX1262, MDS + D-FRAG) starting...");

    randomSeed(analogRead(0) + NODE_ID);
    srand(analogRead(0) + NODE_ID);

    int state = radio.begin();
    if (state != RADIOLIB_ERR_NONE) {
        Serial.print("Radio init failed, code ");
        Serial.println(state);
        while (true) { delay(1000); }
    }

    radio.setFrequency(868.0);
    radio.setBandwidth(125.0);
    radio.setSpreadingFactor(7);
    radio.setCodingRate(5);
    radio.setSyncWord(0x12);
    radio.setPreambleLength(8);
    radio.implicitHeader(FRAGMENT_WIRE_SIZE);
    radio.setCRC(true);
    radio.setOutputPower(10);

    Serial.println("Radio configured. Beginning MDS + D-FRAG transmission loop.");
}

void loop() {
    sendTelegram();

    uint8_t action = bandit.select_action();
    dfrag::TimingConfig cfg = bandit.config_for(action);
    long jitter = random(-(long)cfg.jitter_range_ms, (long)cfg.jitter_range_ms);
    uint32_t delayMs = cfg.mean_delay_ms + jitter;

    Serial.print("  D-FRAG selected action ");
    Serial.print(action);
    Serial.print(" -> delay ");
    Serial.print(delayMs);
    Serial.println("ms");

    delay(delayMs);
}
