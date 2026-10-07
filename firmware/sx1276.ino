// transmitter_sx1276/main.cpp  (flash to each of your 3 ESP32+SX1276 nodes)
// Use NODE_ID 1, 2, 3, one per board.
//
// Same MDS + D-FRAG additions as transmitter_sx1262/main.cpp -- see that
// file's header comment for the full explanation, including the
// *** IMPORTANT *** note that the ACK downlink relay doesn't exist yet.

#include <RadioLib.h>
#include <SPI.h>
#include "mds_encoder.h"
#include "dfrag_bandit.h"

#define NODE_ID 1   // change to 2, 3 for the other boards

#define LORA_MISO 19
#define LORA_MOSI 27
#define LORA_SCK  5
#define LORA_NSS  18
#define LORA_RST  14
#define LORA_DIO0 26
#define LORA_DIO1 35
#define LED1      2
#define LED2      13

SX1276 radio = new Module(LORA_NSS, LORA_DIO0, LORA_RST, LORA_DIO1);

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

        digitalWrite(LED1, HIGH);
        int state = radio.transmit(packet, FRAGMENT_WIRE_SIZE);
        digitalWrite(LED1, LOW);

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
    Serial.println(" (SX1276, MDS + D-FRAG) starting...");

    pinMode(LED1, OUTPUT);
    pinMode(LED2, OUTPUT);

    randomSeed(analogRead(0) + NODE_ID);
    srand(analogRead(0) + NODE_ID);

    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);

    int state = radio.begin();
    if (state != RADIOLIB_ERR_NONE) {
        Serial.print("Radio init failed, code ");
        Serial.println(state);
        while (true) {
            digitalWrite(LED2, !digitalRead(LED2));
            delay(300);
        }
    }

    radio.setFrequency(868.0);
    radio.setBandwidth(125.0);
    radio.setSpreadingFactor(7);
    radio.setCodingRate(5);
    radio.setSyncWord(0x12);
    radio.setPreambleLength(8);
    radio.implicitHeader(FRAGMENT_WIRE_SIZE);
    radio.setCRC(true);

    Serial.println("Radio configured. Beginning MDS + D-FRAG transmission loop.");
    digitalWrite(LED2, HIGH);
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
