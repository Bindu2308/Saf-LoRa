// gateway_esp32/main.cpp  (flash to Gateway 1's ESP32 via Arduino IDE)
//
// NEW in this version -- D-FRAG downlink ACK relay:
//   The RPi4 can now send a 7-byte downlink instruction over the SAME
//   TCP connection this gateway already uses for uplink observations:
//     [0xBB magic][node_id][telegram_id 4B, big-endian][success flag]
//   When received, this gateway transmits a matching 7-byte LoRa ACK
//   packet addressed to that node:
//     [0xAA magic][node_id][telegram_id 4B, big-endian][success flag]
//   which is exactly what transmitter_sx1262_main.cpp / transmitter_
//   sx1276_main.cpp's waitForAck() is already listening for.
//
// This gateway-side relay is complete and tested logically against the
// RPi4's main.py/gateway_server.py, which now sends this exact 7-byte
// format after every telegram finalizes (success or timeout).
//
// Also unchanged from before: uplink fragment forwarding. Transmitters
// now send total_fragments=5 (MDS-coded) instead of 2 -- this gateway
// doesn't need to know or care about that number, it just forwards
// whatever arrives, exactly as before.

#include <RadioLib.h>
#include <WiFi.h>

#define GATEWAY_ID 2
#define RPI4_PORT  5001

const char* WIFI_SSID = "bindu";
const char* WIFI_PASSWORD = "123456789";
const char* RPI4_HOST = "192.168.137.238"; // UPDATE if RPi4's IP differs

#define LORA_CS   5
#define LORA_DIO1 26
#define LORA_RST  14
#define LORA_BUSY 27

SX1262 radio = new Module(LORA_CS, LORA_DIO1, LORA_RST, LORA_BUSY);

const uint8_t FRAGMENT_WIRE_SIZE = 15;
const uint8_t OBS_PAYLOAD_SIZE = 8;
const uint8_t OBS_PROTOCOL_VERSION = 1;
const size_t OBS_WIRE_SIZE = 42;

// Downlink/uplink ACK relay constants
const uint8_t DOWNLINK_WIRE_SIZE = 7;   // [0xBB][node_id][telegram_id 4B][success]
const uint8_t DOWNLINK_MAGIC = 0xBB;
const uint8_t ACK_WIRE_SIZE = 7;         // [0xAA][node_id][telegram_id 4B][success] -- over LoRa
const uint8_t ACK_MAGIC = 0xAA;

WiFiClient tcpClient;

void buildObservation(uint8_t* out, const uint8_t* fragment, float rssi, float snr, bool crcOk) {
    size_t i = 0;
    out[i++] = OBS_PROTOCOL_VERSION;
    out[i++] = GATEWAY_ID;
    out[i++] = fragment[0]; // node_id

    uint32_t telegramId = (static_cast<uint32_t>(fragment[1]) << 24) |
                           (static_cast<uint32_t>(fragment[2]) << 16) |
                           (static_cast<uint32_t>(fragment[3]) << 8) |
                           static_cast<uint32_t>(fragment[4]);
    out[i++] = (telegramId >> 24) & 0xFF;
    out[i++] = (telegramId >> 16) & 0xFF;
    out[i++] = (telegramId >> 8) & 0xFF;
    out[i++] = telegramId & 0xFF;

    out[i++] = fragment[5]; // fragment_id
    out[i++] = fragment[6]; // total_fragments -- now typically 5 (MDS), gateway doesn't care what the number is

    uint64_t timestampMs = static_cast<uint64_t>(millis());
    for (int b = 7; b >= 0; --b) out[i++] = static_cast<uint8_t>((timestampMs >> (8*b)) & 0xFF);

    float receivedPowerDb = rssi;
    float noisePowerDb = rssi - snr;
    float syncConfidence = snr;
    float demodConfidence = snr;
    memcpy(&out[i], &receivedPowerDb, 4); i += 4;
    memcpy(&out[i], &noisePowerDb, 4); i += 4;
    memcpy(&out[i], &syncConfidence, 4); i += 4;
    memcpy(&out[i], &demodConfidence, 4); i += 4;

    out[i++] = crcOk ? 1 : 0;

    for (int k = 0; k < OBS_PAYLOAD_SIZE; ++k) {
        out[i++] = fragment[7 + k];
    }
}

bool ensureConnected() {
    if (tcpClient.connected()) return true;
    Serial.print("Connecting to RPi4 ");
    Serial.print(RPI4_HOST);
    Serial.print(":");
    Serial.println(RPI4_PORT);
    if (!tcpClient.connect(RPI4_HOST, RPI4_PORT)) {
        Serial.println("  connect failed");
        return false;
    }
    Serial.println("  connected");
    return true;
}

bool sendObservation(const uint8_t* obs) {
    if (!ensureConnected()) return false;

    uint8_t header[2] = { static_cast<uint8_t>((OBS_WIRE_SIZE >> 8) & 0xFF),
                           static_cast<uint8_t>(OBS_WIRE_SIZE & 0xFF) };
    size_t sentHeader = tcpClient.write(header, sizeof(header));
    size_t sentBody = tcpClient.write(obs, OBS_WIRE_SIZE);

    if (sentHeader != sizeof(header) || sentBody != OBS_WIRE_SIZE) {
        Serial.println("  send failed, will reconnect next time");
        tcpClient.stop();
        return false;
    }
    return true;
}

// Checks (non-blocking) for a downlink ACK-relay instruction from the
// RPi4, and if a full one has arrived, transmits the corresponding LoRa
// ACK to the node. Safe to call every loop iteration -- does nothing if
// no data is waiting.
void checkAndRelayDownlink() {
    if (!tcpClient.connected()) return;
    if (tcpClient.available() < DOWNLINK_WIRE_SIZE) return;  // wait for a full message

    uint8_t buf[DOWNLINK_WIRE_SIZE];
    size_t got = tcpClient.readBytes(buf, DOWNLINK_WIRE_SIZE);
    if (got != DOWNLINK_WIRE_SIZE || buf[0] != DOWNLINK_MAGIC) {
        Serial.println("  malformed downlink message, discarding");
        return;
    }

    uint8_t nodeId = buf[1];
    uint32_t telegramId = ((uint32_t)buf[2] << 24) | ((uint32_t)buf[3] << 16) |
                           ((uint32_t)buf[4] << 8) | buf[5];
    uint8_t success = buf[6];

    Serial.print("Relaying downlink ACK: node=");
    Serial.print(nodeId);
    Serial.print(" telegram=");
    Serial.print(telegramId);
    Serial.print(" success=");
    Serial.println(success);

    uint8_t ackPacket[ACK_WIRE_SIZE];
    ackPacket[0] = ACK_MAGIC;
    ackPacket[1] = nodeId;
    ackPacket[2] = (telegramId >> 24) & 0xFF;
    ackPacket[3] = (telegramId >> 16) & 0xFF;
    ackPacket[4] = (telegramId >> 8) & 0xFF;
    ackPacket[5] = telegramId & 0xFF;
    ackPacket[6] = success;

    // Briefly switch to transmit -- radio.transmit() handles mode
    // switching internally; the next radio.receive() call in the main
    // loop will switch back to receive mode automatically.
    int state = radio.transmit(ackPacket, ACK_WIRE_SIZE);
    if (state == RADIOLIB_ERR_NONE) {
        Serial.println("  ACK transmitted over LoRa");
    } else {
        Serial.print("  ACK transmit FAILED, code ");
        Serial.println(state);
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.print("ISAC-LoRa Gateway ");
    Serial.print(GATEWAY_ID);
    Serial.println(" (ESP32+SX1262, with D-FRAG downlink relay) starting...");

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("WiFi connected, IP: ");
    Serial.println(WiFi.localIP());

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

    Serial.println("Radio configured. Listening for LoRa packets + relaying downlink ACKs.");
}

void loop() {
    uint8_t fragment[FRAGMENT_WIRE_SIZE];
    int state = radio.receive(fragment, FRAGMENT_WIRE_SIZE);

    if (state == RADIOLIB_ERR_NONE) {
        float rssi = radio.getRSSI();
        float snr = radio.getSNR();

        Serial.print("Received: node=");
        Serial.print(fragment[0]);
        Serial.print(" telegram=");
        Serial.print((fragment[1]<<24)|(fragment[2]<<16)|(fragment[3]<<8)|fragment[4]);
        Serial.print(" frag=");
        Serial.print(fragment[5]);
        Serial.print("/");
        Serial.print(fragment[6]);
        Serial.print(" RSSI=");
        Serial.print(rssi);
        Serial.print(" SNR=");
        Serial.println(snr);

        uint8_t obs[OBS_WIRE_SIZE];
        buildObservation(obs, fragment, rssi, snr, true);

        if (!sendObservation(obs)) {
            Serial.println("  failed to forward observation to RPi4");
        }
    } else if (state == RADIOLIB_ERR_CRC_MISMATCH) {
        Serial.print("Packet received but CRC mismatch -- RSSI=");
        Serial.print(radio.getRSSI());
        Serial.print(" SNR=");
        Serial.println(radio.getSNR());
    } else if (state != RADIOLIB_ERR_RX_TIMEOUT) {
        Serial.print("radio.receive() returned unexpected state=");
        Serial.println(state);
    }

    // Check for a downlink ACK to relay -- runs every loop iteration,
    // interleaved with the LoRa receive above.
    checkAndRelayDownlink();
}
