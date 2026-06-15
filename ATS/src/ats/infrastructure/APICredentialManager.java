package ats.infrastructure;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * API Key/Secret 보안 관리 (Design 2.2). 코드에 하드코딩하지 않고
 * .env 파일에서 로드·저장하며, Binance 요청 서명(HMAC-SHA256)을 담당.
 */
public class APICredentialManager {
    private final Path envPath = Path.of(".env");
    private String apiKey = "";
    private String apiSecret = "";
    private boolean testnet = true;

    /** .env 로드 (없으면 무시 → DEMO 모드) */
    public void load() {
        try {
            if (!Files.exists(envPath)) return;
            for (String line : Files.readAllLines(envPath)) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                int eq = line.indexOf('=');
                if (eq < 0) continue;
                String k = line.substring(0, eq).trim();
                String v = line.substring(eq + 1).trim();
                switch (k) {
                    case "BINANCE_API_KEY":    apiKey = v;    break;
                    case "BINANCE_API_SECRET": apiSecret = v; break;
                    case "BINANCE_TESTNET":    testnet = Boolean.parseBoolean(v); break;
                    default: break;
                }
            }
        } catch (Exception ignored) { }
    }

    public void set(String key, String secret, boolean testnet) {
        this.apiKey = key == null ? "" : key.trim();
        this.apiSecret = secret == null ? "" : secret.trim();
        this.testnet = testnet;
    }

    /** .env 저장 */
    public void save() throws Exception {
        Files.writeString(envPath,
                "BINANCE_API_KEY=" + apiKey + "\n" +
                "BINANCE_API_SECRET=" + apiSecret + "\n" +
                "BINANCE_TESTNET=" + testnet + "\n");
    }

    public boolean isConfigured() { return !apiKey.isBlank() && !apiSecret.isBlank(); }
    public boolean isTestnet()    { return testnet; }
    public String getApiKey()     { return apiKey; }
    public String getApiSecret()  { return apiSecret; }

    /** Binance 서명 (HMAC-SHA256, hex) */
    public String sign(String payload) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(apiSecret.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
            byte[] h = mac.doFinal(payload.getBytes(StandardCharsets.UTF_8));
            StringBuilder b = new StringBuilder();
            for (byte x : h) b.append(String.format("%02x", x));
            return b.toString();
        } catch (Exception e) {
            throw new IllegalStateException("HMAC 서명 실패", e);
        }
    }
}
