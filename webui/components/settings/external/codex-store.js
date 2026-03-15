import { createStore } from "/js/AlpineStore.js";
import * as API from "/js/api.js";

const model = {
  loading: false,
  polling: false,
  refreshing: false,
  applying: false,
  connected: false,
  expired: false,
  proxyRunning: false,
  proxyUrl: "",
  sessionId: "",
  userCode: "",
  verificationUri: "",
  pairingError: "",
  importJson: "",
  availableModels: [],
  accountInfo: null,
  chatModel: "gpt-5.3-codex",
  utilModel: "gpt-5.1-codex-mini",
  browserModel: "gpt-5.1-codex-mini",
  autoConfigure: true,

  async init() {
    await this.refresh();
  },

  async refresh() {
    this.loading = true;
    try {
      const [status, models] = await Promise.all([
        API.callJsonApi("codex_status", { action: "status" }),
        API.callJsonApi("codex_configure", { action: "get_models" }),
      ]);
      this.connected = !!status.connected;
      this.expired = !!status.expired;
      this.proxyRunning = !!status.proxy_running;
      this.proxyUrl = status.proxy_url || "";
      this.accountInfo = status.account_info || null;
      this.chatModel = status.models?.chat || this.chatModel;
      this.utilModel = status.models?.utility || this.utilModel;
      this.browserModel = status.models?.browser || this.browserModel;
      this.autoConfigure = status.auto_configure !== false;
      this.availableModels = models.models || [];
      this.pairingError = "";
    } catch (error) {
      console.error("Failed to refresh Codex status", error);
      this.pairingError = error.message || "Failed to load Codex status";
    } finally {
      this.loading = false;
    }
  },

  async startDeviceFlow() {
    this.loading = true;
    this.pairingError = "";
    try {
      const resp = await API.callJsonApi("codex_oauth", { action: "start_device_flow" });
      if (!resp.ok) throw new Error(resp.error || "Failed to start login");
      this.sessionId = resp.session_id;
      this.userCode = resp.user_code || "";
      this.verificationUri = resp.verification_uri || "";
      this.polling = true;
      this.pollDeviceFlow(resp.interval || 5);
    } catch (error) {
      console.error("Failed to start device flow", error);
      this.pairingError = error.message || "Failed to start login";
      this.polling = false;
    } finally {
      this.loading = false;
    }
  },

  async pollDeviceFlow(intervalSeconds = 5) {
    while (this.polling && this.sessionId) {
      try {
        const resp = await API.callJsonApi("codex_oauth", {
          action: "poll",
          session_id: this.sessionId,
        });
        if (!resp.ok) throw new Error(resp.error || "Authentication failed");
        if (resp.status === "complete") {
          this.polling = false;
          this.connected = true;
          this.accountInfo = resp.account_info || null;
          await this.refresh();
          break;
        }
        if (resp.status === "error" || resp.status === "expired") {
          this.polling = false;
          this.pairingError = resp.error || "Authentication failed";
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, (resp.interval || intervalSeconds) * 1000));
      } catch (error) {
        console.error("Device flow polling failed", error);
        this.polling = false;
        this.pairingError = error.message || "Authentication failed";
        break;
      }
    }
  },

  cancelPairing() {
    this.polling = false;
    this.sessionId = "";
    this.userCode = "";
    this.verificationUri = "";
  },

  async importTokens() {
    if (!this.importJson) return;
    this.loading = true;
    this.pairingError = "";
    try {
      const resp = await API.callJsonApi("codex_oauth", {
        action: "save_tokens",
        auth_json: this.importJson,
      });
      if (!resp.ok) throw new Error(resp.error || "Import failed");
      this.connected = true;
      this.accountInfo = resp.account_info || null;
      this.importJson = "";
      await this.refresh();
    } catch (error) {
      console.error("Failed to import tokens", error);
      this.pairingError = error.message || "Import failed";
    } finally {
      this.loading = false;
    }
  },

  async refreshToken() {
    this.refreshing = true;
    this.pairingError = "";
    try {
      const resp = await API.callJsonApi("codex_oauth", { action: "refresh" });
      if (!resp.ok) throw new Error(resp.error || "Refresh failed");
      this.accountInfo = resp.account_info || this.accountInfo;
      await this.refresh();
    } catch (error) {
      console.error("Failed to refresh token", error);
      this.pairingError = error.message || "Refresh failed";
    } finally {
      this.refreshing = false;
    }
  },

  async applyToAgent() {
    this.applying = true;
    this.pairingError = "";
    try {
      const resp = await API.callJsonApi("codex_configure", {
        action: "apply",
        chat_model: this.chatModel,
        util_model: this.utilModel,
        browser_model: this.browserModel,
        auto_configure: this.autoConfigure,
      });
      if (!resp.ok) throw new Error(resp.error || "Apply failed");
      this.proxyRunning = true;
      this.proxyUrl = resp.proxy_url || this.proxyUrl;
      const freshSettings = await API.callJsonApi("settings_get", null);
      if (freshSettings?.settings && window.Alpine?.store("settings")) {
        const settingsStore = window.Alpine.store("settings");
        settingsStore.settings = freshSettings.settings;
        settingsStore.additional = freshSettings.additional || settingsStore.additional;
      }
      await this.refresh();
    } catch (error) {
      console.error("Failed to apply Codex settings", error);
      this.pairingError = error.message || "Apply failed";
    } finally {
      this.applying = false;
    }
  },

  async stopProxy() {
    this.applying = true;
    this.pairingError = "";
    try {
      const resp = await API.callJsonApi("codex_configure", { action: "stop" });
      if (!resp.ok) throw new Error(resp.error || "Stop failed");
      
      const freshSettings = await API.callJsonApi("settings_get", null);
      if (freshSettings?.settings && window.Alpine?.store("settings")) {
        const settingsStore = window.Alpine.store("settings");
        settingsStore.settings = freshSettings.settings;
        settingsStore.additional = freshSettings.additional || settingsStore.additional;
      }
      this.proxyRunning = false;
      await this.refresh();
    } catch (error) {
      console.error("Failed to stop Codex proxy", error);
      this.pairingError = error.message || "Stop failed";
    } finally {
      this.applying = false;
    }
  },

  async disconnect() {
    this.loading = true;
    this.pairingError = "";
    try {
      const resp = await API.callJsonApi("codex_configure", { action: "disconnect" });
      if (!resp.ok) throw new Error(resp.error || "Disconnect failed");
      const freshSettings = await API.callJsonApi("settings_get", null);
      if (freshSettings?.settings && window.Alpine?.store("settings")) {
        const settingsStore = window.Alpine.store("settings");
        settingsStore.settings = freshSettings.settings;
        settingsStore.additional = freshSettings.additional || settingsStore.additional;
      }
      this.connected = false;
      this.expired = false;
      this.proxyRunning = false;
      this.proxyUrl = "";
      this.accountInfo = null;
      this.sessionId = "";
      this.userCode = "";
      this.verificationUri = "";
      await this.refresh();
    } catch (error) {
      console.error("Failed to disconnect Codex", error);
      this.pairingError = error.message || "Disconnect failed";
    } finally {
      this.loading = false;
    }
  },
};

const store = createStore("codexProvider", model);

export { store };
