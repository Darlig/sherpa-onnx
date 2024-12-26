package com.api;
import com.k2fsa.sherpa.onnx.*;

import java.sql.JDBCType;
import com.google.gson.Gson;
import java.util.HashMap;
import java.util.Map;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class TwoPassASR {
    private static final int sampleRate = 16000;
    private static final int windowSize = 512;

    private Vad vad;
    private OnlineRecognizer onlineRecognizer;
    private OfflineRecognizer offlineRecognizer;
    private OnlineStream onlineStream;
    private String lastOnlineText;
    private String onlineText;
    private String offlineText;
    private boolean vadDetectedSpeech;
    private boolean isOfflineResult;


    private static Vad createVad(String sileroVadModelPath) {
      String model = sileroVadModelPath;
      SileroVadModelConfig sileroVad =
          SileroVadModelConfig.builder()
              .setModel(model)
              .setThreshold(0.5f)
              .setMinSilenceDuration(0.01f)
              .setMinSpeechDuration(0.1f)
              .setWindowSize(windowSize)
              .build();
  
      VadModelConfig config =
          VadModelConfig.builder()
              .setSileroVadModelConfig(sileroVad)
              .setSampleRate(sampleRate)
              .setNumThreads(1)
              .setDebug(true)
              .setProvider("cpu")
              .build();
  
      return new Vad(config);
    }
  

    public void initModel(String configPath) {

        String json = "";
        try {
            json = Files.readString(Paths.get(configPath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        Gson gson = new Gson();
        Map<String, String> modelConfig = gson.fromJson(json, Map.class);
        String sileroVadModelPath = modelConfig.get("sileroVadModelPath");
        String onlineEncoderPath = modelConfig.get("onlineEncoderPath");
        String onlineDecoderPath = modelConfig.get("onlineDecoderPath");
        String onlineTokensPath = modelConfig.get("onlineTokensPath");
        String offlineModelPath = modelConfig.get("offlineModelPath");
        String offlineTokensPath = modelConfig.get("offlineTokensPath");
        // String ruleFstsPath = modelConfig.get("ruleFstsPath");
        String ruleFstsPath = "";
      
        vad = createVad(sileroVadModelPath);

        // Load Online Recognizer
        OnlineParaformerModelConfig paraformer =
            OnlineParaformerModelConfig.builder()
                .setEncoder(onlineEncoderPath)
                .setDecoder(onlineDecoderPath)
                .build();

        OnlineModelConfig onlineModelConfig =
            OnlineModelConfig.builder()
                .setParaformer(paraformer)
                .setTokens(onlineTokensPath)
                .setNumThreads(1)
                .setDebug(true)
                .build();

        OnlineRecognizerConfig onlineConfig =
            OnlineRecognizerConfig.builder()
                .setOnlineModelConfig(onlineModelConfig)
                .setDecodingMethod("greedy_search")
                .build();

        onlineRecognizer = new OnlineRecognizer(onlineConfig);
        onlineStream = onlineRecognizer.createStream();

        // Load Offline Recognizer
        OfflineParaformerModelConfig offlineParaformer =
            OfflineParaformerModelConfig.builder().setModel(offlineModelPath).build();

        OfflineModelConfig offlineModelConfig =
            OfflineModelConfig.builder()
                .setParaformer(offlineParaformer)
                .setTokens(offlineTokensPath)
                .setNumThreads(1)
                .setDebug(true)
                .build();

        OfflineRecognizerConfig offlineConfig =
            OfflineRecognizerConfig.builder()
                .setOfflineModelConfig(offlineModelConfig)
                .setDecodingMethod("greedy_search")
                .setRuleFsts(ruleFstsPath)
                .build();

        offlineRecognizer = new OfflineRecognizer(offlineConfig);

        lastOnlineText = "";
        vadDetectedSpeech = false;
    }

    public void putData(float[] samples) {
        // Process Online Recognizer
        onlineStream.acceptWaveform(samples, 16000);
        while (onlineRecognizer.isReady(onlineStream)) {
            onlineRecognizer.decode(onlineStream);
        }
        onlineText = onlineRecognizer.getResult(onlineStream).getText();
        if (!onlineText.isEmpty() && !onlineText.equals(" ") && !onlineText.equals(lastOnlineText)) {
            // System.out.println("online: " + onlineText);
            lastOnlineText = onlineText;
        }

        // Process VAD
        vad.acceptWaveform(samples);
        vadDetectedSpeech = vad.isSpeechDetected();
        // System.out.println("vadDetectedSpeech: " + vadDetectedSpeech);

        offlineText = "";
        isOfflineResult = false;
        while (!vad.empty()) {
          isOfflineResult = true;
          onlineRecognizer.reset(onlineStream);
          SpeechSegment segment = vad.front();
          float startTime = segment.getStart() / (float) sampleRate;
          float duration = segment.getSamples().length / (float) sampleRate;
  
          OfflineStream stream = offlineRecognizer.createStream();
          stream.acceptWaveform(segment.getSamples(), sampleRate);
          offlineRecognizer.decode(stream);
          offlineText = offlineRecognizer.getResult(stream).getText();
          stream.release();
  
          vad.pop();
        }
    }

    public String getResult() {
        Map<String, String> response = new HashMap<>();
        response.put("2passOnline", onlineText);
        response.put("getOfflineResult", String.valueOf(isOfflineResult));
        response.put("2passOffline", offlineText);
        Gson gson = new Gson();
        return gson.toJson(response);
    }

    public void releaseModel() {
        if (onlineStream != null) onlineStream.release();
        if (onlineRecognizer != null) onlineRecognizer.release();
        if (vad != null) vad.release();
        if (offlineRecognizer != null) offlineRecognizer.release();
    }

}
