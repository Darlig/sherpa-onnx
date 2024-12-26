
import com.api.TwoPassASR;

import javax.sound.sampled.*;

public class TestTwoPassASR {
    private static final int sampleRate = 16000;
    private static final int windowSize = 512;
  
  public static void main(String[] args) {
    TwoPassASR vadFromMic = new TwoPassASR();
    vadFromMic.initModel("./model_config.json");

    AudioFormat format = new AudioFormat(sampleRate, 16, 1, true, false);

    DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
    TargetDataLine targetDataLine;
    try {
      targetDataLine = (TargetDataLine) AudioSystem.getLine(info);
      targetDataLine.open(format);
      targetDataLine.start();
    } catch (LineUnavailableException e) {
      System.out.println("Failed to open target data line: " + e.getMessage());
      vadFromMic.releaseModel();
      return;
    }

    byte[] buffer = new byte[windowSize * 2];
    float[] samples = new float[windowSize];

    System.out.println("Started. Please speak");
    boolean running = true;
    while (targetDataLine.isOpen() && running) {
      int n = targetDataLine.read(buffer, 0, buffer.length);
      if (n <= 0) {
        System.out.printf("Got %d bytes. Expected %d bytes.\n", n, buffer.length);
        continue;
      }
      for (int i = 0; i != windowSize; ++i) {
        short low = buffer[2 * i];
        short high = buffer[2 * i + 1];
        int s = (high << 8) + low;
        samples[i] = (float) s / 32768;
      }

      vadFromMic.putData(samples);
      String result = vadFromMic.getResult();
      System.out.println(result);
    }

    vadFromMic.releaseModel();
  }
}
