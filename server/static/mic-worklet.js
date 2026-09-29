// Captures microphone audio, downsamples to 16 kHz mono int16 and posts
// 20 ms frames (320 samples) to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;          // fractional read position into incoming samples
    this.out = new Int16Array(320);
    this.n = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    // Linear-interpolation resampler; carries the fractional position across blocks.
    while (this.pos < input.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = input[i];
      const b = i + 1 < input.length ? input[i + 1] : input[i];
      const s = Math.max(-1, Math.min(1, a + (b - a) * frac));
      this.out[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === this.out.length) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.n = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= input.length;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
