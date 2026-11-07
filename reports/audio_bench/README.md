# Audio subsystem latency (A4 / TOD-44)

`python -m tools.audio_bench` on Darwin arm64 (Python 3.12.3). **Not the Raspberry Pi** — run the same command there for the deployment numbers.

| Stage | Mean, ms | p95, ms | Budget |
| -- | -- | -- | -- |
| Microphone frame (background + noise profile + burst) | 0.026 | 0.057 | 30 ms per frame |
| Sound type of a 2 s clip (features + prediction) | 6.352 | 10.656 | own thread, per segment |
| Danger words of a transcript | 1.545 | 3.241 | per speech segment |
| Whisper tiny on 1.6 s of speech | 340 | — | real-time factor 0.21 (< 1 = faster than real time) |

The classifier was timed with a model trained on synthetic sounds (timing only).
