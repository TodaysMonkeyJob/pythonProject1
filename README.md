# IoT Guard — fuzzy-bounded LLM security monitoring

Local security system for a Raspberry Pi hub (a Mac works for development): cameras, microphones and a gas sensor
feed one event bus; deterministic modules detect people, actions, identities, speech and gas; GPT explains the
situation, but a **fuzzy risk corridor over trusted physical evidence** decides how low the risk may go.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python main.py                      # whole system → http://127.0.0.1:8000
python -m pytest && ruff check core modules tests experiments
python -m experiments.run --all --seed 42 --scripted --llm sim   # evaluation harness (offline)
```

Details per module: `CLAUDE.md`, `modules/vision/README.md`, `modules/trust/README.md`, `modules/envelope/README.md`,
`core/fuzzy/README.md`, `dashboard/README.md`, `experiments/README.md`, `deploy/NETWORK.md`.

## Module dependencies

```mermaid
flowchart LR
  subgraph Devices
    CAM[Cameras / phone / RTSP]:::dev
    MIC[Microphones]:::dev
    GAS[Gas sensor]:::dev
  end
  subgraph Perception["Perception (one process per device)"]
    VIS[vision@cam<br/>YOLO + ByteTrack, motion]
    ACT[actions<br/>pose rules]
    IDN[identity<br/>YuNet + SFace, quality gate, dedup]
    AUD[audio@mic<br/>VAD, Whisper, ECAPA speakers]
    GASM[gas<br/>classifier + trend]
  end
  REG[(identity registry<br/>data/identity.db)]
  DEV[(device / location registry<br/>data/devices.db)]
  ORCH{{orchestrator router<br/>+ F16 trust scoring}}
  AGG[aggregator<br/>snapshots v2, CSV, incidents]
  OUT[(outbox.db)]
  LLM[LLM worker<br/>F19 gate · GPT · F17 corridor]
  LOC[locations<br/>state + notifications]
  DASH[dashboard<br/>control room]
  FUZ[[core/fuzzy<br/>config/fuzzy.yaml]]

  CAM --> VIS --> ACT & IDN
  MIC --> AUD
  GAS --> GASM
  IDN <--> REG
  AUD <--> REG
  VIS & ACT & IDN & AUD & GASM -->|events| ORCH
  ORCH -->|logs/&lt;date&gt;/events.jsonl| ORCH
  ORCH --> AGG --> OUT --> LLM
  ORCH -->|trusted evidence| LLM
  LLM -->|llm_decision| ORCH
  ORCH --> LOC & DASH
  DEV --- ORCH & LOC & DASH
  FUZ -.-> ORCH & AGG & LLM
  classDef dev fill:#eef,stroke:#88a
```

| Package | Depends on | Provides |
| -- | -- | -- |
| `core/` (events, bus, logger, config, module, orchestrator, devices, identity, fuzzy) | — | event schema v2, processes, registries, fuzzy engine |
| `modules/vision`, `modules/audio`, `modules/gas` | `core` | perception events |
| `modules/trust` | `core/fuzzy` | trust degree τ, provenance, claim type (F16) |
| `modules/aggregator` | `core`, `core/fuzzy` | snapshots, LLM view v2 (F18), outbox |
| `modules/envelope` | `core/fuzzy`, `modules/llm/calibration` | risk corridor + confirmation (F17) |
| `modules/llm` | `envelope`, `aggregator/outbox` | gating (F19), GPT client, calibration (F18) |
| `modules/locations`, `dashboard` | `core` registries | per-location state, notifications, UI |
| `experiments` | all of the above | evaluation harness (F20) |

## How a decision is made

```mermaid
flowchart TD
  E[Event from a device] --> T{F16 trust scoring<br/>τ = min(source, ASR, consistency)}
  T -->|claim lowers risk?| C[claim_type deescalation /<br/>instruction_to_system]
  T --> S[Aggregator snapshot<br/>fuzzified, trusted / untrusted evidence]
  C --> S
  S --> G{F19 gate}
  G -->|critical event · critical floor ·<br/>untrusted lowering claim| CALL[GPT call]
  G -->|novel / uncertain / refresh<br/>and interval allows| CALL
  G -->|routine| FZ[fuzzy-only decision<br/>risk = r_floor, rule summary]
  G -->|needed but interval not over| WAIT[wait — newer snapshot supersedes]
  CALL --> R[GPT: risk_level + risk_distribution]
  R --> K[F18 calibration<br/>r_LLM = centroid Σ p·μ]
  K --> P{F17 corridor<br/>r_final = max(r_LLM, (1 − c)·r_floor)}
  P -->|r_final level > GPT level| REJ[deescalation_rejected]
  P --> D[llm_decision → location state,<br/>notifications, dashboard]
  FZ --> D
  REJ --> D

  subgraph Corridor["F17 inputs (trusted evidence only, τ ≥ θ)"]
    F1[r_floor = MOM of the risk rules]
    F2[r_ceil = α-cut top, informational]
    F3[c = safety confirmation:<br/>clean gas, normal motion, known person,<br/>no stranger, no risky action · operator ack = 1]
  end
  Corridor -.-> P
```

Raising the risk is always allowed; lowering it below the floor needs trusted confirmation. Speech and scene text
never confirm safety — low-trust claims are quoted to GPT as untrusted evidence and listed as ignored.

## Privacy

Images, audio and embeddings stay on the device (`data/`, `logs/`, `exports/` are local); GPT receives only the
compact text / fuzzified JSON view, never media or raw sensor values. Secrets live in `.env` (never committed).
