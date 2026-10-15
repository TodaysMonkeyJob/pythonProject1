# Human behaviour test videos (F2 / TOD-13)

Internet clips used to sanity-check the rule-based action recognition (`modules/vision/actions.py`).
They are edited footage (scene cuts, close-ups, handheld or zooming cameras), not a fixed security camera,
so they test the rules under harder conditions than the target setup.

| File | Length | Content |
|---|---|---|
| `Business footage diverse team …mp4` | 16 s | group standing around a table, night office |
| `Hilarious Office Moment …mp4` | 16 s, portrait | man asleep at the desk, wakes up; colleague standing |
| `Money Talks_ Napping at work.mp4` | 173 s, news report | anchor at a desk, nap chairs, street crowds, offices, capsule bed |
| `THOMAS SHELBY SMOKING …mp4` | 171 s, film montage | smoking close-ups, a cut every 2–5 s |
| `homeless people sleeping …Bangkok…mp4` | 8 s, portrait, handheld | people lying on the pavement, one sitting |
| `office work …mp4` | 13 s | people sitting at desks |

## Labels

`labels.csv` — `video,start,end,expected,note`: time ranges (seconds) where the action of the main
(largest) person is unambiguous. `a|b` accepts either label (reclined nap chairs). Interviews and face
close-ups are not labelled — posture cannot be read from shoulders only. `hand_to_mouth` marks smoking
shots: the montage never keeps one track for 30 s, so only the building block (wrist near the mouth) is
checked, not the full 3-episodes-in-30-s rule.

## Evaluate

```bash
python -m modules.vision.eval_actions data/human_behaviour_video/labels.csv \
    --model yolo11n.pt --pose-model yolo11n-pose.pt            # → reports/actions_eval.json
```
