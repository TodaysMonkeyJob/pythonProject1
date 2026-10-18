"""
IoT security system entry point: starts every enabled module in parallel (see core/orchestrator.py).

    python main.py                 # all modules from config.yaml; Ctrl+C / SIGTERM to stop
    python main.py --only gas,vision
    python -m speech_to_text_whisper   # the original standalone speech-to-text recorder
"""
import argparse

from core.config import load_config
from core.orchestrator import Orchestrator
from system_setup.logger_setup import init_logger, set_logger_params


def main() -> None:
    parser = argparse.ArgumentParser(description="IoT security system orchestrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--only", help="comma-separated module names to run (default: all enabled)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    set_logger_params(init_logger(), log_level=args.log_level, no_log_file=True)
    config = load_config(args.config, env_file=args.env)
    specs = Orchestrator.module_specs(config)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        specs = [spec for spec in specs if spec.name in wanted]
        network = config.setdefault("network", {})  # per-device pipelines follow --only too
        network["discover_cameras"] = network.get("discover_cameras", True) and "vision" in wanted
        network["discover_mics"] = network.get("discover_mics", True) and "audio" in wanted
    Orchestrator(config, specs).run()


if __name__ == "__main__":
    main()
