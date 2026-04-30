from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime


def run_command(args: list[str]) -> int:
    print(f"[{datetime.now().isoformat()}] Run: {' '.join(args)}")
    completed = subprocess.run(args)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Schedule chatbot test suites.")
    parser.add_argument(
        "--plan",
        choices=["hourly-monitoring", "daily-full", "weekly-full"],
        default="hourly-monitoring",
        help="Scheduling plan",
    )
    parser.add_argument("--monitoring-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runner-mode",
        choices=["api", "direct"],
        default="api",
        help="Runner mode passed to run_test_suite.py",
    )
    parser.add_argument(
        "--send-report-email",
        action="store_true",
        help="Send report email after each run when SMTP is configured",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    args = parser.parse_args()

    if args.plan == "hourly-monitoring":
        interval_seconds = 3600
        cmd = [
            "python",
            "run_test_suite.py",
            "--mode",
            "monitoring",
            "--monitoring-size",
            str(args.monitoring_size),
            "--seed",
            str(args.seed),
            "--runner-mode",
            args.runner_mode,
        ]
    elif args.plan == "daily-full":
        interval_seconds = 86400
        cmd = [
            "python",
            "run_test_suite.py",
            "--mode",
            "full",
            "--runner-mode",
            args.runner_mode,
        ]
    else:
        interval_seconds = 86400 * 7
        cmd = [
            "python",
            "run_test_suite.py",
            "--mode",
            "full",
            "--runner-mode",
            args.runner_mode,
        ]

    if args.send_report_email:
        cmd.append("--send-report-email")

    while True:
        code = run_command(cmd)
        print(f"Exit code: {code}")

        if args.once:
            return

        print(f"Sleep {interval_seconds} seconds...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
