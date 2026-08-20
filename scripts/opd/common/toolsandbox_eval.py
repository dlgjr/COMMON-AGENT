#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from tool_sandbox.cli import run_sandbox
from tool_sandbox.cli.utils import (
    AGENT_TYPE_TO_FACTORY,
    USER_TYPE_TO_FACTORY,
    RoleImplType,
    resolve_scenarios,
)
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser


class LocalStudentAgent(OpenAIAPIAgent):
    model_name = "qwen35-student"

    def __init__(self):
        self.openai_client = OpenAI(
            api_key="EMPTY",
            base_url=os.environ["STUDENT_OPENAI_URL"],
            timeout=300,
        )


class LocalTeacherUser(OpenAIAPIUser):
    model_name = "qwen35-teacher"

    def __init__(self):
        self.openai_client = OpenAI(
            api_key="EMPTY",
            base_url=os.environ["TEACHER_OPENAI_URL"],
            timeout=300,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--test-mode", action="store_true")
    args = ap.parse_args()

    role = RoleImplType.GPT_4_o_2024_05_13
    AGENT_TYPE_TO_FACTORY[role] = LocalStudentAgent
    USER_TYPE_TO_FACTORY[role] = LocalTeacherUser

    scenario_names = None
    if args.test_mode:
        from tool_sandbox.cli.utils import TEST_SCENARIO_NAMES
        scenario_names = TEST_SCENARIO_NAMES

    scenarios = resolve_scenarios(
        desired_scenario_names=scenario_names,
        preferred_tool_backend=ToolBackend.DEFAULT,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    before = set(output.glob("agent_*"))

    run_sandbox(
        agent_type=role,
        user_type=role,
        name_to_scenario=scenarios,
        processes=args.parallel,
        output_base_dir=output,
    )

    after = set(output.glob("agent_*"))
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if new_dirs:
        summary = new_dirs[-1] / "result_summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            latest = output / "latest_result_summary.json"
            latest.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(
                data.get("category_aggregated_results", {}),
                ensure_ascii=False,
                indent=2,
            ))


if __name__ == "__main__":
    main()
