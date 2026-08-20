import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

from swift.rollout.gym_env import Env, envs


def norm_name(value):
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9_]", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


_jsonl_cache = {}


def load_jsonl(path):
    path = str(path)
    if path in _jsonl_cache:
        return _jsonl_cache[path]
    rows = []
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    _jsonl_cache[path] = rows
    return rows


def scenario_map(path):
    return {norm_name(x.get("scenario", "")): x for x in load_jsonl(path)}


def parse_tool_call(text):
    text = (text or "").strip()
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    raw = m.group(1) if m else text
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def create_db(schema_record, sample_record, out_dir):
    scenario = norm_name(schema_record.get("scenario", "scenario"))
    db_path = Path(out_dir) / f"{scenario}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    for table in schema_record.get("db_schema", {}).get("tables", []):
        try:
            cur.execute(table["ddl"])
            for sql in table.get("indexes", []):
                cur.execute(sql)
            for sql in table.get("examples", []):
                cur.execute(sql)
        except Exception:
            pass
    conn.commit()

    try:
        cur.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
    sample = (sample_record or {}).get("sample_data", {})
    for table in sample.get("tables", []):
        for stmt in table.get("insert_statements", []):
            try:
                cur.execute(stmt)
            except Exception:
                pass
    conn.commit()
    conn.close()
    return db_path


def patch_generated_code(code, db_path):
    result = [
        "import warnings",
        "warnings.filterwarnings('ignore', category=DeprecationWarning)",
    ]
    for line in code.splitlines():
        if line.startswith("if __name__"):
            break
        if "create_engine(" in line:
            left = line.split("create_engine(", 1)[0]
            line = (
                f"{left}create_engine('sqlite:///{db_path}', "
                "connect_args={'check_same_thread': False})"
            )
        result.append(line)
    return "\n".join(result)


def resolve_ref(schema, components, depth=8):
    if not isinstance(schema, dict) or depth <= 0:
        return schema
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return resolve_ref(components.get(name, {}), components, depth - 1)
    out = {}
    for k, v in schema.items():
        if isinstance(v, dict):
            out[k] = resolve_ref(v, components, depth - 1)
        elif isinstance(v, list):
            out[k] = [resolve_ref(x, components, depth - 1) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def openapi_tools(app):
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    tools = {}
    for path, methods in spec.get("paths", {}).items():
        if path.startswith("/mcp") or path in (
            "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"
        ):
            continue
        for method, info in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            name = info.get("operationId", f"{method}_{path}")
            body = (
                info.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            tools[name] = {
                "method": method.upper(),
                "path": path,
                "description": info.get("summary") or info.get("description") or "",
                "schema": resolve_ref(body, components),
                "parameters": info.get("parameters", []),
            }
    return tools


def format_tools(tools):
    lines = ["Available scenario tools:"]
    for name, info in tools.items():
        lines.append(f"- {name}: {info.get('description', '')}")
        if info.get("schema"):
            lines.append("  JSON body schema: " + json.dumps(info["schema"], ensure_ascii=False))
        if info.get("parameters"):
            lines.append("  Query parameters: " + json.dumps(info["parameters"], ensure_ascii=False))
    return "\n".join(lines)


class AWMBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg["data_root"])
        self.scenario = norm_name(cfg["scenario"])
        self.task_idx = int(cfg.get("task_idx", 0))
        self.task = cfg.get("task", "")
        self.tmpdir = None
        self.client = None
        self.ns = None
        self.tools = {}

    async def reset(self, request):
        import httpx

        schemas = scenario_map(self.root / "gen_db.jsonl")
        samples = scenario_map(self.root / "gen_sample.jsonl")
        env_records = scenario_map(self.root / "gen_envs.jsonl")

        if self.scenario not in schemas or self.scenario not in env_records:
            raise RuntimeError(f"AWM data missing for scenario={self.scenario}")

        self.tmpdir = tempfile.mkdtemp(prefix=f"awm_{self.scenario}_")
        working = create_db(
            schemas[self.scenario],
            samples.get(self.scenario),
            self.tmpdir,
        )
        initial = Path(self.tmpdir) / "initial.db"
        shutil.copy2(working, initial)
        self.working_db = str(working)
        self.initial_db = str(initial)

        rec = env_records[self.scenario]
        code = rec.get("full_code") or rec.get("code") or ""
        patched = patch_generated_code(code, self.working_db)
        self.ns = {"__name__": f"awm_{self.scenario}_{self.task_idx}"}
        exec(compile(patched, f"<awm:{self.scenario}>", "exec"), self.ns)

        app = self.ns.get("app")
        if app is None:
            raise RuntimeError(f"AWM scenario {self.scenario} did not define a FastAPI app")

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://awm.local",
            timeout=30,
        )
        self.tools = openapi_tools(app)

        observation = (
            f"Task: {self.task}\n\n"
            "Use exactly one action per turn in this format:\n"
            '<tool_call>{"name":"list_scenario_tools","arguments":{}}</tool_call>\n'
            '<tool_call>{"name":"call_scenario_tool","arguments":{"tool_name":"TOOL_NAME","arguments":{}}}</tool_call>\n'
            '<tool_call>{"name":"submit","arguments":{"final_answer":""}}</tool_call>'
        )
        system = (
            "You are a stateful tool agent. Inspect the available tools, act on the current "
            "environment, recover from errors, verify completion, and submit only when done."
        )
        return observation, {
            "kind": "awm",
            "scenario": self.scenario,
            "task_idx": self.task_idx,
        }, system

    def verifier_record(self):
        for filename in ("gen_verifier.pure_code.jsonl", "gen_verifier.jsonl"):
            for row in load_jsonl(self.root / filename):
                if (
                    norm_name(row.get("scenario", "")) == self.scenario
                    and int(row.get("task_idx", -1)) == self.task_idx
                ):
                    return row
        return {}

    def verify(self, final_answer=""):
        row = self.verifier_record()
        verification = row.get("verification") or row
        code = verification.get("code", "")
        if not code:
            return False, "No verifier code"

        fn_name = "verify_task"
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("def verify_") and "(" in stripped:
                fn_name = stripped.split("(", 1)[0].replace("def ", "").strip()
                break

        ns = {"sqlite3": sqlite3, "json": json}
        try:
            exec(code, ns)
            fn = ns.get(fn_name)
            if fn is None:
                return False, f"Verifier function missing: {fn_name}"
            signature = ""
            if f"def {fn_name}" in code:
                signature = code.split(f"def {fn_name}", 1)[1].split(")", 1)[0]
            if "final_answer" in signature:
                result = fn(
                    initial_db_path=self.initial_db,
                    final_db_path=self.working_db,
                    final_answer=final_answer,
                )
            else:
                result = fn(
                    initial_db_path=self.initial_db,
                    final_db_path=self.working_db,
                )

            if isinstance(result, dict):
                marker = result.get("result", result)
                ok = bool(marker) and str(marker).lower() not in (
                    "false", "failed", "incomplete", "0", "none"
                )
            else:
                ok = bool(result)
            return ok, result
        except Exception as exc:
            return False, f"Verifier error: {exc}"

    async def step(self, messages):
        call = parse_tool_call(messages[-1].get("content", ""))
        name = call.get("name", "")
        arguments = call.get("arguments") or {}

        if name == "list_scenario_tools":
            return format_tools(self.tools), 0.0, False, {"action": name}

        if name == "call_scenario_tool":
            tool_name = arguments.get("tool_name", "")
            tool_args = arguments.get("arguments") or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            info = self.tools.get(tool_name)
            if info is None:
                return (
                    f"Unknown tool {tool_name}. Call list_scenario_tools first.",
                    0.0,
                    False,
                    {"action": name, "invalid_tool": True},
                )
            try:
                if info["method"] in ("POST", "PUT", "PATCH"):
                    response = await self.client.request(
                        info["method"], info["path"], json=tool_args
                    )
                else:
                    response = await self.client.request(
                        info["method"], info["path"], params=tool_args
                    )
                return response.text, 0.0, False, {
                    "action": tool_name,
                    "status": response.status_code,
                }
            except Exception as exc:
                return f"Tool error: {exc}", 0.0, False, {
                    "action": tool_name,
                    "error": str(exc),
                }

        if name == "submit":
            ok, result = self.verify(arguments.get("final_answer", ""))
            return (
                f"Verification result: {result}",
                1.0 if ok else 0.0,
                True,
                {"action": name, "verified": ok},
            )

        return (
            "Invalid action. Use one <tool_call> JSON action.",
            0.0,
            False,
            {"invalid_action": True},
        )

    async def close(self):
        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception:
                pass
        if self.ns:
            engine = self.ns.get("engine")
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


class ALFWorldBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = None

    async def reset(self, request):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        game_file = self.cfg["game_file"]
        infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )
        env_id = textworld.gym.register_games(
            [game_file],
            infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=int(self.cfg.get("episode_steps", 64)),
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        self.env = textworld.gym.make(env_id)
        obs, info = self.env.reset()
        text = obs[0] if isinstance(obs, (list, tuple)) else obs
        commands = (info or {}).get("admissible_commands", [])
        if commands and isinstance(commands[0], (list, tuple)):
            commands = commands[0]
        if commands:
            text += "\n\nAvailable commands:\n" + "\n".join(map(str, commands[:80]))

        system = (
            "You are acting in ALFWorld. Output exactly one environment command per turn. "
            "Do not add explanations around the command."
        )
        return text, {"kind": "alfworld", "game_file": game_file}, system

    async def step(self, messages):
        command = (messages[-1].get("content") or "").strip()
        m = re.search(r"<action>\s*(.*?)\s*</action>", command, re.S)
        if m:
            command = m.group(1).strip()
        if "\n" in command:
            command = next(
                (line.strip() for line in command.splitlines() if line.strip()),
                command,
            )

        try:
            obs, _score, done, info = self.env.step([command])
            text = obs[0] if isinstance(obs, (list, tuple)) else obs
            done_value = bool(done[0] if isinstance(done, (list, tuple)) else done)
            won = (info or {}).get("won", False)
            if isinstance(won, (list, tuple)):
                won = won[0]
            commands = (info or {}).get("admissible_commands", [])
            if commands and isinstance(commands[0], (list, tuple)):
                commands = commands[0]
            if commands and not done_value:
                text += "\n\nAvailable commands:\n" + "\n".join(map(str, commands[:80]))
            return text, 1.0 if bool(won) else 0.0, done_value, {
                "command": command,
                "won": bool(won),
            }
        except Exception as exc:
            return (
                f"Environment rejected command: {exc}",
                0.0,
                False,
                {"command": command, "error": str(exc)},
            )

    async def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


class WebShopBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = None

    async def reset(self, request):
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        self.env = WebAgentTextEnv(
            observation_mode="text",
            file_path=self.cfg["file_path"],
            num_products=int(self.cfg.get("num_products", 1000)),
            session=int(self.cfg.get("session", 0)),
        )
        obs, _ = self.env.reset(session=int(self.cfg.get("session", 0)))
        instruction = getattr(self.env, "instruction_text", "")
        available = self.env.get_available_actions()
        observation = (
            f"Instruction: {instruction}\n\nObservation:\n{obs}\n\n"
            f"Available actions: {available}"
        )
        system = (
            "You are acting in WebShop. Output exactly one action per turn using "
            "search[keywords] or click[value]."
        )
        return observation, {
            "kind": "webshop",
            "session": self.cfg.get("session", 0),
        }, system

    async def step(self, messages):
        action = (messages[-1].get("content") or "").strip()
        m = re.search(r"<action>\s*(.*?)\s*</action>", action, re.S)
        if m:
            action = m.group(1).strip()
        if "\n" in action:
            action = next(
                (line.strip() for line in action.splitlines() if line.strip()),
                action,
            )

        obs, reward, done, _ = self.env.step(action)
        available = self.env.get_available_actions() if not done else {}
        next_obs = (
            f"{obs}\n\nAvailable actions: {available}"
            if not done
            else obs
        )
        return next_obs, float(reward or 0.0), bool(done), {"action": action}

    async def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


class PromptOnlyBackend:
    def __init__(self, cfg):
        self.cfg = cfg

    async def reset(self, request):
        return (
            self.cfg.get("prompt", ""),
            {"kind": "prompt_only"},
            self.cfg.get(
                "system",
                "Solve the task carefully and provide the best final answer.",
            ),
        )

    async def step(self, messages):
        return "", 0.0, True, {"kind": "prompt_only"}

    async def close(self):
        pass


class CommonAgentEnv(Env):
    async def reset(self, config):
        kind = self.env_config.get("kind", "prompt_only")
        if kind == "awm":
            self.backend = AWMBackend(self.env_config)
        elif kind == "alfworld":
            self.backend = ALFWorldBackend(self.env_config)
        elif kind == "webshop":
            self.backend = WebShopBackend(self.env_config)
        else:
            self.backend = PromptOnlyBackend(self.env_config)
        return await self.backend.reset(config)

    async def step(self, action):
        return await self.backend.step(action)

    async def close(self):
        if hasattr(self, "backend"):
            await self.backend.close()


envs["common_agent"] = CommonAgentEnv
