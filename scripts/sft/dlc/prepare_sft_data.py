#!/usr/bin/env python3
import argparse
import ast
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoProcessor

ROLE_MAP = {
    "user": "user", "human": "user", "prompt": "user",
    "assistant": "assistant", "gpt": "assistant", "model": "assistant",
    "tool": "tool", "function": "tool", "observation": "tool",
    "function_response": "tool", "tool_response": "tool",
}


def json_loads_maybe(v):
    if isinstance(v, (dict, list)):
        return v
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return v
    try:
        return json.loads(s)
    except Exception:
        return v


def text_of(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def normalize_arguments(v):
    v = json_loads_maybe(v)
    if isinstance(v, dict):
        return v
    if v is None:
        return {}
    return {"input": text_of(v)}


def normalize_tool_call(call):
    call = json_loads_maybe(call)
    if not isinstance(call, dict):
        return None
    if isinstance(call.get("function"), dict):
        call = call["function"]
    name = call.get("name") or call.get("tool_name") or call.get("function_name")
    if not name:
        return None
    args = call.get("arguments", call.get("parameters", call.get("args", {})))
    return {"type": "function", "function": {"name": str(name), "arguments": normalize_arguments(args)}}


def normalize_tools(v):
    v = json_loads_maybe(v)
    if not isinstance(v, list):
        return []
    out = []
    for t in v:
        t = json_loads_maybe(t)
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
        elif t.get("name"):
            fn = t
        else:
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        out.append({"type": "function", "function": {
            "name": str(name),
            "description": text_of(fn.get("description", "")),
            "parameters": params,
        }})
    return out


def split_top_level(s, sep=","):
    out, buf = [], []
    depth = 0
    quote = None
    esc = False
    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            buf.append(ch)
            quote = ch
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def parse_scalar(s):
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        if s.lower() in ("none", "null"):
            return None
        return s.strip("'\"")


def parse_function_expr(expr):
    expr = expr.strip()
    m = re.match(r"^(.+?)\((.*)\)$", expr, flags=re.S)
    if not m:
        return None
    name = m.group(1).strip()
    arg_s = m.group(2).strip()
    args = {}
    if arg_s:
        pos = 0
        for part in split_top_level(arg_s):
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = parse_scalar(v)
            else:
                args[f"arg{pos}"] = parse_scalar(part)
                pos += 1
    return {"type": "function", "function": {"name": name, "arguments": args}}


def parse_bracket_calls(text):
    s = text.strip()
    if s.startswith("API-Request:"):
        s = s[len("API-Request:"):].strip()
    if not (s.startswith("[") and s.endswith("]")):
        return []
    inner = s[1:-1].strip()
    if not inner:
        return []
    calls = []
    for part in split_top_level(inner):
        call = parse_function_expr(part)
        if call:
            calls.append(call)
    return calls


def parse_json_call_text(text):
    m = re.search(r"<functioncall>\s*(\{.*\})\s*$", text, flags=re.S)
    if not m:
        return None, text
    obj = json_loads_maybe(m.group(1))
    call = normalize_tool_call(obj)
    prefix = text[:m.start()].strip()
    return call, prefix


def extract_json_objects(s):
    out = []
    depth = 0
    start = None
    quote = None
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                obj = json_loads_maybe(s[start:i + 1])
                if isinstance(obj, dict):
                    out.append(obj)
                start = None
    return out


def tools_from_glaive_system(system):
    funcs = []
    for obj in extract_json_objects(system):
        if obj.get("name") and ("parameters" in obj or "description" in obj):
            funcs.append(obj)
    return normalize_tools(funcs)


def parse_glaive_chat(chat):
    pattern = re.compile(r"(?m)(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
    matches = list(pattern.finditer(chat))
    messages = []
    for i, m in enumerate(matches):
        role = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chat)
        content = chat[m.end():end].replace("<|endoftext|>", "").strip()
        if not content:
            continue
        if role == "USER":
            messages.append({"role": "user", "content": content})
        elif role == "FUNCTION RESPONSE":
            messages.append({"role": "tool", "content": content})
        else:
            call, prefix = parse_json_call_text(content)
            if call:
                messages.append({"role": "assistant", "content": prefix, "tool_calls": [call]})
            else:
                messages.append({"role": "assistant", "content": content})
    return messages


def api_bank_tool(v):
    obj = json_loads_maybe(v)
    if not isinstance(obj, dict) or not obj.get("apiCode"):
        return []
    props = {}
    required = []
    for k, spec in (obj.get("parameters") or {}).items():
        if not isinstance(spec, dict):
            spec = {"type": "string", "description": text_of(spec)}
        p = {kk: vv for kk, vv in spec.items() if kk != "required"}
        props[k] = p
        if spec.get("required") is True:
            required.append(k)
    params = {"type": "object", "properties": props}
    if required:
        params["required"] = required
    return normalize_tools([{
        "name": obj["apiCode"],
        "description": obj.get("description", ""),
        "parameters": params,
    }])


def normalize_message_list(v):
    v = json_loads_maybe(v)
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        if not isinstance(item, dict):
            continue
        raw_role = str(item.get("role", item.get("from", item.get("speaker", "")))).lower()
        content = item.get("content", item.get("value", item.get("text", "")))

        if raw_role in ("function_call", "tool_call"):
            call = normalize_tool_call(json_loads_maybe(content))
            if call:
                out.append({"role": "assistant", "content": "", "tool_calls": [call]})
            continue

        role = ROLE_MAP.get(raw_role)
        if not role:
            continue
        msg = {"role": role, "content": text_of(content)}

        if role == "assistant":
            tcs = item.get("tool_calls")
            if tcs:
                tcs = json_loads_maybe(tcs)
                if isinstance(tcs, list):
                    calls = [normalize_tool_call(x) for x in tcs]
                    calls = [x for x in calls if x]
                    if calls:
                        msg["tool_calls"] = calls
            if item.get("reasoning_content") is not None:
                msg["reasoning_content"] = text_of(item.get("reasoning_content"))

        out.append(msg)
    return out


def canonicalize_toolace(row):
    messages = []
    system = text_of(row.get("system", "")).strip()
    tools = normalize_tools(extract_json_objects(system))
    if system:
        messages.append({"role": "system", "content": "You are a helpful tool-using assistant." if tools else system})
    conv = json_loads_maybe(row.get("conversations"))
    if not isinstance(conv, list):
        return None
    for item in conv:
        role = str(item.get("from", "")).lower()
        value = text_of(item.get("value", "")).strip()
        if role in ("user", "human"):
            messages.append({"role": "user", "content": value})
        elif role in ("tool", "function", "observation"):
            messages.append({"role": "tool", "content": value})
        elif role in ("assistant", "gpt"):
            calls = parse_bracket_calls(value)
            if calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            else:
                messages.append({"role": "assistant", "content": value})
    return {"messages": messages, "tools": tools}


def canonicalize_apigen(row):
    tools = normalize_tools(row.get("tools"))
    messages = []
    system = text_of(row.get("system", "")).strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(normalize_message_list(row.get("conversations")))
    return {"messages": messages, "tools": tools}


def canonicalize_glaive(row):
    system = text_of(row.get("system", ""))
    tools = tools_from_glaive_system(system)
    messages = []
    if system:
        messages.append({"role": "system", "content": "You are a helpful assistant." if tools else system})
    messages.extend(parse_glaive_chat(text_of(row.get("chat", ""))))
    return {"messages": messages, "tools": tools}


def canonicalize_api_bank(row):
    tools = api_bank_tool(row.get("input"))
    user = text_of(row.get("instruction", "")).strip()
    output = text_of(row.get("output", "")).strip()
    messages = [{"role": "user", "content": user}]
    calls = parse_bracket_calls(output)
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:
        messages.append({"role": "assistant", "content": output})
    return {"messages": messages, "tools": tools}


def strip_toucan_system(messages, tools):
    if not messages or not tools or messages[0].get("role") != "system":
        return messages
    s = text_of(messages[0].get("content", ""))
    if "<|im_system|>tool_declare<|im_middle|>" in s:
        messages = list(messages)
        messages[0] = {"role": "system", "content": "You are a helpful tool-using assistant."}
    return messages


def canonicalize_generic(row, source):
    if source.startswith("smoltalk"):
        return {"messages": normalize_message_list(row.get("messages")), "tools": []}
    if source == "glaive":
        return canonicalize_glaive(row)
    if source == "toolace":
        return canonicalize_toolace(row)
    if source == "apigen_mt":
        return canonicalize_apigen(row)
    if source == "api_bank":
        return canonicalize_api_bank(row)

    tools = normalize_tools(row.get("tools", row.get("available_tools", [])))
    field = None
    for key in ("messages", "conversations", "trajectory"):
        if row.get(key) is not None:
            field = key
            break
    messages = normalize_message_list(row.get(field)) if field else []
    if source == "toucan":
        messages = strip_toucan_system(messages, tools)
    return {"messages": messages, "tools": tools}


def collect_called_names(messages):
    names = []
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            if fn.get("name"):
                names.append(str(fn["name"]))
        if m.get("role") == "assistant":
            names.extend(re.findall(r"<function=([^>]+)>", text_of(m.get("content", ""))))
    return set(names)


def prune_tools(tools, messages, max_tools):
    if not tools or len(tools) <= max_tools:
        return tools
    used = collect_called_names(messages)
    chosen = []
    rest = []
    for t in tools:
        name = str(t.get("function", {}).get("name", ""))
        (chosen if name in used else rest).append(t)
    return (chosen + rest)[:max_tools]


def valid_messages(messages):
    if not messages:
        return False
    if not any(m.get("role") == "assistant" for m in messages):
        return False
    if messages[0].get("role") == "tool":
        return False
    return True


def iter_parquet(path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield row


def iter_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            if isinstance(obj.get("data"), list):
                yield from obj["data"]
            else:
                yield obj
        return
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def source_specs(root):
    specs = []
    for p in sorted((root / "smoltalk").glob("*/data.parquet")):
        specs.append((f"smoltalk/{p.parent.name}", p, "parquet"))
    specs += [
        ("glaive", root / "glaive-function-calling-v2/data.parquet", "parquet"),
        ("toolace", root / "ToolACE/data.parquet", "parquet"),
        ("apigen_mt", root / "APIGen-MT-5k/apigen-mt_5k.json", "json"),
        ("agentinstruct", root / "AgentInstruct/data.parquet", "parquet"),
        ("agenttraj_l", root / "AgentTraj-L/data.parquet", "parquet"),
        ("toucan", root / "Toucan-1.5M/data.parquet", "parquet"),
        ("openhands_swegym", root / "OpenHands-SFT-Trajectories/data.parquet", "parquet"),
        ("swe_hero", root / "SWE-Hero-openhands-trajectories/data.parquet", "parquet"),
    ]
    for p in sorted((root / "API-Bank/training-data").glob("*-train.json")):
        specs.append(("api_bank", p, "json"))

    meta = root / "AgentBank/_meta.json"
    if meta.exists():
        obj = json.loads(meta.read_text(encoding="utf-8"))
        for f in obj.get("files", []):
            specs.append((f"agentbank/{f['config']}", root / "AgentBank" / f["path"], "parquet"))
    return [(s, p, k) for s, p, k in specs if p.exists()]


def find_subseq(seq, sub, start):
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i+n] == sub:
            return i
    return -1


def make_labels(input_ids, assistant_start, im_end_id):
    labels = [-100] * len(input_ids)
    pos = 0
    supervised = 0
    while True:
        s = find_subseq(input_ids, assistant_start, pos)
        if s < 0:
            break
        content_start = s + len(assistant_start)
        try:
            e = input_ids.index(im_end_id, content_start)
        except ValueError:
            break
        for j in range(content_start, e + 1):
            labels[j] = input_ids[j]
            supervised += 1
        pos = e + 1
    return labels, supervised


def build_generator(args, tokenizer, stats):
    root = Path(args.data_root)
    assistant_start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for source, path, kind in source_specs(root):
        base_source = source.split("/")[0]
        iterator = iter_parquet(path) if kind == "parquet" else iter_json_file(path)
        seen_source = 0
        for row in iterator:
            if args.max_records_per_source > 0 and seen_source >= args.max_records_per_source:
                break
            seen_source += 1
            stats[source]["input"] += 1
            try:
                canon = canonicalize_generic(row, base_source)
                messages = canon["messages"]
                tools = prune_tools(canon.get("tools", []), messages, args.max_tools)
                if not valid_messages(messages):
                    stats[source]["invalid"] += 1
                    continue
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tools=tools or None,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                enc = tokenizer(
                    rendered,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_length,
                    return_attention_mask=True,
                )
                ids = enc["input_ids"]
                labels, n_sup = make_labels(ids, assistant_start, im_end_id)
                if n_sup < args.min_assistant_tokens:
                    stats[source]["no_loss"] += 1
                    continue
                if len(ids) >= args.max_length:
                    stats[source]["truncated"] += 1
                stats[source]["kept"] += 1
                if stats[source]["input"] % 1000 == 0:
                    print(f"[{source}] input={stats[source]['input']} kept={stats[source]['kept']}", flush=True)
                yield {
                    "input_ids": ids,
                    "attention_mask": enc["attention_mask"],
                    "labels": labels,
                    "source": source,
                    "length": len(ids),
                }
            except Exception as e:
                stats[source]["error"] += 1
                if stats[source]["error"] <= 3:
                    print(f"[WARN] {source}: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/nas/bihaoran/agent_data")
    ap.add_argument("--model-path", default="/mnt/nas/bihaoran/common_agent/model/Qwen3.5-4B-Base")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-tools", type=int, default=32)
    ap.add_argument("--min-assistant-tokens", type=int, default=4)
    ap.add_argument("--max-records-per-source", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists() and (out / "dataset_info.json").exists() and not args.overwrite:
        print(f"[SKIP] tokenized dataset exists: {out}")
        return
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "right"

    stats = defaultdict(lambda: defaultdict(int))
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")),
        "source": Value("string"),
        "length": Value("int32"),
    })

    ds = Dataset.from_generator(
        lambda: build_generator(args, tokenizer, stats),
        features=features,
        cache_dir=str(out.parent / ".hf_cache"),
    )
    ds.save_to_disk(str(out))

    stat_obj = {k: dict(v) for k, v in stats.items()}
    stat_obj["total"] = {
        "rows": len(ds),
        "max_length": args.max_length,
        "max_tools": args.max_tools,
    }
    with open(out / "prepare_stats.json", "w", encoding="utf-8") as f:
        json.dump(stat_obj, f, ensure_ascii=False, indent=2)
    print(json.dumps(stat_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
