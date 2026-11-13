#!/usr/bin/env python3
"""
dpuf_profile.py — In-container Profile & Rules Manager for DPUF Models.
Allows inspecting, modifying, adding, clearing, and exporting/importing 
user rules, persona, system prompt, and telemetry directly inside the .dpuf container.
"""

import sys
import os
import struct
import json
import zlib
import argparse

TOP_HDR_STRUCT = "<8sIIII"
TOP_HDR_SIZE = struct.calcsize(TOP_HDR_STRUCT)

def read_dpuf_header(filepath):
    with open(filepath, "rb") as f:
        hdr_bytes = f.read(TOP_HDR_SIZE)
        if len(hdr_bytes) < TOP_HDR_SIZE:
            raise ValueError("File is too small to be a DPUF file.")
        magic, num_tensors, hdr_crc, pay_crc, header_len = struct.unpack(TOP_HDR_STRUCT, hdr_bytes)
        if magic not in (b"DPUF0001", b"LRFLAC03"):
            raise ValueError(f"Invalid magic: {magic}")
        
        json_bytes = f.read(header_len)
        if len(json_bytes) < header_len:
            raise ValueError("Truncated JSON header.")
        
        computed_hcrc = zlib.crc32(json_bytes) & 0xFFFFFFFF
        if computed_hcrc != hdr_crc:
            print(f"Warning: Header CRC mismatch (stored 0x{hdr_crc:08X}, computed 0x{computed_hcrc:08X})")
            
        data = json.loads(json_bytes.decode("utf-8"))
        return {
            "magic": magic,
            "num_tensors": num_tensors,
            "hdr_crc": hdr_crc,
            "pay_crc": pay_crc,
            "header_len": header_len,
            "data": data,
            "payload_offset": TOP_HDR_SIZE + header_len
        }

def update_dpuf_profile(filepath, modifier_fn):
    """
    Reads DPUF, applies modifier_fn(profile_dict), recalculates header and writes back.
    Streaming copy ensures memory safety even on 50GB models.
    """
    info = read_dpuf_header(filepath)
    meta = info["data"].setdefault("__metadata__", {})
    profile = meta.setdefault("profile", {
        "system_prompt": "",
        "rules": [],
        "telemetry": {}
    })
    
    # Run user modification
    modifier_fn(profile)
    
    # Serialize new JSON
    new_json_bytes = json.dumps(info["data"], ensure_ascii=False, indent=None, separators=(',', ':')).encode("utf-8")
    new_header_len = len(new_json_bytes)
    new_hdr_crc = zlib.crc32(new_json_bytes) & 0xFFFFFFFF
    
    new_top_hdr = struct.pack(
        TOP_HDR_STRUCT,
        info["magic"],
        info["num_tensors"],
        new_hdr_crc,
        info["pay_crc"],
        new_header_len
    )
    
    tmp_path = filepath + ".tmp"
    with open(filepath, "rb") as f_in, open(tmp_path, "wb") as f_out:
        # Write updated top header & json
        f_out.write(new_top_hdr)
        f_out.write(new_json_bytes)
        
        # Stream payload without modifying a single weight byte
        f_in.seek(info["payload_offset"])
        chunk_size = 64 * 1024 * 1024
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            f_out.write(chunk)
            
    # Atomic replace
    os.replace(tmp_path, filepath)
    print(f"[dpuf_profile] Successfully updated '{filepath}' (new header: {new_header_len} bytes)")

def cmd_show(args):
    info = read_dpuf_header(args.model)
    meta = info["data"].get("__metadata__", {})
    model_name = meta.get("model_name", "Unknown")
    profile = meta.get("profile")
    
    print("=" * 60)
    print(f" DPUF Model Profile Inspector: {args.model}")
    print(f" Model Name   : {model_name}")
    print(f" Header Size  : {info['header_len']:,} bytes")
    print(f" Payload CRC  : 0x{info['pay_crc']:08X}")
    print("=" * 60)
    
    if not profile:
        print(" [No embedded profile found. Model is using default factory persona.]")
        return
        
    sys_prompt = profile.get("system_prompt", "")
    print(f"\n[System Prompt]:")
    print(sys_prompt if sys_prompt else "  (none)")
    
    rules = profile.get("rules", [])
    print(f"\n[Active User Rules ({len(rules)})]:")
    if rules:
        for idx, r in enumerate(rules, 1):
            print(f"  {idx}. {r}")
    else:
        print("  (no custom rules)")
        
    telemetry = profile.get("telemetry", {})
    if telemetry:
        print(f"\n[Telemetry & Tuning Stats]:")
        for k, v in telemetry.items():
            print(f"  {k}: {v}")
    print("=" * 60)

def cmd_set_system(args):
    prompt_text = args.text
    if os.path.isfile(prompt_text):
        with open(prompt_text, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
            
    def mod(prof):
        prof["system_prompt"] = prompt_text
    update_dpuf_profile(args.model, mod)
    print(f"System prompt set to:\n{prompt_text}")

def cmd_add_rule(args):
    rule_text = args.rule
    def mod(prof):
        rules = prof.setdefault("rules", [])
        rules.append(rule_text)
    update_dpuf_profile(args.model, mod)
    print(f"Added rule: '{rule_text}'")

def cmd_del_rule(args):
    target_idx = args.index - 1
    def mod(prof):
        rules = prof.get("rules", [])
        if 0 <= target_idx < len(rules):
            removed = rules.pop(target_idx)
            print(f"Removed rule #{args.index}: '{removed}'")
        else:
            print(f"Index {args.index} out of range (total rules: {len(rules)})")
    update_dpuf_profile(args.model, mod)

def cmd_clear(args):
    def mod(prof):
        prof["system_prompt"] = ""
        prof["rules"] = []
        prof["telemetry"] = {}
    update_dpuf_profile(args.model, mod)
    print("Cleared profile (reverted to factory default).")

def cmd_export(args):
    info = read_dpuf_header(args.model)
    meta = info["data"].get("__metadata__", {})
    profile = meta.get("profile", {})
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"Exported profile to '{args.output}'")

def cmd_import(args):
    with open(args.input, "r", encoding="utf-8") as f:
        new_prof = json.load(f)
    def mod(prof):
        prof.clear()
        prof.update(new_prof)
    update_dpuf_profile(args.model, mod)
    print(f"Imported profile from '{args.input}'")

def main():
    parser = argparse.ArgumentParser(description="Manage embedded profiles and rules in DPUF containers.")
    sub = parser.add_subparsers(dest="command", required=True)
    
    # show
    p_show = sub.add_parser("show", help="Show embedded profile, rules, and telemetry")
    p_show.add_argument("model", help="Path to .dpuf model")
    
    # set-system
    p_sys = sub.add_parser("set-system", help="Set the default system prompt")
    p_sys.add_argument("model", help="Path to .dpuf model")
    p_sys.add_argument("text", help="Prompt text or path to text/markdown file")
    
    # add-rule
    p_rule = sub.add_parser("add-rule", help="Add a user rule")
    p_rule.add_argument("model", help="Path to .dpuf model")
    p_rule.add_argument("rule", help="Rule string")
    
    # del-rule
    p_del = sub.add_parser("del-rule", help="Delete a user rule by 1-based index")
    p_del.add_argument("model", help="Path to .dpuf model")
    p_del.add_argument("index", type=int, help="Rule number to delete")
    
    # clear
    p_clr = sub.add_parser("clear", help="Clear all rules and system prompt")
    p_clr.add_argument("model", help="Path to .dpuf model")
    
    # export
    p_exp = sub.add_parser("export", help="Export profile to a JSON file")
    p_exp.add_argument("model", help="Path to .dpuf model")
    p_exp.add_argument("output", help="Path to output .json")
    
    # import
    p_imp = sub.add_parser("import", help="Import profile from a JSON file")
    p_imp.add_argument("model", help="Path to .dpuf model")
    p_imp.add_argument("input", help="Path to input .json")
    
    args = parser.parse_args()
    if args.command == "show":
        cmd_show(args)
    elif args.command == "set-system":
        cmd_set_system(args)
    elif args.command == "add-rule":
        cmd_add_rule(args)
    elif args.command == "del-rule":
        cmd_del_rule(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)

if __name__ == "__main__":
    main()
