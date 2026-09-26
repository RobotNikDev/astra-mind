#!/usr/bin/env python3
"""
send_text_to_mac.py

Простой CLI для отправки текста на Mac (через ssh alias, например "astramind-mac"),
где Mac создаст аудиофайл через `say` и сразу его проиграет через `afplay`.

Поведение:
  1. Создаёт временный текстовый файл локально.
  2. Копирует его на Mac через scp.
  3. На Mac вызывает: say -f /tmp/xxx.txt -o /tmp/xxx.aiff && afplay /tmp/xxx.aiff
  4. Очищает временные файлы на Mac и локально.

Примеры:
  python send_text_to_mac.py --text "Привет, это тест" 
  python send_text_to_mac.py --text-file message.txt
  python send_text_to_mac.py --text "Привет" --host astramind-mac --keep-remote
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_HOST = "astramind-mac"


def run_cmd(cmd: list[str], capture_output: bool = False, check: bool = False, input_bytes: bytes | None = None):
    # Thin wrapper to run subprocess with clearer error messages
    try:
        res = subprocess.run(cmd, capture_output=capture_output, check=check, input=input_bytes)
        return res
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        if e.stdout:
            print(e.stdout.decode(errors="ignore"), file=sys.stderr)
        if e.stderr:
            print(e.stderr.decode(errors="ignore"), file=sys.stderr)
        raise


def ensure_tools():
    for tool in ("ssh", "scp"):
        if shutil.which(tool) is None:
            print(f"Error: required tool '{tool}' is not found in PATH.", file=sys.stderr)
            sys.exit(2)


def send_text_and_play(host: str, text_path_local: Path, remote_dir: str = "/tmp", keep_remote: bool = False) -> bool:
    """
    1) scp the local text file to remote_dir on host
    2) on remote, call say -f <text> -o <audio> && afplay <audio>
    3) remove remote files (unless keep_remote)
    """
    ts = int(time.time() * 1000)
    remote_text = f"{remote_dir}/astra_{ts}.txt"
    remote_audio = f"{remote_dir}/astra_{ts}.aiff"

    scp_cmd = ["scp", str(text_path_local), f"{host}:{remote_text}"]
    print("Copying text to Mac:", " ".join(scp_cmd))
    res = run_cmd(scp_cmd, capture_output=True)
    if res.returncode != 0:
        print("scp failed", file=sys.stderr)
        return False

    # Build remote command
    # Use -f to read text from file, -o to write aiff
    remote_cmd = f"say -f {remote_text} -o {remote_audio} && afplay {remote_audio}"
    if not keep_remote:
        remote_cmd += f" && rm -f {remote_text} {remote_audio}"
    # run ssh host <remote_cmd>
    ssh_cmd = ["ssh", host, remote_cmd]
    print("Running remote command:", " ".join(ssh_cmd))
    res2 = run_cmd(ssh_cmd, capture_output=True)
    if res2.returncode != 0:
        print("ssh remote command failed", file=sys.stderr)
        return False

    print("Remote play finished.")
    return True


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="send_text_to_mac.py", description="Send text to Mac and play via say/afplay")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", help="Text to send (UTF-8).")
    group.add_argument("--text-file", "-f", help="Path to local text file to send.")
    p.add_argument("--host", "-H", default=DEFAULT_HOST, help="ssh host alias (from ~/.ssh/config). Default: %(default)s")
    p.add_argument("--keep-remote", action="store_true", help="Keep remote files (do not delete on Mac).")
    p.add_argument("--no-delete-local", action="store_true", help="Do not remove the local temporary text file.")
    args = p.parse_args(argv)

    ensure_tools()

    # Prepare local text file
    if args.text:
        content = args.text
        local_tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, prefix="astra_text_", suffix=".txt")
        try:
            local_tmp.write(content)
            local_tmp.flush()
            local_tmp.close()
            local_path = Path(local_tmp.name)
        except Exception as e:
            print("Failed to write temp file:", e, file=sys.stderr)
            return 2
    else:
        # use provided file
        local_path = Path(args.text_file).expanduser()
        if not local_path.exists():
            print("Text file not found:", local_path, file=sys.stderr)
            return 2

    try:
        ok = send_text_and_play(args.host, local_path, keep_remote=args.keep_remote)
        if not ok:
            print("Failed to send/play text.", file=sys.stderr)
            return 3
    finally:
        if not args.no_delete_local:
            try:
                local_path.unlink(missing_ok=True)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())