"""tools/hf_remote.sh 的凭据过滤测试（暂无契约 ID：运行规范待成文）。只用标准库，在 Mac 上跑。

临时目录当仓库根，放真的 hf_remote.sh 与 scripts/dev/_load-env，一份假的 .env（假 token、带用户名密码的假代理地址），
PATH 前面放一个假的 ssh（把最后一个参数交给本机 bash 执行，stdin 照传）。远端命令故意把 token 与代理地址打出来：
  1. 输出里没有 token，也没有代理的密码；不含凭据的行照常输出；退出码照传远端命令的；
  2. 远端读不到两行凭据时报“没读到凭据”。
用法：python3 tools/test_hf_remote.py
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN = "hf_ABCDEFGHIJKLMNOPQRSTUVWX"
PROXY = "http://someuser:secretpw@proxy.example:3128"
FAKE_SSH = '#!/usr/bin/env bash\ncmd="${@: -1}"\nexec bash -c "$cmd"\n'


def run(cmd: str, env_text: str = None) -> tuple:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "tools").mkdir()
        (root / "scripts/dev").mkdir(parents=True)
        (root / "bin").mkdir()
        (root / "remote").mkdir()
        shutil.copy(HERE / "hf_remote.sh", root / "tools")
        shutil.copy(HERE.parent / "scripts/dev/_load-env", root / "scripts/dev")
        (root / ".env").write_text(env_text if env_text is not None else
                                   f"HF_TOKEN={TOKEN}\nV2D_PROXY={PROXY}\nV2D_HOST=fakehost\nV2D_REMOTE={root / 'remote'}\n")
        (root / "bin/ssh").write_text(FAKE_SSH)
        (root / "bin/ssh").chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "V2D_PROXY", "V2D_ENV")}
        env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
        r = subprocess.run(["bash", str(root / "tools/hf_remote.sh"), cmd], env=env, capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout + r.stderr


def test_credentials_filtered() -> None:
    rc, out = run('echo "token=$HF_TOKEN"; echo "proxy=$https_proxy"; echo "endpoint=$HF_ENDPOINT"; echo plain line; exit 3')
    assert TOKEN not in out and "secretpw" not in out and "someuser" not in out, out
    assert "plain line" in out and "endpoint=https://huggingface.co" in out and rc == 3, (rc, out)


def test_missing_credentials() -> None:
    rc, out = run("echo should not run", env_text="HF_TOKEN=\nV2D_PROXY=\nV2D_HOST=fakehost\nV2D_REMOTE=/tmp\n")
    assert rc != 0 and "缺 HF_TOKEN 或 V2D_PROXY" in out and "should not run" not in out, (rc, out)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
    print("ALL_OK")
