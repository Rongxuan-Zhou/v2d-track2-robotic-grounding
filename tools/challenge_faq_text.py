"""下载赛事官网的 FAQ（PDF）并打出 sha256 与全文（Mac 上跑；只读公开网页，不带任何凭据）。
用途：引用官方规则原文时留底（记忆 cite-rules-verbatim）。10/17 用户问“Track 2 冠军怎么定”，官网正文没写各指标怎么合，FAQ 里写了。
要 pypdf：UV_NO_CONFIG=1 UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --no-project --python 3.12 --with pypdf python -I tools/challenge_faq_text.py
加 --sentences 时合并空白、每个问答一行。最后一行 FAQ_TEXT_DONE <页数> <字数>。
"""
import hashlib
import io
import re
import sys
import urllib.request

from pypdf import PdfReader

URL = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf"


def main() -> int:
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    print(f"来源 {URL}")
    print(f"大小 {len(data)} 字节，sha256 {hashlib.sha256(data).hexdigest()}")
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    if "--sentences" in sys.argv:
        # 按问句切行：pypdf 抽出的文字一词一行，合并空白后每个问题与它的回答放在同一行，便于逐条引用
        flat = re.sub(r"\s+", " ", text)
        print("---- 全文（空白合并，按问号断行）----")
        for part in re.split(r"(?<=[.!]) (?=[A-Z][^?]{3,160}\?)", flat):
            print(part.strip())
    else:
        print("---- 全文（pypdf 抽取，空白已合并成单个空格）----")
        print(re.sub(r"[ \t]+", " ", text))
    print(f"FAQ_TEXT_DONE {len(reader.pages)} {len(text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
