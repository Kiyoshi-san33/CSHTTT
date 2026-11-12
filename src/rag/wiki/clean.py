"""Làm sạch MediaWiki markup -> văn bản thuần (thay cho wikiextractor đã 'patch' trong paper)."""
from __future__ import annotations

import re

import mwparserfromhell

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_TAGS_DROP = re.compile(r"<(gallery|math|score|timeline|syntaxhighlight|source|imagemap|table)[^>]*>.*?</\1>",
                        re.S | re.I)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_TAIL_SECTIONS = re.compile(
    r"^==+\s*(References|External links|See also|Further reading|Notes|Bibliography|Sources|Citations|Footnotes)\s*==+\s*$",
    re.M | re.I)
_FILE_PREFIX = re.compile(r"\[\[\s*(File|Image|Category|Media)\s*:", re.I)


class _M:
    def __init__(self, k: int):
        self._k = k

    def start(self) -> int:
        return self._k


def _remove_balanced(text: str, start_pat: re.Pattern | str, open_tok: str, close_tok: str) -> str:
    out, i = [], 0
    finder = (lambda s, pos: start_pat.search(s, pos)) if isinstance(start_pat, re.Pattern) else \
        (lambda s, pos: (lambda k: None if k < 0 else _M(k))(s.find(start_pat, pos)))
    while True:
        m = finder(text, i)
        if not m:
            out.append(text[i:])
            break
        s = m.start()
        out.append(text[i:s])
        depth, j = 0, s
        while j < len(text):
            if text.startswith(open_tok, j):
                depth += 1
                j += len(open_tok)
            elif text.startswith(close_tok, j):
                depth -= 1
                j += len(close_tok)
                if depth == 0:
                    break
            else:
                j += 1
        i = j
    return "".join(out)


def clean_wikitext(text: str) -> str:
    m = _TAIL_SECTIONS.search(text)
    if m:
        text = text[: m.start()]
    text = _COMMENT.sub("", text)
    text = _REF.sub("", text)
    text = _TAGS_DROP.sub("", text)
    text = _remove_balanced(text, "{|", "{|", "|}")
    text = _remove_balanced(text, _FILE_PREFIX, "[[", "]]")
    try:
        text = mwparserfromhell.parse(text).strip_code(normalize=True, collapse=True)
    except Exception:  # noqa: BLE001
        text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)
    text = _TAG.sub("", text)
    text = re.sub(r"'{2,}", "", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("thumb|", "|", "!"))]
    text = "\n".join(lines)
    return re.sub(r"[ \t]+", " ", text).strip()

