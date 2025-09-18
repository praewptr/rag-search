import re


def remove_citation_markers(text: str) -> str:
    return re.sub(r"\[doc\d+\]", "", text)
