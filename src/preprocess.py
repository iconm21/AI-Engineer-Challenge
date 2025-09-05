import re
import string

def clean_text(text: str) -> str:
    """Clean email text by removing URLs, numbers, punctuation; lowercasing; and trimming."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"\d+", "", text)       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()
