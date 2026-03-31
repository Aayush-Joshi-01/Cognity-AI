"""Simple token estimation utilities (word-based approximation)."""


def estimate_tokens(text: str, method: str = "words") -> int:
    """
    Estimate token count for a text string.

    method='words': word count (fast, ~1.3 tokens/word heuristic)
    method='chars': character count / 4 (OpenAI's rough rule of thumb)
    """
    if not text:
        return 0
    if method == "words":
        return len(text.split())
    elif method == "chars":
        return max(1, len(text) // 4)
    return len(text.split())


def split_to_token_limit(text: str, max_tokens: int, method: str = "words") -> list[str]:
    """Split text into segments that each fit within max_tokens."""
    words = text.split()
    chunks = []
    current = []
    current_count = 0
    for word in words:
        wt = estimate_tokens(word, method)
        if current_count + wt > max_tokens and current:
            chunks.append(" ".join(current))
            current = [word]
            current_count = wt
        else:
            current.append(word)
            current_count += wt
    if current:
        chunks.append(" ".join(current))
    return chunks
