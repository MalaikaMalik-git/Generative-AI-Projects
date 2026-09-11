"""
Vision helper used by the /vision endpoint.

Reuses the same OpenAI client and provider as llm.py — one AI provider
for both text answers and image analysis, per the project plan. GPT-4o
mini accepts a base64 data URL directly in `image_url.url`, so no
manual base64 splitting is needed (unlike some other providers).

Flow as of this revision:
1. Look at the image and try to read a written/displayed question out of
   it (a photo of text, a screenshot, a whiteboard, a document, etc.).
2. If a question was found, answer it the same grounded way /chat does —
   retrieve context from the ChromaDB company-knowledge collection and
   generate an answer from ONLY that context, with real sources.
3. If no question is present in the image (it's just a photo of an
   object/scene), fall back to a plain visual description/answer, same
   as before — this path is not grounded in the knowledge base since
   there's no question to look up.
"""

import re
from typing import Optional

from llm import get_client, generate_answer, LLMNotConfiguredError  # noqa: F401 (re-exported)
from rag import retrieve_context, RagNotReadyError

MODEL = "gpt-4o-mini"
MAX_TOKENS = 400

DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z]+);base64,(.+)$", re.DOTALL)
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

DEFAULT_QUESTION = "What do you see in this image?"
NO_QUESTION_MARKER = "NO_QUESTION_FOUND"

DESCRIBE_SYSTEM_PROMPT = """You are the voice of a 3D avatar assistant. A visitor just \
showed you something through their camera. Describe or answer their question \
about what you see, naturally and conversationally.

Rules:
- Keep it short: 2-4 sentences. Your answer will be read aloud by \
text-to-speech, so avoid bullet points or lists.
- If the image is blurry, dark, or unclear, say so honestly rather than \
guessing at details you can't actually make out.
- Speak like a helpful assistant reacting to what's in front of you, not a \
technical image-analysis report."""

EXTRACT_SYSTEM_PROMPT = f"""You read images that may contain a written or \
displayed question — a photo of text on paper, a screenshot, a whiteboard, \
a sign, a document, etc.

Rules:
- If the image contains a clear question or query written or displayed in \
it, reply with ONLY that question, copied as written, and nothing else.
- Do not answer the question yourself here — extraction only.
- If the image does NOT contain any readable question (e.g. it's a photo \
of an object, person, or general scene with no question written in it), \
reply with exactly: {NO_QUESTION_MARKER}"""


class InvalidImageError(ValueError):
    """Raised when the image payload isn't a well-formed, supported data URL."""


def validate_data_url(data_url: str) -> str:
    """Confirms 'data:image/jpeg;base64,<data>' shape and a supported MIME
    type, then returns the original string unchanged — OpenAI's API takes
    the full data URL directly."""
    if not isinstance(data_url, str):
        raise InvalidImageError("Image must be a base64 data URL string.")

    stripped = data_url.strip()
    match = DATA_URL_RE.match(stripped)
    if not match:
        raise InvalidImageError(
            "Image must be a data URL like 'data:image/jpeg;base64,...' "
            "(this is exactly what canvas.toDataURL() produces in the browser)."
        )

    media_type = match.group(1)
    if media_type not in SUPPORTED_TYPES:
        raise InvalidImageError(
            f"Unsupported image type '{media_type}'. Use JPEG, PNG, WEBP, or GIF."
        )

    return stripped


def _extract_question(client, validated_url: str) -> Optional[str]:
    """Asks the vision model to read a question out of the image, verbatim.
    Returns None if the image doesn't contain one."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=120,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": validated_url}},
                ],
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text or NO_QUESTION_MARKER in text.upper():
        return None
    return text


def _describe_image(client, validated_url: str, question: Optional[str]) -> str:
    """Plain visual description/answer — used when there's no question
    written in the image to look up."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": validated_url}},
                    {"type": "text", "text": (question or DEFAULT_QUESTION).strip()},
                ],
            },
        ],
    )
    return response.choices[0].message.content.strip()


def generate_vision_answer(
    image_data_url: str, question: Optional[str] = None
) -> tuple[str, list[str]]:
    """Returns (answer, sources). sources is non-empty only when the answer
    came from the ChromaDB-grounded path."""
    validated_url = validate_data_url(image_data_url)
    client = get_client()

    extracted_question = _extract_question(client, validated_url)

    if extracted_question:
        try:
            chunks = retrieve_context(extracted_question)
        except RagNotReadyError:
            chunks = []

        if chunks:
            answer = generate_answer(extracted_question, chunks)
            sources = sorted({c["source"] for c in chunks})
            return answer, sources

        return (
            f'I found this question in the image — "{extracted_question}" — but '
            "I don't have information about that yet. Try asking about Agentix "
            "System's services, agents, or how to get in touch.",
            [],
        )

    # No question detected in the image — fall back to plain visual
    # description, optionally guided by whatever the visitor typed.
    answer = _describe_image(client, validated_url, question)
    return answer, []
