# Prompt Engineering Notes

Prompt engineering is the practice of shaping instructions and context so a model produces more reliable outputs.

A strong prompt usually clarifies the task, the expected format, the boundaries of the answer, and how uncertainty should be handled.

System prompts are especially important because they define the assistant's behavior. Good system prompts reduce ambiguity.

For example, a system prompt may instruct the model to answer only from provided context. This reduces hallucination and improves reliability.

Prompt failures often happen because the request is underspecified. If the model does not know the format or audience, the output becomes inconsistent.

Sometimes the explanation of system prompts and failure modes appears separately. If chunking splits these ideas, retrieval may not connect cause and effect.

Another important concept is constraints. Constraints guide the model toward desired behavior, such as limiting output length or enforcing structure.

These constraints are often described in different sections, which can create boundary issues during chunking.

A common RAG pattern is: answer only from context, and say clearly if the answer is not found. This pattern reduces hallucination.

However, this idea is often discussed alongside evaluation and trust, which may appear elsewhere in the document.

In summary, prompt engineering is about clarity, structure, and constraints, even though these ideas may be distributed across multiple sections.