# Testing and Observability for AI Systems

Production AI systems need more than correct code. They need observability. Observability means having enough telemetry to understand what the system is doing.

For a RAG system, useful telemetry includes query text, retrieval latency, generation latency, retrieved sources, and error counts.

Observability matters because AI failures are often subtle. The system may return a fluent answer that is actually incorrect.

If you only monitor uptime, you will miss relevance failures. This is why observability must include quality signals.

Sometimes the explanation of metrics and the explanation of failures appear separately. If chunking splits them, retrieval may not connect the importance of monitoring.

A good monitoring setup helps answer questions such as:
- Which queries fail most often?
- Did retrieval degrade after a change?
- Are certain documents never retrieved?

These questions are often discussed in different contexts, but they are closely related.

Another important concept is feedback loops. User feedback helps improve the system over time.

This feedback loop may be described separately from observability, even though they are connected.

In summary, observability turns a black-box AI system into something that can be inspected, debugged, and improved over time.