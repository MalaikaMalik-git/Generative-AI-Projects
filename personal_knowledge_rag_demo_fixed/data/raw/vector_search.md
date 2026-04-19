# Vector Search and Hybrid Retrieval

Dense retrieval uses embeddings. A query is converted into a vector and compared against document chunk vectors using similarity.

This approach is strong for semantic understanding. For example, "server logs" and "observability data" may match even if the words differ.

However, dense retrieval has limitations. It may miss exact keywords, product names, or codes. This is where keyword-based retrieval becomes useful.

Keyword retrieval focuses on exact matches. It works well when queries include specific identifiers, such as error codes or version numbers.

Hybrid retrieval combines both approaches. It gathers results from semantic and keyword search, then merges or reranks them.

Sometimes explanations of dense retrieval and keyword retrieval are separated, even though they complement each other. If chunking splits them, the relationship may not be clear.

Hybrid retrieval often performs better because it captures both semantic meaning and exact matches.

For example, a dense retriever may understand general meaning, while a keyword retriever captures exact signals. Combining both improves coverage.

This combination becomes especially important in technical documents, where both semantic understanding and exact matching are required.

In summary, hybrid retrieval addresses the weaknesses of individual methods and is often preferred in production systems.