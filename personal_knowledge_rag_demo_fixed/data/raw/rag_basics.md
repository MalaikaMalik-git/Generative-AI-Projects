# RAG Basics

Retrieval-Augmented Generation (RAG) combines retrieval and generation. Instead of asking a language model to answer only from its internal parameters, a RAG system first retrieves relevant external knowledge and then uses that knowledge as context for answering.

A simple pipeline has five stages: document loading, chunking, embedding, retrieval, and answer generation. These steps are often described together, but sometimes explanations of chunking and retrieval are separated, which can affect how systems interpret them.

Chunking matters because embedding an entire long document often produces a representation that is too broad. Important details become diluted. Smaller chunks preserve local meaning and make retrieval sharper.

However, chunking introduces a problem. When splitting text, important ideas may lie at the boundary between chunks. This is where overlap becomes important.

Overlap is used so that information near a chunk boundary is not lost. Without overlap, one idea might be split across two chunks and neither chunk alone contains enough context to match the query well.

Overlap repeats a small portion of text from the previous chunk in the next chunk so boundary information remains recoverable. This repetition may seem redundant, but it significantly improves retrieval performance.

Very small chunks can lose context. Very large chunks can become noisy. This creates a trade-off that must be carefully managed.

Sometimes the explanation of "small chunks vs large chunks" appears separately from "overlap," even though they are closely related. If chunking breaks these connections, retrieval may not capture the full reasoning.

Recursive chunking helps preserve semantic structure by respecting paragraphs and sentences. Fixed chunking, on the other hand, may cut ideas arbitrarily.

This difference becomes more visible in documents with mixed structure, repeated ideas, and boundary-sensitive information.

In summary, chunking is not just a preprocessing step. It directly impacts retrieval quality, and strategies like overlap and recursive splitting help preserve meaning.