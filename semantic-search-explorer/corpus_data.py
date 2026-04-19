DOCUMENTS = [
    {
        "source": "doc_01",
        "title": "Flutter Basics",
        "text": """
Flutter is a UI toolkit developed by Google for building cross-platform mobile applications.
It allows developers to use one codebase to build apps for Android and iOS.
Flutter uses the Dart programming language and includes widgets for responsive UI design.
Flutter has a rich set of pre-built material design widgets that speed up development.
Hot reload in Flutter lets developers see code changes instantly without restarting the app.
        """.strip()
    },
    {
        "source": "doc_02",
        "title": "React Overview",
        "text": """
React is a JavaScript library for building interactive user interfaces.
It is commonly used for web applications and supports reusable components.
React focuses on efficient rendering through a virtual DOM.
React hooks allow functional components to manage state and side effects.
The React ecosystem includes tools like Redux for state management and React Router for navigation.
        """.strip()
    },
    {
        "source": "doc_03",
        "title": "MongoDB",
        "text": """
MongoDB is a NoSQL database designed for scalability and flexibility.
It stores data in JSON-like documents and is widely used in modern web applications.
MongoDB works well in systems that need fast development and schema flexibility.
It supports powerful aggregation pipelines for complex data transformations.
MongoDB Atlas provides a fully managed cloud database service with built-in scaling.
        """.strip()
    },
    {
        "source": "doc_04",
        "title": "Embeddings",
        "text": """
Embeddings are numerical vector representations of text, images, or other data.
In natural language processing, embeddings capture semantic meaning.
Similar meanings tend to produce vectors that lie close together in vector space.
Word embeddings like Word2Vec were early breakthroughs in representing language numerically.
Modern sentence embeddings encode entire phrases into a single dense vector for retrieval tasks.
        """.strip()
    },
    {
        "source": "doc_05",
        "title": "Semantic Search",
        "text": """
Semantic search retrieves results based on meaning instead of exact keyword matching.
It is useful when the user asks a question using different words from the stored documents.
Semantic search is often powered by embeddings and vector similarity.
Unlike keyword search, semantic search can handle synonyms and paraphrased queries effectively.
It is widely used in search engines, question answering systems, and recommendation engines.
        """.strip()
    },
    {
        "source": "doc_06",
        "title": "Cosine Similarity",
        "text": """
Cosine similarity is a measure of how similar two vectors are based on the angle between them.
A smaller angle means the vectors point in a similar direction.
In semantic search, cosine similarity helps compare query embeddings with document embeddings.
Cosine similarity values range from minus one to one, where one means identical direction.
It is preferred over Euclidean distance for text similarity because it ignores vector magnitude.
        """.strip()
    },
    {
        "source": "doc_07",
        "title": "Chroma Vector Database",
        "text": """
Chroma is a vector database used for storing embeddings and running similarity search.
It allows developers to index text chunks and retrieve relevant items for a query.
Chroma is often used in retrieval augmented generation systems.
It supports persistent storage so embeddings survive application restarts.
Chroma provides a simple Python API for adding, querying, and deleting collections.
        """.strip()
    },
    {
        "source": "doc_08",
        "title": "Streamlit",
        "text": """
Streamlit is a Python framework for quickly building interactive web applications.
It is popular for machine learning demos, dashboards, and data applications.
With Streamlit, developers can build a user interface using only Python code.
Streamlit automatically reruns the script from top to bottom whenever a user interacts with a widget.
Session state in Streamlit allows data to persist across reruns within a single user session.
        """.strip()
    },
    {
        "source": "doc_09",
        "title": "RAG Systems",
        "text": """
Retrieval-Augmented Generation combines document retrieval with text generation.
A system first retrieves relevant chunks from a knowledge base, then uses them to generate an answer.
RAG improves factual grounding for language models.
It reduces hallucination by anchoring model responses to retrieved source documents.
RAG pipelines typically consist of an indexing phase and a query phase.
        """.strip()
    },
    {
        "source": "doc_10",
        "title": "Transformers",
        "text": """
Transformers are deep learning models that use self-attention mechanisms.
They are widely used in natural language processing for tasks like translation, search, and generation.
Modern embedding models are often based on transformer architectures.
The original transformer was introduced in the paper Attention Is All You Need in 2017.
BERT and GPT are two influential transformer models that shaped modern NLP research.
        """.strip()
    },
    {
        "source": "doc_11",
        "title": "JWT Authentication",
        "text": """
JWT stands for JSON Web Token.
It is used for authentication and secure information exchange between systems.
A server can issue a token after login, and the client can send it with future requests.
JWTs consist of three parts: a header, a payload, and a signature separated by dots.
The signature ensures that the token has not been tampered with during transmission.
        """.strip()
    },
    {
        "source": "doc_12",
        "title": "REST APIs",
        "text": """
REST APIs allow systems to communicate over HTTP.
They typically use endpoints for creating, reading, updating, and deleting resources.
REST is widely used in web and mobile backend development.
RESTful APIs are stateless, meaning each request must contain all the information the server needs.
Common HTTP methods used in REST include GET, POST, PUT, PATCH, and DELETE.
        """.strip()
    },
    {
        "source": "doc_13",
        "title": "Vector Databases",
        "text": """
Vector databases are purpose-built systems for storing and querying high-dimensional vectors.
They use approximate nearest neighbour algorithms to find similar vectors efficiently.
Popular vector databases include Chroma, Pinecone, Weaviate, and Qdrant.
Vector databases are essential infrastructure for AI-powered search and recommendation systems.
They can store metadata alongside embeddings to enable filtered retrieval.
        """.strip()
    },
    {
        "source": "doc_14",
        "title": "Sentence Transformers",
        "text": """
Sentence Transformers is a Python library for generating dense vector embeddings from text.
It builds on top of Hugging Face transformers and provides pre-trained models for semantic similarity.
The all-MiniLM-L6-v2 model is a popular lightweight model that encodes sentences into 384 dimensions.
Sentence Transformers models are trained using contrastive learning on large text pair datasets.
They can be fine-tuned on domain-specific data to improve retrieval performance.
        """.strip()
    },
    {
        "source": "doc_15",
        "title": "Natural Language Processing",
        "text": """
Natural language processing is the field of AI that enables computers to understand human language.
It powers applications like chatbots, machine translation, sentiment analysis, and search engines.
Key NLP tasks include tokenisation, named entity recognition, and text classification.
Deep learning has dramatically improved NLP performance over traditional rule-based approaches.
Large language models are now the dominant paradigm in NLP research and applications.
        """.strip()
    },
    {
        "source": "doc_16",
        "title": "Python for Data Science",
        "text": """
Python is the most widely used programming language for data science and machine learning.
Libraries like NumPy, Pandas, and Matplotlib provide tools for data manipulation and visualisation.
Scikit-learn offers a consistent API for classical machine learning algorithms.
Python's simplicity and extensive ecosystem make it the default choice for AI research and prototyping.
Jupyter notebooks allow interactive development and inline visualisation for data exploration.
        """.strip()
    },
    {
        "source": "doc_17",
        "title": "Docker and Containers",
        "text": """
Docker is a platform for packaging applications into lightweight, portable containers.
Containers include the application code, runtime, libraries, and configuration in one unit.
Docker ensures that an application runs consistently across different environments.
Docker Compose allows developers to define and run multi-container applications with a single file.
Container orchestration tools like Kubernetes manage containers at scale in production.
        """.strip()
    },
    {
        "source": "doc_18",
        "title": "Large Language Models",
        "text": """
Large language models are neural networks trained on massive text datasets to generate human-like text.
They use transformer architectures with billions of parameters to learn language patterns.
Examples include GPT-4, Claude, Gemini, and LLaMA.
LLMs can perform a wide range of tasks including summarisation, coding, translation, and reasoning.
Fine-tuning and prompt engineering are common techniques for adapting LLMs to specific use cases.
        """.strip()
    },
    {
        "source": "doc_19",
        "title": "PostgreSQL",
        "text": """
PostgreSQL is an open-source relational database system known for reliability and feature richness.
It supports advanced SQL queries, indexing, and full-text search out of the box.
PostgreSQL is ACID compliant, ensuring data integrity even during system failures.
The pgvector extension adds native vector similarity search support to PostgreSQL.
PostgreSQL is widely used in production systems across startups and enterprise applications.
        """.strip()
    },
    {
        "source": "doc_20",
        "title": "GraphQL",
        "text": """
GraphQL is a query language for APIs that allows clients to request exactly the data they need.
It was developed by Facebook as an alternative to REST for more flexible data fetching.
GraphQL uses a strongly typed schema to define the structure of available data.
A single GraphQL endpoint can replace many REST endpoints in complex applications.
Subscriptions in GraphQL enable real-time data updates over WebSocket connections.
        """.strip()
    },
]
