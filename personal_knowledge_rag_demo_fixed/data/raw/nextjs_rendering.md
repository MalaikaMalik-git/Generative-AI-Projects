# Next.js Rendering Notes

Next.js supports multiple rendering strategies. Static Site Generation, or SSG, produces HTML at build time. This is useful when content changes infrequently and you want fast page loads.

Server-Side Rendering, or SSR, generates HTML on each request. This is useful when content is dynamic or user-specific.

The practical difference is timing. SSG does work before users visit the page, while SSR does work during the request lifecycle.

SSG is usually faster and cheaper because content is prebuilt and served from a CDN. SSR is more flexible because it can generate fresh data for each request.

Sometimes the explanation of performance and flexibility appears in different parts of documentation. If chunking separates them, retrieval may miss the trade-off between SSG and SSR.

Another important factor is caching. SSG benefits heavily from caching, while SSR may require additional caching strategies to perform well.

Developers often confuse SSR with client-side rendering, but they are different. SSR happens on the server, while client-side rendering happens in the browser.

This distinction is sometimes explained separately, which can lead to confusion if chunks are split incorrectly.

Choosing between SSG and SSR depends on freshness requirements, personalization needs, SEO, and cost considerations.

In summary, the difference between SSG and SSR is not just technical but also architectural, and understanding their trade-offs requires connecting multiple ideas together.