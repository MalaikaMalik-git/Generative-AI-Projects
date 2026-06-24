// src/components/CitationChips.tsx
import type { Citation } from "../types/api";

interface Props {
  citations: Citation[];
}

export function CitationChips({ citations }: Props) {
  if (citations.length === 0) return null;

  return (
    <div className="citations">
      <span className="citations-label">Sources</span>
      <div className="citations-list">
        {citations.map((c) => (
          <a
            key={c.index}
            href={c.url}
            target="_blank"
            rel="noopener noreferrer"
            className="citation-chip"
            title={c.url}
          >
            <span className="citation-num">{c.index}</span>
            <span className="citation-domain">{c.domain}</span>
          </a>
        ))}
      </div>
    </div>
  );
}
