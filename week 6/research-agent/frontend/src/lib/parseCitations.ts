// src/lib/parseCitations.ts
//
// Splits the agent's markdown report into:
//   - body: everything before "## Sources"
//   - citations: the numbered URL list under "## Sources"

import type { Citation } from "../types/api";

export interface ParsedReport {
  body: string;
  citations: Citation[];
}

function extractDomain(url: string): string {
  try {
    const { hostname } = new URL(url);
    return hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

export function parseCitations(raw: string): ParsedReport {
  const sourcesIdx = raw.search(/^## Sources\b/m);

  if (sourcesIdx === -1) {
    return { body: raw.trim(), citations: [] };
  }

  const body = raw.slice(0, sourcesIdx).trim();
  const sourcesBlock = raw.slice(sourcesIdx);

  // Match lines like:  1. https://example.com/path
  const urlRegex = /^\d+\.\s+(https?:\/\/\S+)/gm;
  const citations: Citation[] = [];
  let match: RegExpExecArray | null;
  let index = 1;

  while ((match = urlRegex.exec(sourcesBlock)) !== null) {
    const url = match[1].replace(/[.,;)]+$/, ""); // strip trailing punctuation
    citations.push({ index, url, domain: extractDomain(url) });
    index++;
  }

  return { body, citations };
}
