// src/components/MessageSkeleton.tsx
// Shown while waiting for the first streaming token.

export function MessageSkeleton() {
  return (
    <div className="skeleton-wrap">
      <div className="skeleton-line skeleton-line--long" />
      <div className="skeleton-line skeleton-line--med" />
      <div className="skeleton-line skeleton-line--short" />
    </div>
  );
}
