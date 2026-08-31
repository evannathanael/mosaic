export function SkeletonFeed() {
  return (
    <div className="feed" aria-busy="true" aria-label="Loading feed">
      <div className="slide">
        <div className="skeleton-card">
          <div className="sk-row" style={{ width: '30%' }} />
          <div className="sk-img" />
          <div className="sk-row" />
        </div>
      </div>
    </div>
  )
}
