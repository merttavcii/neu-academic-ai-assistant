function getMostCommonFinding(matches) {
  if (!matches?.length) return 'None';
  const counts = matches.reduce((acc, item) => {
    acc[item.finding] = (acc[item.finding] || 0) + 1;
    return acc;
  }, {});
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'None';
}

function ToothSummary({ result }) {
  const teeth = result?.teeth || [];
  const findings = result?.findings || [];
  const matches = result?.matched_results || [];
  const anatomySummary = result?.anatomy_summary || {};
  const lowConfidenceCount = findings.filter((finding) => finding.confidence < 0.5).length;

  return (
    <section className="summary-panel">
      <div className="section-heading compact">
        <p className="eyebrow">Analysis summary</p>
        <h2>Detection overview</h2>
      </div>
      <div className="metric-grid">
        <div className="metric">
          <span>Total teeth</span>
          <strong>{teeth.length}</strong>
        </div>
        <div className="metric">
          <span>Total findings</span>
          <strong>{findings.length}</strong>
        </div>
        <div className="metric wide">
          <span>Most common finding</span>
          <strong>{getMostCommonFinding(matches)}</strong>
        </div>
      </div>
      {lowConfidenceCount > 0 ? (
        <div className="warning-message">
          {lowConfidenceCount} finding{lowConfidenceCount === 1 ? '' : 's'} had confidence below 50%.
        </div>
      ) : null}
      {result?.warnings?.length ? (
        <div className="warning-list">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export default ToothSummary;
