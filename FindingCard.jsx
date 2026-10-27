const FINDING_COLORS = {
  Crown: '#ffb547',
  Filling: '#2ecc71',
  Implant: '#ff7858',
  'Root Canal Treatment': '#aa70ff',
  'Root Piece': '#445cff',
  'impacted tooth': '#ff5cbe',
};

function FindingCard({ match }) {
  const confidence = Math.round((match.confidence || 0) * 100);
  const overlap = Math.round((match.overlap_score || 0) * 100);
  const toothLabel = match.tooth_number ? `Tooth ${match.tooth_number}` : 'Unknown tooth';
  const accentColor = FINDING_COLORS[match.finding] || '#40dce6';

  return (
    <article
      className="finding-card"
      style={{
        borderLeft: `6px solid ${accentColor}`,
      }}
    >
      <div className="finding-card-header">
        <span
          className="tooth-pill"
          style={{
            backgroundColor: `${accentColor}22`,
            borderColor: accentColor,
            color: accentColor,
          }}
        >
          {toothLabel}
        </span>

        <span className="confidence">{confidence}%</span>
      </div>

      <h3 style={{ color: accentColor }}>{match.finding}</h3>

      <dl>
        <div>
          <dt>Overlap score</dt>
          <dd>{overlap}%</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{confidence}%</dd>
        </div>
      </dl>

      <p>{match.explanation}</p>
    </article>
  );
}

export default FindingCard;