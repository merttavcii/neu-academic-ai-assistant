import DentalChatbot from './DentalChatbot.jsx';
import FindingCard from './FindingCard.jsx';
import ToothSummary from './ToothSummary.jsx';
import XrayViewer from './XrayViewer.jsx';

function ResultDashboard({ result, viewerOptions, onViewerOptionChange }) {
  const matches = result?.matched_results || result?.matches || [];

  return (
    <main className="dashboard-grid">
      <XrayViewer result={result} viewerOptions={viewerOptions} onViewerOptionChange={onViewerOptionChange} />

      <aside className="results-sidebar">
        <ToothSummary result={result} />
        {result ? <DentalChatbot result={result} /> : null}

        <section className="findings-panel">
          <div className="section-heading compact">
            <p className="eyebrow">Computer vision detection results</p>
            <h2>Tooth-level matches</h2>
          </div>

          {matches.length ? (
            <div className="finding-list scrollable-finding-list">
              {matches.map((match, index) => (
                <FindingCard key={`${match.finding}-${match.tooth_number ?? 'unknown'}-${index}`} match={match} />
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <h3>No dental findings were detected by the AI model.</h3>
              <p>Please remember this does not guarantee the absence of disease.</p>
            </div>
          )}
        </section>
      </aside>
    </main>
  );
}

export default ResultDashboard;
