import { useEffect, useState } from 'react';
import { analyzeXray } from './api';
import Disclaimer from './components/Disclaimer.jsx';
import ResultDashboard from './components/ResultDashboard.jsx';
import UploadCard from './components/UploadCard.jsx';

const DEFAULT_DISCLAIMER =
  'This system is for educational and preliminary analysis only. It is not a medical diagnosis tool.';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [viewerOptions, setViewerOptions] = useState({
    showToothNumbers: true,
    showFindings: true,
    showMasks: true,
  });

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileSelect = (file) => {
    setError('');
    setResult(null);
    setSelectedFile(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError('');

    try {
      const payload = await analyzeXray(selectedFile);
      setResult(payload);
    } catch (err) {
      setResult(null);
      setError(err.message || 'Something went wrong during analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleViewerOptionChange = (key, value) => {
    setViewerOptions((current) => ({ ...current, [key]: value }));
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Dental X-ray AI assistant</p>
          <h1>DentAI Vision</h1>
          <p>AI-assisted panoramic dental X-ray analysis</p>
        </div>
        <div className="status-badge">Preliminary analysis</div>
      </header>

      <UploadCard
        selectedFile={selectedFile}
        previewUrl={previewUrl}
        onFileSelect={handleFileSelect}
        onAnalyze={handleAnalyze}
        isLoading={isLoading}
        error={error}
      />

      <ResultDashboard
        result={result}
        viewerOptions={viewerOptions}
        onViewerOptionChange={handleViewerOptionChange}
      />

      <Disclaimer text={result?.disclaimer || DEFAULT_DISCLAIMER} />
    </div>
  );
}

export default App;
