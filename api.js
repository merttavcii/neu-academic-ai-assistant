export const API_BASE_URL = 'http://127.0.0.1:8010';

export async function analyzeXray(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok || !payload?.success) {
    throw new Error(payload?.error || payload?.detail || 'The image could not be analyzed.');
  }

  return payload;
}

export async function askDentalChat({ question, analysisContext, history }) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      analysis_context: analysisContext,
      history,
    }),
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok || !payload?.success) {
    throw new Error(payload?.error || payload?.detail || 'The chatbot could not answer right now.');
  }

  return payload;
}

export function mediaUrl(path) {
  if (!path) return '';
  if (path.startsWith('http') || path.startsWith('data:')) return path;
  return `${API_BASE_URL}${path}`;
}
