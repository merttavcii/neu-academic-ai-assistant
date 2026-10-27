import { useMemo, useState } from 'react';
import { askDentalChat } from '../api';

const DEFAULT_MESSAGES = [
  {
    role: 'assistant',
    content:
      'Hello, I am your AI-powered analysis assistant. I can help answer any questions you may have about your X-ray results. Feel free to ask about the findings, tooth structures, or any detected conditions.',
  },
];

function DentalChatbot({ result }) {
  const [messages, setMessages] = useState(DEFAULT_MESSAGES);
  const [question, setQuestion] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState('');
  const analysisContext = useMemo(() => buildAnalysisContext(result), [result]);

  const handleSubmit = async (event) => {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isSending) return;

    const userMessage = { role: 'user', content: trimmedQuestion };
    const nextMessages = [...messages, userMessage];

    setMessages(nextMessages);
    setQuestion('');
    setError('');
    setIsSending(true);

    try {
      const payload = await askDentalChat({
        question: trimmedQuestion,
        analysisContext,
        history: messages.slice(-6),
      });

      setMessages([...nextMessages, { role: 'assistant', content: payload.answer }]);
    } catch (err) {
      setError(err.message || 'The chatbot could not answer right now.');
      setMessages([
        ...nextMessages,
        {
          role: 'assistant',
          content:
            'The chatbot is unavailable right now. AI-generated results can have false positives or false negatives and are not a diagnosis.',
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <section className="chatbot-panel" aria-label="Dental AI chatbot">
      <div className="section-heading compact">
        <p className="eyebrow">Chatbot</p>
        <h2>Ask about this analysis</h2>
      </div>

      <div className="chat-messages" aria-live="polite">
        {messages.map((message, index) => (
          <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
            <span>{message.role === 'user' ? 'You' : 'DentAI Chat'}</span>
            <p>{message.content}</p>
          </div>
        ))}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Example: What does the filling finding mean?"
          rows="3"
        />
        <button className="primary-button" type="submit" disabled={!question.trim() || isSending}>
          {isSending ? 'Answering...' : 'Ask'}
        </button>
      </form>

      {error ? <p className="chat-error">{error}</p> : null}
      <p className="chat-disclaimer">
        This chatbot is designed solely to assist and provide general information about your X-ray results. The
        information provided may not always be fully accurate and should not be considered a medical diagnosis. It
        cannot replace a professional evaluation or diagnosis made by a licensed doctor or dentist.
      </p>
    </section>
  );
}

function buildAnalysisContext(result) {
  if (!result) {
    return {
      matches: [],
      findings: [],
      warnings: [],
    };
  }

  return {
    matches: (result.matched_results || result.matches || []).map((item) => ({
      tooth_number: item.tooth_number ?? item.matched_tooth_number ?? null,
      finding: item.finding ?? item.finding_class ?? '',
      confidence: item.confidence,
      overlap_score: item.overlap_score,
    })),
    findings: (result.findings || []).map((item) => ({
      finding_name: item.finding_name ?? item.finding ?? '',
      confidence: item.confidence,
    })),
    warnings: result.warnings || [],
  };
}

export default DentalChatbot;
