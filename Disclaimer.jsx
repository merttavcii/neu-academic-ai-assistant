function Disclaimer({ text }) {
  return (
    <section className="disclaimer" aria-label="Medical disclaimer">
      <strong>Important medical disclaimer</strong>
      <span>
        {text ||
          'This system is for educational and preliminary analysis only. It is not a medical diagnosis tool.'}
      </span>
    </section>
  );
}

export default Disclaimer;
