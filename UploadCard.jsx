import { useRef, useState } from 'react';

function UploadCard({ selectedFile, previewUrl, onFileSelect, onAnalyze, isLoading, error }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const [file] = event.dataTransfer.files;
    if (file) onFileSelect(file);
  };

  return (
    <section className="upload-card">
      <div className="section-heading">
        <p className="eyebrow">Panoramic X-ray input</p>
        <h2>Upload an image for AI-assisted segmentation</h2>
      </div>

      <div
        className={`dropzone ${isDragging ? 'is-dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <img className="preview-image" src={previewUrl} alt="Selected dental X-ray preview" />
        ) : (
          <div className="dropzone-empty">
            <div className="upload-icon">+</div>
            <h3>Drop a panoramic dental X-ray here</h3>
            <p>JPEG, PNG, WEBP, BMP, and TIFF images are supported.</p>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        className="file-input"
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
        onChange={(event) => {
          const [file] = event.target.files;
          if (file) onFileSelect(file);
        }}
      />

      <div className="upload-actions">
        <button className="secondary-button" type="button" onClick={() => inputRef.current?.click()}>
          Choose file
        </button>
        <button className="primary-button" type="button" onClick={onAnalyze} disabled={!selectedFile || isLoading}>
          {isLoading ? <span className="spinner" /> : null}
          {isLoading ? 'Analyzing...' : 'Analyze X-ray'}
        </button>
      </div>

      {selectedFile ? (
        <div className="file-meta">
          <span>{selectedFile.name}</span>
          <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
        </div>
      ) : null}

      {error ? <div className="error-message">{error}</div> : null}
    </section>
  );
}

export default UploadCard;
