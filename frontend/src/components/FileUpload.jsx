import React, { useCallback, useState } from 'react';

const ACCEPTED = ['.dat', '.hea', '.edf', '.csv', '.txt'];

export default function FileUpload({ onFilesSelected, selectedFiles, onRemove }) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) onFilesSelected(files);
  }, [onFilesSelected]);

  const handleChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length) onFilesSelected(files);
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const hasFiles = selectedFiles && selectedFiles.length > 0;

  return (
    <div>
      <label
        htmlFor="ecg-file-input"
        className={`dropzone ${dragging ? 'active' : ''} ${hasFiles ? 'has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{ display: 'block', cursor: 'pointer' }}
      >
        <input
          id="ecg-file-input"
          type="file"
          multiple
          accept={ACCEPTED.join(',')}
          onChange={handleChange}
          style={{ display: 'none' }}
        />

        {!hasFiles ? (
          <>
            <span className="dropzone-icon">📂</span>
            <p className="dropzone-title">Drop ECG files here or click to browse</p>
            <p className="dropzone-sub">Supports single or multi-channel recordings</p>
            <div className="dropzone-formats">
              {ACCEPTED.map((f) => (
                <span key={f} className="format-badge">{f}</span>
              ))}
            </div>
          </>
        ) : (
          <>
            <span className="dropzone-icon">✅</span>
            <p className="dropzone-title">{selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} selected</p>
            <p className="dropzone-sub">Click to replace or drag new files</p>
          </>
        )}
      </label>

      {hasFiles && (
        <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {selectedFiles.map((file, i) => (
            <div key={i} className="file-selected">
              <span className="file-icon">
                {file.name.endsWith('.edf') ? '🧠' : file.name.endsWith('.dat') ? '📊' : '📄'}
              </span>
              <div className="file-info">
                <p className="file-name">{file.name}</p>
                <p className="file-size">{formatSize(file.size)}</p>
              </div>
              <button
                className="file-remove-btn"
                onClick={(e) => { e.preventDefault(); onRemove(i); }}
                title="Remove file"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
