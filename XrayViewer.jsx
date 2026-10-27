import { useState } from 'react';
import { mediaUrl } from '../api';

const FINDING_COLORS = {
  Crown: '#ffb547',
  Filling: '#2ecc71',
  Implant: '#ff7858',
  'Root Canal Treatment': '#aa70ff',
  'Root Piece': '#445cff',
  'impacted tooth': '#ff5cbe',
};

function toothColor(tooth) {
  if (tooth.sequence_level === 'serious_sequence_conflict') return '#ff4444';
  if (tooth.sequence_level === 'mild_sequence_warning') return '#ffd22e';
  if (tooth.anatomy_status === 'suspicious') return '#ff8a22';
  return '#2ecc71';
}

function toothLabel(tooth) {
  return String(tooth.tooth_number);
}

function XrayViewer({ result, viewerOptions, onViewerOptionChange }) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const imageSrc = result?.original_image;
  const teeth = result?.teeth || [];
  const findings = result?.findings || [];
  const canDrawOverlay = imageSize.width > 0 && imageSize.height > 0;

  return (
    <section className="viewer-panel">
      <div className="panel-topline">
        <div>
          <p className="eyebrow">Annotated output</p>
          <h2>X-ray viewer</h2>
        </div>
        <div className="viewer-toggles" aria-label="Viewer options">
          <label>
            <input
              type="checkbox"
              checked={viewerOptions.showToothNumbers}
              onChange={(event) => onViewerOptionChange('showToothNumbers', event.target.checked)}
            />
            Tooth numbers
          </label>
          <label>
            <input
              type="checkbox"
              checked={viewerOptions.showFindings}
              onChange={(event) => onViewerOptionChange('showFindings', event.target.checked)}
            />
            Findings
          </label>
          <label>
            <input
              type="checkbox"
              checked={viewerOptions.showMasks}
              onChange={(event) => onViewerOptionChange('showMasks', event.target.checked)}
            />
            Masks
          </label>
        </div>
      </div>

      <div className="xray-frame">
        {imageSrc ? (
          <div className="image-stage">
            <img
              src={mediaUrl(imageSrc)}
              alt="Dental X-ray analysis"
              onLoad={(event) => {
                setImageSize({
                  width: event.currentTarget.naturalWidth,
                  height: event.currentTarget.naturalHeight,
                });
              }}
            />
            {canDrawOverlay ? (
              <svg
                className="xray-overlay"
                viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {viewerOptions.showMasks
                  ? teeth.map((tooth, index) => (
                      <Polygon
                        key={`tooth-mask-${tooth.tooth_number}-${index}`}
                        points={tooth.polygon}
                        color={toothColor(tooth)}
                        opacity="0.18"
                      />
                    ))
                  : null}
                {viewerOptions.showMasks || viewerOptions.showFindings
                  ? findings.map((finding, index) => (
                      <Polygon
                        key={`finding-mask-${finding.finding_name}-${index}`}
                        points={finding.polygon}
                        color={FINDING_COLORS[finding.finding_name] || '#40dce6'}
                        opacity="0.44"
                      />
                    ))
                  : null}
                {viewerOptions.showToothNumbers
                  ? teeth.map((tooth, index) => (
                      <SvgLabel
                        key={`tooth-label-${tooth.tooth_number}-${index}`}
                        text={toothLabel(tooth)}
                        bbox={tooth.bbox}
                        color={toothColor(tooth)}
                      />
                    ))
                  : null}
                {viewerOptions.showFindings
                  ? findings.map((finding, index) => (
                      <SvgLabel
                        key={`finding-label-${finding.finding_name}-${index}`}
                        text={`${finding.finding_name} ${Math.round((finding.confidence || 0) * 100)}%`}
                        bbox={finding.bbox}
                        color={FINDING_COLORS[finding.finding_name] || '#40dce6'}
                        yOffset={28}
                      />
                    ))
                  : null}
              </svg>
            ) : null}
          </div>
        ) : (
          <div className="viewer-empty">Upload and analyze an X-ray to view segmentation results.</div>
        )}
      </div>
    </section>
  );
}

function Polygon({ points, color, opacity }) {
  if (!points?.length) return null;
  const pointString = points.map(([x, y]) => `${x},${y}`).join(' ');
  return <polygon points={pointString} fill={color} fillOpacity={opacity} stroke={color} strokeWidth="2" />;
}

function SvgLabel({ text, bbox, color, yOffset = 0 }) {
  if (!bbox) return null;
  const x1 = Array.isArray(bbox) ? bbox[0] : bbox.x1;
  const y1 = Array.isArray(bbox) ? bbox[1] : bbox.y1;
  const x = Number(x1) || 8;
  const y = Math.max(24, (Number(y1) || 8) + yOffset);
  const width = Math.max(34, text.length * 8 + 18);

  return (
    <g>
      <rect x={x} y={y - 22} width={width} height="26" rx="5" fill="#0c1724" stroke={color} strokeWidth="1.5" />
      <text x={x + 8} y={y - 5} fill="#ffffff" fontSize="14" fontWeight="800">
        {text}
      </text>
    </g>
  );
}

export default XrayViewer;
