DentAI Vision

DentAI Vision is a full-stack local web application for preliminary panoramic dental X-ray analysis. A user can upload an X-ray image, run two YOLOv11 models, and view tooth numbering, dental finding segmentation, matched tooth-level findings, and an annotated image.

Medical Disclaimer

This AI result is for educational and preliminary analysis purposes only. It does not replace professional dental diagnosis. The application is not a clinical diagnosis tool and should not be used as the sole basis for treatment decisions.

Folder Structure

dental-ai-assistant/
├── backend/
│   ├── main.py
│   ├── model_service.py
│   ├── matching.py
│   ├── visualization.py
│   ├── requirements.txt
│   ├── models/
│   │   ├── tooth_numbering.pt
│   │   ├── tooth_numbering.yaml
│   │   ├── dental_segmentation.pt
│   │   └── dental_segmentation.yaml
│   └── outputs/
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── components/
│       └── styles/
└── README.md


Model Files Download

Due to GitHub file size limitations, the trained YOLOv11 model weights (.pt files) are stored externally.

Please download the required model files from the Google Drive link below and place them inside the backend/models/ directory before running the backend server:

📁 Download YOLO Model Files (Google Drive)

Required files in backend/models/:

tooth_numbering.pt

tooth_numbering.yaml

dental_segmentation.pt

dental_segmentation.yaml

The backend loads both YOLO models once during startup:

tooth_model = YOLO("models/tooth_numbering.pt")
finding_model = YOLO("models/dental_segmentation.pt")


If a model is missing or cannot be loaded, /health and /analyze return a clear error instead of a traceback.

Backend Setup

cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload


The backend runs at http://localhost:8000.

Frontend Setup

cd frontend
npm install
npm run dev


The frontend runs at http://localhost:5173.

How It Works

Upload a panoramic dental X-ray from the web interface.

The FastAPI backend saves the image to backend/outputs/.

The tooth numbering YOLO model runs with imgsz=1024 and conf=0.25.

The dental finding segmentation model runs with imgsz=768 and conf=0.25.

The backend extracts class IDs, confidence scores, bounding boxes, polygons, and masks.

Dental finding masks are matched to tooth masks using overlap ratio.

The backend saves an annotated image and returns JSON to the frontend.

API

POST /analyze

Accepts a multipart image upload named file.

Example response:

{
  "success": true,
  "original_image": "/outputs/example_original.png",
  "annotated_image": "/outputs/example_annotated.jpg",
  "teeth": [
    {
      "tooth_number": 26,
      "confidence": 0.94,
      "bbox": [120, 80, 190, 210],
      "polygon": [[120, 80], [190, 80]],
      "class_id": 25
    }
  ],
  "findings": [
    {
      "finding_name": "Filling",
      "confidence": 0.82,
      "bbox": [130, 100, 170, 150],
      "polygon": [[130, 100], [170, 100]],
      "class_id": 1
    }
  ],
  "matched_results": [
    {
      "tooth_number": 26,
      "finding": "Filling",
      "confidence": 0.82,
      "overlap_score": 0.34,
      "explanation": "A filling-like restoration was detected on tooth 26."
    }
  ],
  "matches": [
    {
      "tooth_number": 26,
      "finding": "Filling",
      "confidence": 0.82,
      "overlap_score": 0.34,
      "explanation": "A filling-like restoration was detected on tooth 26."
    }
  ],
  "ai_explanation": null,
  "disclaimer": "This AI result is for educational and preliminary analysis purposes only. It does not replace professional dental diagnosis."
}


Class Mappings

Tooth numbering model:

class_id 0 = Tooth 1

class_id 1 = Tooth 2

...

class_id 31 = Tooth 32

Dental finding segmentation model:

0: Crown

1: Filling

2: Implant

3: Root Canal Treatment

4: Root Piece

5: impacted tooth

POST /chat

Accepts a JSON question and optional structured analysis context from the current X-ray result. The chatbot does not receive image data.

Example request:

{
  "question": "What does filling mean?",
  "analysis_context": {
    "matches": [
      {
        "tooth_number": 26,
        "finding": "Filling",
        "confidence": 0.91
      }
    ]
  },
  "history": []
}


Example response:

{
  "success": true,
  "answer": "A filling-like AI finding around tooth 26 commonly refers to an area that appears previously restored. This is AI-generated information, may be inaccurate, is not a diagnosis, and should be reviewed by a licensed dental professional.",
  "disclaimer": "This AI result is for educational and preliminary analysis purposes only. It does not replace professional dental diagnosis."
}


Local Chatbot

The backend can answer follow-up questions using a local Ollama model. The LLM does not receive or analyze image data directly; it only receives structured YOLO and mask-overlap matching results plus the user's text question.

Default environment configuration:

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b
ENABLE_LLM_EXPLANATION=true


Chatbot answers are short, cautious, and educational. They must state that results are AI-generated, not a diagnosis, may include false positives or false negatives, and require review by a licensed dental professional. If Ollama is unavailable or disabled, /analyze still returns the computer vision results, and /chat returns a safe fallback answer.

Known Limitations

This is not a diagnostic system.

Matching depends on mask overlap and may be imperfect when segmentations overlap poorly.

The current viewer receives a single annotated image, so frontend toggles switch between original and annotated views rather than redrawing separate layers.

Tooth numbering follows the provided 1-32 class rule. It does not convert to FDI numbering.

Model performance depends entirely on the supplied YOLO model weights and training data.

Very large image uploads can increase inference time and memory usage.
