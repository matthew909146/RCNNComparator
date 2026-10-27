"""Flask API for the RCNN Family Comparator."""
import hashlib
import uuid
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sqlalchemy import (
    Column, Float, ForeignKey, Integer, String, Text, create_engine, DateTime,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import (
    DATABASE_URL, MAX_CONTENT_LENGTH, DEVICE,
    FRCNN_INPUT_SIZE, VGG_INPUT_SIZE, RCNN_INPUT_SIZE,
)
from .preprocessing import load_image, to_variable_size_tensor, to_tensor_normalised
from .models.registry import get_detectors

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

Base = declarative_base()
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)


class SessionRow(Base):
    __tablename__ = "sessions"
    session_id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_hash = Column(String(64))
    image_filename = Column(String(255))
    results = relationship("ModelResultRow", back_populates="session",
                           cascade="all, delete-orphan")


class ModelResultRow(Base):
    __tablename__ = "model_results"
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id"))
    model_name = Column(String(64))
    inference_ms = Column(Float)
    detection_count = Column(Integer)
    proposal_count = Column(Integer)
    payload = Column(Text)
    session = relationship("SessionRow", back_populates="results")


Base.metadata.create_all(engine)

print("[init] loading detectors …")
DETECTORS = get_detectors()
print("[init] ready")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE,
                    "models": [d.name for d in DETECTORS]})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400
    f = request.files["image"]
    raw = f.read()
    if not raw:
        return jsonify({"error": "empty upload"}), 400

    image_hash = hashlib.sha256(raw).hexdigest()
    session_id = str(uuid.uuid4())

    pil = load_image(raw)

    rcnn_tensor, _, _ = to_tensor_normalised(pil, RCNN_INPUT_SIZE)
    frcnn_tensor, _, _ = to_tensor_normalised(pil, VGG_INPUT_SIZE)
    faster_tensor, _, _ = to_variable_size_tensor(pil, FRCNN_INPUT_SIZE)

    inputs = [rcnn_tensor, frcnn_tensor, faster_tensor]

    responses = []
    db = SessionLocal()
    try:
        row = SessionRow(session_id=session_id, image_hash=image_hash,
                         image_filename=f.filename or "upload")
        db.add(row)

        for det, inp in zip(DETECTORS, inputs):
            try:
                result = det.detect(inp, original_pil=pil)
            except Exception as exc:
                from .models.base import DetectionResult
                result = DetectionResult(model_name=det.name, inference_ms=0.0,
                                         error=f"{type(exc).__name__}: {exc}")
            payload = result.to_dict()
            responses.append(payload)
            db.add(ModelResultRow(
                session_id=session_id, model_name=det.name,
                inference_ms=payload["inference_ms"],
                detection_count=payload["detection_count"],
                proposal_count=payload["proposal_count"],
                payload=str(payload["detections"]),
            ))
        db.commit()
    finally:
        db.close()

    return jsonify({
        "session_id": session_id,
        "image_hash": image_hash,
        "results": responses,
    })


@app.route("/api/results/<session_id>")
def results(session_id):
    db = SessionLocal()
    try:
        row = db.query(SessionRow).filter_by(session_id=session_id).first()
        if not row:
            return jsonify({"error": "session not found"}), 404
        return jsonify({
            "session_id": row.session_id,
            "timestamp": row.timestamp.isoformat(),
            "image_filename": row.image_filename,
            "results": [
                {"model_name": r.model_name,
                 "inference_ms": r.inference_ms,
                 "detection_count": r.detection_count,
                 "proposal_count": r.proposal_count}
                for r in row.results
            ],
        })
    finally:
        db.close()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)