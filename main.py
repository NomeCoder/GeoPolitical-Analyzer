from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# ---------- CONFIG ----------

# If you saved your model locally:
# MODEL_NAME = "geopolitics_model"
# Else use the HF model directly:
MODEL_NAME = "geopolitics_model"


LABELS = [
    "cooperation",
    "conflict",
    "neutral",
    "military escalation",
    "diplomacy"
]

# Load model once at startup (this can take a bit the first time)
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1  # CPU; use 0 if you have GPU
)


# ---------- ROUTES ----------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    scores = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            result = classifier(text, LABELS)
            # Top prediction
            prediction = result["labels"][0]
            # All labels + scores
            scores = list(zip(result["labels"],
                              [round(s, 3) for s in result["scores"]]))

    return render_template(
        "index.html",
        text=text,
        prediction=prediction,
        scores=scores,
        labels=LABELS,
    )


if __name__ == "__main__":
    # debug=True for dev, turn off in production
    app.run(host="0.0.0.0", port=5000, debug=True)
