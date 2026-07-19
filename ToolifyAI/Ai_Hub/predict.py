import torch
import torch.nn as nn
import os
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = "Ai_Hub/ml_models/IntentModel.pt"
VECTORIZER_PATH = "Ai_Hub/ml_models/vectorizer.pkl"
LABELS_PATH = "Ai_Hub/ml_models/labels.pkl"

model = None
vectorizer = None
LABELS = None  # loaded from labels.pkl now instead of hardcoded --
                # this guarantees the order always matches what
                # LabelEncoder actually used during training.

# ---------------------------------------------------------------------------
# Load model + vectorizer + labels together. All three must be present
# and must all come from the SAME training run, or predictions will be
# meaningless (right numbers, wrong words, or a crash).
# ---------------------------------------------------------------------------
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(LABELS_PATH):
    try:
        # 1. Load the label order LabelEncoder actually produced.
        LABELS = list(joblib.load(LABELS_PATH))

        # 2. Load the vectorizer -- tells us the real input size.
        vectorizer = joblib.load(VECTORIZER_PATH)
        input_size = len(vectorizer.get_feature_names_out())

        # 3. Load the model weights.
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        hidden_size = state_dict["0.weight"].shape[0]
        num_classes = len(LABELS)

        # 4. Recreate the exact same architecture used during training.
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # layer 0
            nn.ReLU(),                           # layer 1
            nn.Linear(hidden_size, num_classes)  # layer 2
        )
        model.load_state_dict(state_dict)
        model.eval()

        print(f"✅ Model loaded. Labels in order: {LABELS}")

    except Exception as e:
        print(f"❌ Could not load model, vectorizer, or labels: {e}")
        model = None
        vectorizer = None
        LABELS = None
else:
    print("⚠️ Warning: model, vectorizer, or labels file not found, using keyword fallback.")


# Fallback label names -- used only if labels.pkl couldn't be loaded.
FALLBACK_LABELS = [
    "chatbot", "image_generator", "image_prompt", "code_writer",
    "resume_builder", "essay_writer", "grammar_checker", "qr_generator"
]


def keyword_fallback(text: str) -> str:
    """Simple rule-based backup used when the ML model isn't available."""
    text = text.lower()
    if "image" in text and "prompt" in text:
        return "image_prompt"
    elif "image" in text or "picture" in text or "photo" in text or "draw" in text:
        return "image_generator"
    elif "code" in text or "python" in text or "program" in text:
        return "code_writer"
    elif "resume" in text or "cv" in text:
        return "resume_builder"
    elif "essay" in text:
        return "essay_writer"
    elif "grammar" in text or "spelling" in text:
        return "grammar_checker"
    elif "qr" in text:
        return "qr_generator"
    return "chatbot"


def predict_intent(user_input: str) -> str:
    if model is None or vectorizer is None or LABELS is None:
        return keyword_fallback(user_input)

    # Turn the real text into the SAME kind of vector the model was
    # trained on, using the exact vectorizer fit during training.
    vector = vectorizer.transform([user_input]).toarray()
    x = torch.tensor(vector, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output, 1)

    return LABELS[predicted.item()]


def route_to_module(user_input: str) -> str:
    label = predict_intent(user_input)
    routes = {
        "chatbot": "/tools/chat/",
        "image_generator": "/tools/image/",
        "image_prompt": "/tools/prompt/",
        "code_writer": "/tools/code/",
        "resume_builder": "/tools/resume/",
        "essay_writer": "/tools/essay/",
        "grammar_checker": "/tools/grammar/",
        "qr_generator": "/tools/qr/",
    }
    return routes.get(label, "/tools/chat/")