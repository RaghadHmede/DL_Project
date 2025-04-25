from flask import Flask, request, jsonify
import torch
import pickle
from model import Encoder, Decoder, Seq2Seq  
from utils import clean_text  
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("fr_vocab.pkl", "rb") as f:
    fr_vocab = pickle.load(f)
with open("en_rev_vocab.pkl", "rb") as f:
    en_rev_vocab = pickle.load(f)

encoder = Encoder(len(fr_vocab), emb_dim=256, hidden_dim=512, n_layers=2).to(device)
decoder = Decoder(len(en_rev_vocab), emb_dim=256, hidden_dim=512, n_layers=2).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('translation_model.pth', map_location=torch.device('cpu')))
model.eval()

def translate(sentence):
    print(f"[Input sentence]: {sentence}")

    tokens = clean_text(sentence, language='fr')
    print(f"[Cleaned tokens]: {tokens}")

    token_ids = [fr_vocab.get(token, fr_vocab['<unk>']) for token in tokens]
    print(f"[Token IDs]: {token_ids}")

    src_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        encoder_outputs, _ = model.encoder.lstm(model.encoder.embedding(src_tensor))

    input_token = torch.tensor([fr_vocab.get('<start>', 1)], device=device)
    print(f"[Start token ID]: {input_token.item()}")

    output_sentence = []
    for step in range(50):
        with torch.no_grad():
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()
            word = en_rev_vocab.get(top1, '<unk>')

        print(f"[Step {step}] Predicted token ID: {top1} → '{word}'")

        if word == '<end>':
            break
        output_sentence.append(word)
        input_token = torch.tensor([top1], device=device)

    translated_sentence = " ".join(output_sentence).strip()
    print(f"[Translated sentence]: {translated_sentence}")

    return translated_sentence if translated_sentence else "Translation not found."

@app.route("/translate", methods=["POST"])
def handle_translate():
    data = request.get_json()
    french_sentence = data.get("sentence")
    translated = translate(french_sentence)
    return jsonify({"translation": translated})

if __name__ == "__main__":
    app.run(debug=True)
