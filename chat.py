import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load file data
with open('intents.json', 'r',encoding='utf8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
# Khai báo size
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

# Khai báo model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Senki"
# Get response from message
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    # Lấy output từ model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    # Lấy tag từ predicted
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    # Nếu prob > 0.75 thì lấy câu trả lời từ file intents.json
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Xin lỗi, tôi không hiểu những gì bạn đang nói..."

# Test
if __name__ == "__main__":
    print("Hãy chat với tôi ! (Gõ 'exit' để thoát)")
    while True:
        # sentence = "Xin chào"
        sentence = input("You: ")
        if sentence == "exit":
            break
        resp = get_response(sentence)
        print(resp)

