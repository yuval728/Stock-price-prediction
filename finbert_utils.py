from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device)
labels = ['positive', 'negative', 'neutral']

def estimate_sentiment(news):
    if not news:
        return 0, labels[-1]
    
    tokens = tokenizer(news, return_tensors='pt', padding=True).to(device)
    
    results = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])[ 'logits']
    
    results = torch.nn.functional.softmax(torch.sum(results, dim=0), dim=-1)
    probability = results[torch.argmax(results)]
    sentiment = labels[torch.argmax(results)]
    return probability, sentiment

if __name__ == '__main__':
    news = ['The stock market is doing well', 'The stock market is not doing well']
    for n in news:
        probability, sentiment = estimate_sentiment(n)
        print(f'News: {n}\nSentiment: {sentiment}\nProbability: {probability}\n')