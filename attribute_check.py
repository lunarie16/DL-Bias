import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
import json

logging.basicConfig(level=logging.INFO)  # OPTIONAL

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()


straightValue = 0


def ownAttributeCheck(sentence: str, word: str, topk: int):
    global straightValue
    text = f"[CLS] {sentence} [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, topk, sorted=True)
    foundSomething = False
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        if word == predicted_token:
            result = f"{sentence.replace('[MASK]', word):40s} at {str(i):5s} with {top_k_weights[i]:0.15f}"
            resultTuple = (i, '%.15f'%top_k_weights[i])
            foundSomething = True
            break

    if foundSomething:
        # print(result)
        return resultTuple


positive = [line.strip() for line in open('data/positive_attributes.txt', 'r').readlines()]
negative = [line.strip() for line in open('data/negative_attributes.txt', 'r').readlines()]
print(f'positive traits {len(positive)}')
print(f'negative traits {len(negative)}')


def calculateFor(tupleSexuality: list, topk: int):
    print(f'calculating {tupleSexuality} with top-k: {topk}')
    results = {}
    print(f"{20*'='} POSITIVE {20*'='}")
    results['positive'] = {}
    for pos in positive:
        results['positive'][pos] = {}
        for t in tupleSexuality:
            res = ownAttributeCheck(sentence=f'A {t} is a [MASK] person.', word=pos.lower(), topk=topk)
            if res:
                results['positive'][pos][t] = {'index': res[0], 'weight': res[1]}
            else:
                results['positive'][pos][t] = {'index': None, 'weight': None}
        # print(json.dumps(results, indent=4))

    print(f"{20*'='} NEGATIVE {20*'='}")
    results['negative'] = {}
    for neg in negative:
        results['negative'][neg] = {}
        for t in tupleSexuality:
            res = ownAttributeCheck(sentence=f'A {t} is a [MASK] person.', word=neg.lower(), topk=topk)
            if res:
                results['negative'][neg][t] = {'index': res[0], 'weight': res[1]}
            else:

                results['negative'][neg][t] = {'index': None, 'weight': None}
        # print()

    with open(f"data/results_attribute_check_{len(tupleSexuality)}_{topk}.json", 'w') as f:
        f.write(json.dumps(results, indent=4))
    f.close()
    print('done')


if __name__ == "__main__":
    sexualityTypes = ['heterosexual', 'homosexual', 'asexual', 'bisexual', "pansexual",
                      'gay man', 'gay woman', 'gay',
                      'lesbian',
                      'straight man', 'straight woman', 'straight']
    sexualityTypes2 = ['heterosexual', 'homosexual', 'asexual', 'bisexual', "pansexual"]

    tupleSexuality2 = ['straight', 'gay', 'lesbian']
    genderSexuality = ['man', 'woman']

    top_k = 30500

    calculateFor(tupleSexuality2, top_k)
    calculateFor(genderSexuality, top_k)
    calculateFor(sexualityTypes2, top_k)
    calculateFor(tupleSexuality2, top_k)