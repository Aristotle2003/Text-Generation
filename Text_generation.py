import string
import math
import random

humpath = "hum.txt"
gptpath = "gpt.txt"
allowed_punctuation = [',', '.', '?', '!']

def modify(text):
    documents = text.split('\n')
    processed_text = ""
    for document in documents:
        cleaned_document = ""
        for char in document.lower():
            if char not in string.punctuation or char in allowed_punctuation:
                cleaned_document += char
        processed_document = "<START> " + cleaned_document.strip() + " <END>"
        processed_text += processed_document + " "
    return processed_text.strip()

def divide(text):
    documents = text.split(' <END> ')
    documentnum = len(documents)
    split = int(documentnum * 0.9)

    train_docs = documents[:split]
    test_docs = documents[split:]
    train_data = ' <END> '.join(train_docs)
    test_data = ' <END> '.join(test_docs)
    return train_data, test_data

def extract_ngrams(document, n):
    words = document.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = words[i:i+n]
        ngrams.append(tuple(ngram))
    return ngrams

def generate_ngrams(text, n):
    documents = text.split(' <END> ')
    ngrams = []
    for document in documents:
        document_ngrams = []
        for ngram in extract_ngrams(document, n):
            document_ngrams.append(ngram)
        ngrams.extend(document_ngrams)
    return ngrams

def calculate_frequency(ngrams):
    frequency = {}
    for ngram in ngrams:
        if ngram in frequency:
            frequency[ngram] += 1
        else:
            frequency[ngram] = 1
    return frequency

def add_tokens(text):
    documents = text.split(' <END> ')
    new_docs = []
    for document in documents:
        new_doc = '<START> ' + document
        new_docs.append(new_doc)
    return ' <END> '.join(new_docs)

def prepare_ngrams(training_set, n):
    ngrams_list = []
    for doc in training_set:
        doc_with_tokens = add_tokens(doc, n)
        ngrams_list.extend(generate_ngrams(doc_with_tokens, n))
    return ngrams_list

def pick_next_word(ngram_freq, current_ngram, T):
    next_word_probs = {}
    for ngram in ngram_freq:
        if ngram[:-1] == current_ngram:
            next_word = ngram[-1]
            next_word_probs[next_word] = ngram_freq[ngram]
    adjusted_probs = {word: math.exp(freq / T) for word, freq in next_word_probs.items()}
    next_words = list(adjusted_probs.keys())
    probabilities = list(adjusted_probs.values())
    next_word = random.choices(next_words, weights=probabilities, k=1)[0]
    return next_word

def compose_sentence(ngram_freq, n, T):
    if n == 2:
        current_word = ("<START>",)
    elif n == 3:
        current_word = ("<START>", "<START>")
    sentence = []
    while len(sentence) <= 20:
        next_word = pick_next_word(ngram_freq, current_word, T)
        if next_word == "<END>":
            break
        sentence.append(next_word)
        if n == 2:
            current_word = (next_word,)
        elif n == 3:
            current_word = (current_word[1], next_word)
    return (' '.join(sentence))

with open(humpath) as hum:
    hum = hum.read()
with open(gptpath) as gpt:
    gpt = gpt.read()

hum = modify(hum)
gpt = modify(gpt)
hum_train, hum_test = divide(hum)
gpt_train, gpt_test = divide(gpt)

hum_train_for_trigram = add_tokens(hum_train)
gpt_train_for_trigram = add_tokens(gpt_train)

hum_train_bigrams = generate_ngrams(hum_train, 2)
hum_train_trigrams = generate_ngrams(hum_train_for_trigram, 3)
gpt_train_bigrams = generate_ngrams(gpt_train, 2)
gpt_train_trigrams = generate_ngrams(gpt_train_for_trigram, 3)

hum_train_bi_freq = calculate_frequency(hum_train_bigrams)
hum_train_tri_freq = calculate_frequency(hum_train_trigrams)
gpt_train_bi_freq = calculate_frequency(gpt_train_bigrams)
gpt_train_tri_freq = calculate_frequency(gpt_train_trigrams)

T = 35

hum_bigram_sentence = []
gpt_bigram_sentence = []
hum_trigram_sentence = []
gpt_trigram_sentence = []

for i in range(5):
    hum_bigram_sentence.append(compose_sentence(hum_train_bi_freq, 2, T))
    gpt_bigram_sentence.append(compose_sentence(gpt_train_bi_freq, 2, T))
    hum_trigram_sentence.append(compose_sentence(hum_train_tri_freq, 3, T))
    gpt_trigram_sentence.append(compose_sentence(gpt_train_tri_freq, 3, T))

for i in range(5):
    print("human bigram sentence: ", hum_bigram_sentence[i])
    print("gpt bigram sentence: ", gpt_bigram_sentence[i])
    print("human trigram sentence: ", hum_trigram_sentence[i])
    print("gpt trigram sentence: ", gpt_trigram_sentence[i])
