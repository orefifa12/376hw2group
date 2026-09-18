from collections import defaultdict
import random

import nltk
from nltk import trigrams
from nltk.corpus import nps_chat

# Download required corpora if missing
for package in ['nps_chat', 'punkt', 'punkt_tab']:
    try:
        nltk.download(package, quiet=True)
    except Exception:
        pass

try:
    corpus_posts = nps_chat.posts()
    if not corpus_posts:
        raise ValueError("NPS Chat corpus is empty")
    print("NPS Chat corpus loaded successfully")
except Exception as e:
    print(f"Error loading NPS Chat corpus: {e}")
    raise

# Build a trigram model from the chat posts
model = defaultdict(lambda: defaultdict(float))

for sentence in corpus_posts:
    if len(sentence) < 2:
        continue
    for w1, w2, w3 in trigrams(sentence):
        model[(w1, w2)][w3] += 1.0

# Convert counts into probabilities
for context in model:
    total = sum(model[context].values())
    for next_word in model[context]:
        model[context][next_word] /= total

# Print an example of the learned model
print("Example context:", dict(model.get(("part", "of"), {}).most_common(5) if hasattr(model.get(("part", "of"), {}), 'most_common') else {}))

# Generate sentences from a starting sequence

def generate_sentence(starting_words):
    text = list(starting_words)

    while len(text) < 20:
        context = tuple(text[-2:])
        choices = model.get(context)
        if not choices:
            break

        next_word = random.choices(
            population=list(choices.keys()),
            weights=list(choices.values()),
            k=1,
        )[0]
        text.append(next_word)

        # Stop if a natural sentence boundary is reached
        if next_word in {'.', '?', '!'}:
            break

    return ' '.join(text)


# Generate 20 sentences
for _ in range(20):
    print(generate_sentence(["part", "of", "the"]))
