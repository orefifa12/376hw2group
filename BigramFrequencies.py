import nltk
from nltk.corpus import nps_chat
from nltk import bigrams
from collections import Counter

# Ensure the necessary parts of NLTK are available
nltk.download('nps_chat')

# Tokenize the NPS Chat corpus
tokens = nps_chat.words()

# Create bigrams from the tokens
bigram_list = list(bigrams(tokens))

# Count the frequency of each bigram
bigram_freq = Counter(bigram_list)

# Find the most common bigram
most_common_bigram = bigram_freq.most_common(1)

print(most_common_bigram)
