import re

with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# NOTE: Preprocessing the raw text to remove punctuation and split into words or tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# NOTE: Converting tokens into Token IDs
# Let's first create a vocabulary
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {word: index for index, word in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
