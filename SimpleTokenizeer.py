import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab #A
        self.int_to_str = {i:s for s,i in vocab.items()} #B

    def encode(self, text): #C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids): #D
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #E
        return text

def test():
    from importlib.metadata import version
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Akwirw ier"
    # text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)

# def testSimpleTokenizer():
    # tokenizer = SimpleTokenizerV1(vocab)
    # text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable
    # pride."""
#     ids = tokenizer.encode(text)
#     print(ids)

def testLoad():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        enc_text = tokenizer.encode(raw_text)
        print(len(enc_text))

if __name__ == "__main__":
    testLoad()