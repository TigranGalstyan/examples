import torch
import data

def generate(data_src='data/', checkpoint='./model.pt', prime='', words=200, seed=1111, cuda=False, temperature=1.0, log_interval=100):
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    if temperature < 1e-3:
        raise ValueError("temperature has to be greater or equal 1e-3")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    corpus = data.Corpus(data_src)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    line_words = []

    if prime:
        for word in prime.split(' '):
            input.fill_(corpus.dictionary.word2idx[word])
            line_words.append(word + ' ')
    with torch.no_grad():  # no tracking history
        for i in range(words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            line_words.append(word + ('\n' if i % 20 == 19 else ' '))

    return ''.join(line_words)
