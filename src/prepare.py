import torch

# carrega o txt para leitura
with open('data/dom-casmurro/dom-casmurro.txt', 'r') as f:
    text = f.read()

# cria o vocabulário de caracteres únicos em lista de ordem alfabética
# o desafio pede que seja "caracteres por caracteres"
chars = sorted(list(set(text)))
vocab_size = len(chars)

# traduz para inteiros
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# transforma uma frase em lista de números
encode = lambda s: [stoi[c] for c in s]

# transforma uma lista de números em texto
decode = lambda l: ''.join([itos[i] for i in l])

# converte a lista em tensor
data = torch.tensor(encode(text), dtype=torch.long)

# calcula as 90% iniciais do texto
n = int(0.9 * len(data))

# realiza o slicing para obter 90% valores e os 10% para treino
train_data = data[:n]
val_data = data[n:]

# hiperparâmetros de infraestrutura
batch_size = 32  # Quantas sequências processar em paralelo
block_size = 64  # Comprimento máximo do contexto (janela de atenção)

#  retorne as sequências x (entrada) e y (alvo), deslocado um caractere para a direita
def get_batch(split):
    # selecionar se usamos dados de treino ou teste
    data_subset = train_data if split == 'train' else val_data

    # gerar índices aleatórios para o início das sequências
    ix = torch.randint(len(data_subset) - block_size, (batch_size,))

    # pega um texto do tamanho do block
    x = torch.stack([data_subset[i:i + block_size] for i in ix])
    # pega do mesmo bloco, deslocando um caractere para frente
    y = torch.stack([data_subset[i + 1:i + block_size + 1] for i in ix])

    return x, y