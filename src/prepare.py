import torch

# carrega o txt para leitura
with open('data/duna/duna.txt', 'r') as f:
# with: gerencia recursos automaticamente, garantindo que ações sejam executadas
    text = f.read()
    # lê o arquivo inteiro e salva em um texto

# cria o vocabulário de caracteres únicos em lista de ordem alfabética
# o desafio pede que seja "caracteres por caracteres"
chars = sorted(list(set(text)))
# set: remove duplicatas, list: converte em lista, sorted: ordena
# sorted: remove duplicatas
vocab_size = len(chars)

# traduz para inteiros
stoi = { ch:i for i,ch in enumerate(chars) }
# string to integer

itos = { i:ch for i,ch in enumerate(chars) }

# transforma uma frase em lista de números
encode = lambda s: [stoi[c] for c in s]
# função de encoding char -> id

# transforma uma lista de números em texto
decode = lambda l: ''.join([itos[i] for i in l])
# função de decoding id -> char

# converte a lista em tensor
data = torch.tensor(encode(text), dtype=torch.long)
# tensores de pytorch

# calcula as 90% iniciais do texto
n = int(0.9 * len(data))

# realiza o slicing para obter 90% valores e os 10% para treino
train_data = data[:n]
val_data = data[n:]

# hiperparâmetros de infraestrutura
batch_size = 32  # Quantas sequências processar em paralelo
block_size = 64  # Comprimento máximo do contexto (janela de atenção)

# RANDOMIZAÇÃO
#  retorne as sequências x (entrada) e y (alvo), deslocado um caractere para a direita
def get_batch(split, b_size, blk_size):
    # selecionar se usamos dados de treino ou teste
    data_subset = train_data if split == 'train' else val_data

    # gerar índices aleatórios para o início das sequências
    ix = torch.randint(len(data_subset) - blk_size, (b_size,))
    # vetor para acessar indíces aleatórios de 0 até b_size

    # pega um texto do tamanho do block
    x = torch.stack([data_subset[i:i + blk_size] for i in ix])
    # transforma em uma matriz de b_size x blk_size

    # pega do mesmo bloco, deslocando um caractere para frente
    y = torch.stack([data_subset[i + 1:i + blk_size + 1] for i in ix])
    # target: mesma coisa de x, só que deslocado um caractere para frente, para prever o caractere seguinte

    return x, y