import torch
from model import GPTLanguageModel
from prepare import vocab_size, itos, decode, stoi, encode

# 1. MESMOS HIPERPARÂMETROS DO TREINO
n_embd = 128
n_head = 4
n_layer = 4
block_size = 128
dropout = 0.2
device = 'cpu' # mantem cpu: AMD® Ryzen 7 5700 × 16

# 2. CARREGAR O MODELO
# Criamos a estrutura primeiro
model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device)

# Carregamos os pesos guardados
model.load_state_dict(torch.load('modelo_duna.pth'))
m = model.to(device)
m.eval() # Modo de avaliação (desliga dropout)

# 3. LÓGICA DE GERAÇÃO
print("--- GERANDO TEXTO AO ESTILO DE DUNA ---")

# Começamos com um contexto vazio (um caractere de "nova linha" ou espaço)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Geramos 500 novos caracteres
# A função .generate() já está na tua classe GPTLanguageModel
generated_tokens = m.generate(context, max_new_tokens=500)[0].tolist()

# Traduzimos de números para texto e imprimimos
print(decode(generated_tokens))