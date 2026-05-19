import torch
from model import GPTLanguageModel
from prepare import vocab_size, itos, decode, stoi, encode

# MESMOS HIPERPARÂMETROS DO TREINO
n_embd = 256
n_head = 8
n_layer = 6
block_size = 128
dropout = 0.2
device = 'cpu' # mantem cpu: AMD® Ryzen 7 5700 × 16

# cria estrutura primeiro
model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device)

# carrega os pesos guardados
model.load_state_dict(torch.load('melhor_modelo_duna.pth'))
m = model.to(device)
m.eval() # modo de avaliação (desliga dropout)

# GERAÇÃO
print("--- GERANDO TEXTO AO ESTILO DE DUNA ---")

# começamos com um contexto vazio (um caractere de "nova linha" ou espaço)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# geramos 500 novos caracteres
generated_tokens = m.generate(context, max_new_tokens=1000)[0].tolist()

# traduz de números para texto e imprime
print(decode(generated_tokens))

memoria_pesos_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
memoria_pesos_mb = memoria_pesos_bytes / (1024 * 1024)

print(f"Número total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"Tamanho de cada elemento na memória: {model.lm_head.weight.element_size()} bytes")
print(f"Memória RAM consumida puramente pelos pesos: {memoria_pesos_mb:.2f} MB")