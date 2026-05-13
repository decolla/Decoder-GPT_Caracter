import torch
import math
from model import GPTLanguageModel
from prepare import vocab_size, get_batch

torch.set_num_threads(16) # OTMIZAÇÃO PARA PROCESSADORES COM MUITOS CORES

# HIPERPARÂMETROS
batch_size = 32  # exemplos processados em paralelo P: 64
block_size = 128 # janela de contexto P: 256
max_iters = 5000 # rodadas de treino
eval_interval = 500 # a cada X iterações, avalia o modelo
learning_rate = 3e-4 # taxa de aprendizado
device = 'gpu' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # quantos batches usar para calcular a loss média
n_embd = 128 # dimensão do embedding para calcúlo de matriz. P: 384
n_head = 4 # número de cabeças. P: 6
n_layer = 4 # número de camadas. P: 6
dropout = 0.2

model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device)
m = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# (nanoGPT) separa os parâmetros que devem sofrer weight decay daqueles que não devem

# cria um dicionário com todos os parâmetros que precisam de ajuste
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

optim_groups = [
    {'params': decay_params, 'weight_decay': 1e-1},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
# otimizador é configurado em dois grupos distintos
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

@torch.no_grad()
def estimate_loss():
    out = {}

    # muda o modelo para modo de avaliação, para que o dropout não descarte informações durante o treino
    model.eval()
    for split in ['train', 'val']:
        # tira a média de vários batches para ter uma visão estável de loss
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # garante que o modelo volte para o modo de aprendizado antes de continuar o loop
    model.train()
    return out

def get_lr(it):
    # Decaimento de cosseno simples para aprendizado estacionar
    if it > max_iters: return 3e-5 # min_lr
    decay_ratio = it / max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 3e-5 + coeff * (learning_rate - 3e-5)


# Loop de Treino principal
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Passo {iter}: Loss Treino {losses['train']:.4f}, Loss Val {losses['val']:.4f}")

    # busca uma amostra de entrada e o alvo esperado
    xb, yb = get_batch('train', batch_size, block_size)
    # move os dados para o device (CPU)
    xb, yb = xb.to(device), yb.to(device)

    # atualiza a taxa de aprendizado do otimizador
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # tentar prever o próximo caractere e calcula a taxa de erro
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    # BACKPROPAGATION: calcula como peso contribuiu para cada erro
    loss.backward()
    # ajusta os pesos para diminuir o erro na próxima vez
    optimizer.step()

# salvar o modelo
torch.save(model.state_dict(), 'modelo_duna.pth')