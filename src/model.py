import torch
import math
import torch.nn as nn
from torch.nn import functional as F

# classe para criar um bloco de transformer
class Head(nn.Module):

    # classe head para comunicação entre caracteres

    # n-embd: tamanho do embedding (vetor que representa o caractere)
    # head-size: dimensão da cabeça de atenção (n_embd / n_head)
    # bias: deslocamento da função, não usarei porque o LayerNorm normaliza cada pnto de dados individualmente
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        # o qure o caractere contém
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # o que o caractere esta procurando
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # a informação que será passada caso a key e query sejam iguais (dar match)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # CASUAL MASKING
        # criar uma matriz triangular inferior (0 e 1) para evitar que o modelo veja a resposta no futuro
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    # calcula a importância de cada caractere para o contexto
    def forward(self, x):
        # B: batch_size, exemplos processados em paralelo
        # T: tempo, comprimento da sequência de caracteres, o block size
        # C: channels, dimensão do embedding (n_embd)
        B, T, C = x.shape

        # compara o caractere com as informações que ele pode ter
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # calcula pesos de atenção ("afinidade")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # o C evita valores muito altos, estabilizando o treinamento

        # CASUAL MASKING
        # evita que o modelo veja a resposta no futuro, preenchendo com -inf as posições futuras
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax normaliza os valores para que a soma das probabilidades seja igual a 1
        wei = F.softmax(wei, dim=-1)

        # dropout para evitar overfitting
        wei = self.dropout(wei)

        # gera os valores para cada caractere
        v = self.value(x)

        # multiplica as probabilidades pelos valores
        # resulta uma média ponderada das informações que o modelo considerou relevantes
        return wei @ v

# classe para usar várias cabeças de atenção em paralelo
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()

        # cria uma lista de cabeças de atenção independentes, onde cada uma aprenderá um contexto diferente
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])

        # camada linear final para juntar o que cada cabeça aprendeu
        self.proj = nn.Linear(n_embd, n_embd)

        # dropout para evitar overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # executa cada cabeça individualmente sobre a entrada x e concatena os resultados ao longo do n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # aplica um descarte aleatório de neurônios para evitar overfitting
        return self.dropout(self.proj(out))

# após comunicação entre caracteres, o modelo passa para uma camada de feed-forward
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        # permite estacar múltiplas camadas e modulos em uma ordem específica
        self.net = nn.Sequential(

            # camada que expande o vetor em 4 vezes, criando um espaço de cálculo maior
            nn.Linear(n_embd, 4 * n_embd),

            # função de ativação não linear
            nn.GELU(), #nn.ReLU(), troquei por GELU por ser mais suave e preferível em transformers modernos

            # comprime o vetor de volta para o tamanho original
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# modulo que une a atenção e o feed-forward
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        # define o tamanho de cada cabeça, dividindo pela dimensão total
        head_size = n_embd // n_head

        # realiza multi head attention
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        # realiza feed-forward (fluxo de dados)
        self.ffwd = FeedForward(n_embd, dropout)

        # LayerNorm para normalizar os dados de entrada, garantindo média e variância consistentes
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Conexão residual: aplica atenção e soma o resultado ao x original
        x = x + self.sa(self.ln1(x))
        # aplica o feed-forward e soma o resultado ao x original
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        self.device = device
        self.block_size = block_size

        # converte os Ids dos caracteres em embeddings (vetores de tamanho n_embd)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # noção de onde caractere está na sequência
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # gera profundidade aplicando blocos de transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        # normalização final
        self.ln_f = nn.LayerNorm(n_embd)

        # camad afinal para gerar de volta as probabilidades de cada caractere
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # weight tying para economizar memória
        self.token_embedding_table.weight = self.lm_head.weight
        # aplica os pesos iniciais para os embeddings
        self.apply(self._init_weights)

        # (nanoGPT) evita que ativações cresçam muito
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('net.2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    # define que todos os pesos lineares devem começar com uma distribuição normal de desvio padrão 0.02 e vieses em 0
    # evita que o modelo comece com valores muito altos ou muito baixos, o que trava o aprendizado
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx e targets são ambos tensores (B,T) de inteiros
        # transforma os indices de caracteres em embeddings de token e posição
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # os vetores são somados e passam pela pilha de blocos
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # realiza a atenção e feed-forward nos blocos
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        # gera os logits (pontuações) para prever qual será a próxima letra
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # se houver targets calcula-se a cross entropy loss (entre 0 e 1)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # RODAR O MODELO
    def generate(self, idx, max_new_tokens, temperature=1.2):
        # idx é uma matriz (B, T) de índices no contexto atual
        for _ in range(max_new_tokens):
            # corta o idx para o tamanho máximo do bloco (block_size)
            idx_cond = idx[:, -self.block_size:]
            # obtém as previsões
            logits, loss = self(idx_cond)

            # foca apenas no último passo no tempo
            logits = logits[:, -1, :] / temperature # TEMPERATURA DO MODELO

            # aplica softmax para obter probabilidades
            probs = F.softmax(logits, dim=-1)

            # amostra da distribuição
            idx_next = torch.multinomial(probs, num_samples=1)
            # anexa o índice amostrado à sequência corrente
            idx = torch.cat((idx, idx_next), dim=1)
        return idx