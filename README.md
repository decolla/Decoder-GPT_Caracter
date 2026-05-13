https://colab.research.google.com/drive/1ibo_hgIqV9si1HDGWCdUB_t1KBCC9uz_#scrollTo=2YdPkWPmN9kK

# Decoder-GPT: LLM em Nível de Caractere

## 📖 Descrição do Projeto

Este projeto consiste no desenvolvimento de uma Rede Neural com arquitetura **Decoder-only**, baseada no modelo **Transformers**, com foco em Processamento de Linguagem Natural (NLP). O objetivo principal foi construir uma LLM (Large Language Model) do zero, desde o tratamento de dados até o loop de treinamento e inferência autoregressiva.

O modelo foi treinado em nível de caractere (estilo *nanoGPT*), utilizando obras literárias como *Dom Casmurro* (Machado de Assis) e *Duna* (Frank Herbert) para aprender padrões linguísticos, morfologia e estilos narrativos específicos.

## 🛠️ Arquitetura e Engenharia

A arquitetura foi implementada utilizando **PyTorch** e conta com as seguintes características técnicas:

* **Mecanismo de Atenção**: Multi-Head Self-Attention para captura de dependências de longo prazo.
* **Função de Ativação**: Uso de **GELU** (Gaussian Error Linear Unit) para transições de gradiente mais suaves.
* **Estabilidade**: Implementação de **LayerNorm** e **Conexões Residuais** para permitir o empilhamento de camadas sem degradação do sinal.
* **Otimização de Memória**: Técnica de **Weight Tying** (compartilhamento de pesos) entre as tabelas de embedding e a camada de saída (LM Head).
* **Inicialização**: Pesos inicializados com distribuição normal ($\sigma=0.02$) e escalonamento residual para evitar explosão de ativações.

## 💻 Infraestrutura e Otimização

O treinamento foi otimizado para execução em **CPU**, considerando as limitações de hardware local (GPU AMD RX 550 com suporte limitado a ROCm em arquiteturas Polaris):

* **Hardware**: Processador **AMD Ryzen 7 5700X** (8 cores / 16 threads).
* **Sistema**: Linux (Ubuntu) utilizando o kernel **XanMod** para otimização de processamento multitarefa.
* **Paralelismo**: O código utiliza 16 threads simultâneos para maximizar a vazão de dados no treinamento.
* **Isolamento**: Uso de ambiente virtual (`venv`) para gestão de dependências e reprodutibilidade.

## 📊 Análise de Resultados

O modelo demonstrou uma curva de aprendizado sólida, partindo de uma Loss inicial de **4.7** (condizente com a entropia de um vocabulário de ~110 caracteres).

### Métricas de Convergência (Dataset: Duna)

* **Melhor Performance**: O "Sweet Spot" de generalização foi atingido no **Passo 15.000**, com uma **Loss de Validação de 1.1444**.
* **Overfitting**: O treinamento foi interrompido no passo 22.500 ao observar que a Loss de Treino continuava caindo (0.82), mas a de Validação começou a divergir (1.17), indicando memorização excessiva do dataset.

### Geração de Texto (Amostragem)

O modelo foi capaz de reproduzir entidades complexas e estruturas de diálogo:

* **Temperatura 0.6**: Produziu textos coerentes e fiéis ao vocabulário original ("Reverenda Madre", "Atreides").
* **Temperatura 1.2**: Gerou textos criativos e exploratórios, mantendo a sonoridade do universo de Arrakis.

## 🚀 Como Executar

1. Clone o repositório: `git clone https://github.com/decolla/Decoder_Obra_Literaria.git`
2. Crie o ambiente virtual: `python3 -m venv venv && source venv/bin/activate`
3. Instale as dependências: `pip install torch numpy`
4. Para gerar texto: `python src/generate.py` (Certifique-se de ter o arquivo `melhor_modelo_duna.pth` no diretório).
