# PLN na Prática: dos Dados à Modelagem

Workshop hands-on de Processamento de Linguagem Natural para a **Semana da Computação da UFPA (SECOMP 2026)**.

Sessão única de 2 horas onde cada participante executa um notebook completo no Google Colab, treinando modelos para prever a nota (1–5 estrelas) de reviews de e-commerce em português brasileiro e competindo por prêmio via leaderboard ao vivo.

## O que é o workshop

O notebook [`plncc.ipynb`](plncc.ipynb) é o roteiro completo da aula: texto explicativo, código executável e exercícios guiados, sem slides separados. Os participantes não escrevem código, apenas executam células prontas e ajustam hiperparâmetros via formulários interativos do Colab (`#@param`).

O fio condutor é a provocação **"Odiei, 5 estrelas"**: quando texto e nota discordam, qual dos dois é o rótulo certo? Essa pergunta reaparece ao longo do pipeline e é respondida no fechamento.

## Dataset

[B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01): 132.373 reviews de e-commerce em pt-BR nativo, com `overall_rating` (1 a 5 estrelas) como alvo.

O enquadramento de negócio é **supervisão distante** (*distant supervision*): treina-se onde o rótulo é gratuito (review com estrela), aplica-se o modelo onde a nota não existe (SAC, WhatsApp, respostas abertas de NPS).

## Pipelines

| # | Pipeline | Papel | Premiável? |
|---|----------|-------|:----------:|
| 1 | TF-IDF + Regressão Logística | Baseline rápido | Não |
| 2 | Embeddings (NILC/fastText) + LSTM | **Foco principal, tunável** | **Sim** |
| 3 | Zero-shot com Gemini API | Comparação custo-benefício | Não |

A métrica oficial da competição é **QWK** (*Quadratic Weighted Kappa*), que penaliza erros maiores na escala ordinal. Macro-F1 e MAE aparecem como contexto e critério de desempate.

## Estrutura do notebook

```
1. Introdução
   1.1 Processamento de Linguagem Natural
   1.2 Motivos para ensinar máquinas a entender linguagem
   1.3 Programação de computadores vs linguagem natural
   1.4 Palavras, tokens e morfemas
   1.5 NLU vs NLG
   1.6 Ambiente de desenvolvimento: Notebooks

2. Preliminares
   2.1 Importação dos pacotes
   2.2 Dados e regra de negócio

3. Pipeline como um Processo KDD
   3.1 Seleção (EDA, filtro de atributos)
   3.2 Pré-processamento (ausentes, duplicados, capitalização,
       stopwords, pontuação, emojis, elongação)
   3.3 Pipeline 1: TF-IDF + Regressão Logística
   3.4 Pipeline 2: Embeddings + LSTM
   3.5 Pipeline 3: Zero-shot com Gemini
   3.6 Detecção de incoerência

4. Submissão da competição
```

## Hiperparâmetros configuráveis (competição)

Os participantes ajustam estes parâmetros via formulário do Colab para o pipeline LSTM (único premiável):

| Parâmetro | Default | Opções |
|-----------|---------|--------|
| `TEST_SIZE` | 0.2 | 0.1, 0.15, 0.2, 0.25, 0.3 |
| `VOCAB_SIZE` | 50000 | livre |
| `MAX_LEN` | 100 | livre |
| `LSTM_UNITS` | 32 | 32, 64, 128 |
| `DROPOUT` | 0.3 | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `RECURRENT_DROPOUT` | 0.0 | 0.0, 0.1, 0.2 |
| `LEARNING_RATE` | 1e-3 | 1e-4, 5e-4, 1e-3, 5e-3 |
| `USE_CLASS_WEIGHTS` | False | True, False |
| `EPOCHS` | 5 | livre |
| `BATCH_SIZE` | 256 | livre |

## Como usar

### Participante

1. Abra o notebook no Google Colab
2. Execute as células em ordem
3. Ajuste hiperparâmetros nos formulários e re-execute
4. Submeta o arquivo `submissao.csv` via formulário do leaderboard

### Instrutor

**Antes do workshop:**

- Gerar o subconjunto de embeddings com e fazer upload para o Google Drive
- Separar o dataset de leaderboard
- Verificar rate limits do tier gratuito da API Gemini

**Durante o workshop:**

- Projetar o leaderboard ao vivo
- Usar a compilação de hiperparâmetros (seção 4) para discutir escolhas dos vencedores

## Estrutura do repositório

```
plncc.ipynb              # Notebook principal (roteiro completo)
requirements.txt         # Dependências Python
images/                  # Imagens usadas no notebook
```

## Requisitos

- Python 3.10+
- Google Colab (recomendado) ou ambiente local com GPU

Instalar dependências localmente:

```bash
pip install -r requirements.txt
```

## Licença

[MIT](LICENSE) — Helder Matos, 2026.
