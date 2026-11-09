![Cabecalho](figuras/Cabecalho.png)

# Análise da biodegradabilidade de moléculas com algoritmos preditivos

Projeto desenvolvido para a disciplina de Aprendizado de Máquina da Ilum Escola de Ciências, com o objetivo de investigar e comparar modelos de classificação para a previsão da biodegradabilidade de moléculas a partir de seus descritores moleculares.

## O que é este projeto?

Este projeto utiliza técnicas de aprendizado de máquina supervisionado para classificar moléculas em duas categorias:

- **RB (Readily Biodegradable):** moléculas prontamente biodegradáveis.
- **NRB (Not Readily Biodegradable):** moléculas não prontamente biodegradáveis.

Para isso, é utilizado o conjunto de dados [QSAR Biodegradation](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation), disponibilizado pelo UCI Machine Learning Repository.

O conjunto contém 1.055 amostras, representadas por 41 descritores moleculares e uma variável-alvo que indica a classe de biodegradabilidade.

O projeto contempla a análise exploratória dos dados, construção de pipelines de classificação, comparação de modelos, otimização de hiperparâmetros e avaliação de desempenho com base na métrica AP (Average Precision).

## Modelos utilizados

São investigados cinco algoritmos de classificação:

- **K-Vizinhos Mais Próximos (KNN):** classifica as moléculas considerando a proximidade entre seus descritores e os exemplos do conjunto de treinamento.
- **Regressão Logística:** estima a probabilidade de uma molécula pertencer a determinada classe por meio de uma combinação linear dos atributos.
- **Árvore de Decisão:** realiza classificações por meio de divisões hierárquicas dos atributos.
- **Floresta Aleatória:** combina múltiplas árvores de decisão para produzir uma classificação mais robusta.
- **Naive Bayes Gaussiano:** utiliza o teorema de Bayes, assumindo independência condicional entre os atributos e distribuições gaussianas.

Um modelo *baseline*, baseado na previsão da classe majoritária, também é utilizado como referência de desempenho.

## Metodologia

O desenvolvimento do projeto é dividido nas seguintes etapas:

1. **Análise exploratória:** investigação da distribuição das classes, tipos de dados, estatísticas descritivas e distribuições dos descritores moleculares.
2. **Preparação dos dados:** separação entre atributos preditores e variável-alvo, identificação de valores ausentes e agrupamento de amostras com descritores semelhantes.
3. **Separação dos conjuntos:** utilização do `StratifiedGroupKFold` para separar treinamento e teste, considerando o balanceamento das classes e os grupos de amostras.
4. **Construção dos modelos:** criação de pipelines com os algoritmos selecionados e seus respectivos procedimentos de pré-processamento.
5. **Validação cruzada inicial:** comparação dos modelos com configurações iniciais.
6. **Otimização de hiperparâmetros:** utilização da biblioteca Optuna, com o sampler TPE e 150 tentativas por algoritmo.
7. **Validação cruzada final:** comparação dos modelos após a otimização, considerando a média da e o desvio padrão do desempenho.
8. **Avaliação no conjunto de teste:** análise das matrizes de confusão e das curvas Precisão-Recall para avaliar a capacidade de generalização dos modelos.

### Pré-processamento

Os pipelines utilizam a padronização dos atributos por meio de `StandardScaler` e a Análise de Componentes Principais (PCA), cuja utilização é opcional durante a otimização.

O PCA é avaliado com uma variância explicada entre 70% e 99%, permitindo investigar se a redução de dimensionalidade contribui para o desempenho dos classificadores.

### Métrica de avaliação

A principal métrica utilizada é o **Average Precision (AP)**, considerando a classe RB como positiva.

A escolha é motivada pelo desbalanceamento entre as classes do conjunto de dados. O AP permite avaliar o desempenho dos modelos considerando diferentes níveis de precisão e recall, oferecendo uma análise mais direcionada à identificação das moléculas prontamente biodegradáveis.

## Estrutura do projeto

```text
BioPredict/
│
├── HPC/                              # Arquivos relacionados à execução em ambiente HPC
│
├── data/                             # Conjunto de dados utilizado no projeto
│
├── figuras/                          # Figuras utilizadas no README
│
├── .gitignore                        # Arquivos e diretórios ignorados pelo Git
│
├── Biodeg_previsao.ipynb             # Notebook principal de análise e modelagem
│
├── LICENSE                           # Licença de uso do projeto
│
├── README.md                         # Documentação do projeto
│
├── biodegradabilidade_arvore.db      # Banco SQLite da otimização da Árvore de Decisão
│
├── biodegradabilidade_floresta.db    # Banco SQLite da otimização da Floresta Aleatória
│
├── biodegradabilidade_gaussNB.db     # Banco SQLite da otimização do Naive Bayes Gaussiano
│
├── biodegradabilidade_knn.db         # Banco SQLite da otimização do KNN
│
├── biodegradabilidade_regressão.db   # Banco SQLite da otimização da Regressão Logística
│
└── requisitos_projeto_final_ML.todo  # Requisitos e tarefas do projeto
```

O diretório `data/` deve conter o arquivo `biodeg.csv`. Os resultados são gerados durante a execução do script.

## Como instalar

O repositório pode ser instalado clonando o repositório ou realizando o download do arquivo zip. Além disso, se utiliza Python e bibliotecas de aprendizado de máquina e análise de dados.

Instale com:

```bash
pip install numpy pandas scikit-learn optuna joblib matplotlib seaborn
```

Para executar o notebook, também é necessário instalar o Jupyter:

```bash
pip install jupyter
```

## Como executar

### Notebook

Após instalar as bibliotecas, abra o arquivo `Biodeg_previsao.ipynb` e execute as células na sequência apresentada.

## Execução em HPC

O script foi preparado para execução em ambientes de computação de alto desempenho (HPC), utilizando o backend não interativo do Matplotlib e permitindo configurar o número de CPUs utilizadas pelos estimadores.

Em clusters que utilizam o gerenciador de tarefas SLURM, pode ser utilizado um arquivo de submissão como o exemplo abaixo:

```bash
#!/bin/bash
#SBATCH --job-name=Tiamat_Optuna_HPC
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodelist=work1
#SBATCH --time=00-01:00:00
#SBATCH --output=logs/saida_%j.out
#SBATCH --error=logs/erro_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

 
# Execução correta chamando o python:

python3 CAMINHO/tiamat_optuna_hpc.py \
    --data data/biodeg.csv \
    --n-trials 150 \
    --output resultados_hpc
```

Para submeter o trabalho:

```bash
sbatch tiamat.sh
```

Os recursos de processamento e memória devem ser ajustados de acordo com as configurações e regras do HPC utilizado.

### Resultados gerados

Durante a execução, são armazenados:

- Bancos SQLite independentes para os estudos Optuna de cada algoritmo.
- Histórico dos trials e hiperparâmetros avaliados.
- Melhores hiperparâmetros encontrados.
- Importância estimada dos hiperparâmetros.
- Resultados da validação cruzada.
- Métricas de avaliação no conjunto de teste.
- Matrizes de confusão.
- Curvas Precisão-Recall.
- Modelos treinados no formato `.joblib`.
- Resumo da execução em formato JSON.

Esses resultados permitem comparar os algoritmos e investigar quais configurações apresentam melhor desempenho na classificação da biodegradabilidade molecular.

## Referências e ferramentas

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Optuna](https://optuna.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

## Autoria

**Autor:** Leonardo Ramos Gomes da Silva

**Instituição:** Ilum Escola de Ciências — Centro Nacional de Pesquisa em Energia e Materiais (CNPEM).

**Disciplina:** Aprendizado de Máquina.

## Agradecimentos

**Docente:** Prof. Dr. Daniel Roberto Cassar, ppela dedicação, paciência e por compartilhar seu conhecimento de forma tão leve e didática. Suas aulas fizeram toda a diferença no nosso aprendizado!

## Anexo

### Descritores moleculares

1. SpMax_L: Maior autovalor da matriz de Laplace.
2. J_Dz(e): Índice semelhante ao de Balaban, calculado a partir da matriz de Barysz ponderada pela eletronegatividade de Sanderson.
3. nHM: Número de átomos pesados (átomos diferentes de hidrogênio).
4. F01[N-N]: Frequência de ligações N–N a uma distância topológica de 1.
5. F04[C-N]: Frequência de pares C–N a uma distância topológica de 4.
6. NssssC: Número de átomos do tipo ssssC.
7. nCb-: Número de átomos de carbono C(sp²) substituídos em anéis benzênicos.
8. C%: Porcentagem de átomos de carbono.
9. nCp: Número de átomos de carbono primários terminais C(sp³).
10. nO: Número de átomos de oxigênio.
11. F03[C-N]: Frequência de pares C–N a uma distância topológica de 3.
12. SdssC: Soma dos estados E dos átomos do tipo dssC.
13. HyWi_B(m): Índice semelhante ao de Hyper-Wiener (função logarítmica), calculado a partir da matriz de Burden ponderada pela massa.
14. LOC: Índice de centralidade de Lopping.
15. SM6_L: Momento espectral de ordem 6 da matriz de Laplace.
16. F03[C-O]: Frequência de pares C–O a uma distância topológica de 3.
17. Me: Eletronegatividade média de Sanderson dos átomos, normalizada em relação ao carbono.
18. Mi: Potencial médio de primeira ionização, normalizado em relação ao carbono.
19. nN-N: Número de grupos hidrazina.
20. nArNO2: Número de grupos nitro aromáticos.
21. nCRX3: Número de grupos CRX3.
22. SpPosA_B(p): Soma espectral positiva normalizada da matriz de Burden ponderada pela polarizabilidade.
23. nCIR: Número de ciclos (anéis) na estrutura molecular.
24. B01[C-Br]: Presença ou ausência de ligação C–Br a uma distância topológica de 1.
25. B03[C-Cl]: Presença ou ausência de ligação C–Cl a uma distância topológica de 3.
26. N-073: Grupos Ar₂NH / Ar₃N / Ar₂N-Al / R..N..R.
27. SpMax_A: Maior autovalor da matriz de adjacência (índice de Lovász–Pelikan).
28. Psi_i_1d: Índice de pseudoconectividade do estado intrínseco — tipo 1d.
29. B04[C-Br]: Presença ou ausência de ligação C–Br a uma distância topológica de 4.
30. SdO: Soma dos estados E dos átomos do tipo dO.
31. TI2_L: Segundo índice de Mohar da matriz de Laplace.
32. nCrt: Número de átomos de carbono terciários C(sp³) pertencentes a anéis.
33. C-026: Grupo estrutural R–CX–R.
34. F02[C-N]: Frequência de pares C–N a uma distância topológica de 2.
35. nHDon: Número de átomos doadores de ligações de hidrogênio (N e O).
36. SpMax_B(m): Maior autovalor da matriz de Burden ponderada pela massa.
37. Psi_i_A: Índice de pseudoconectividade do estado intrínseco — média do tipo S.
38. nN: Número de átomos de nitrogênio.
39. SM6_B(m): Momento espectral de ordem 6 da matriz de Burden ponderada pela massa.
40. nArCOOR: Número de grupos ésteres aromáticos.
41. nX: Número de átomos de halogênios.

Classe experimental:

- RB (Readily Biodegradable): Prontamente biodegradável.
- NRB (Not Readily Biodegradable): Não prontamente biodegradável.