#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classificação da biodegradabilidade molecular (QSAR Biodegradation).

Script adaptado do notebook Tiamat(2).ipynb para execução em HPC.
- Lê data/biodeg.csv (ou caminho informado em --data).
- Separa treino/teste com StratifiedGroupKFold (10 partições; primeira partição).
- Otimiza KNN, Regressão Logística, Árvore de Decisão, Floresta Aleatória
  e Naive Bayes Gaussiano com Optuna/TPE.
- Usa AP (Average Precision), considerando RB como classe positiva.
- Salva estudos Optuna (SQLite), resultados, modelos e gráficos.

Exemplo:
    python tiamat_hpc.py --data data/biodeg.csv --n-trials 100 --output resultados

O script usa paralelismo dentro dos estimadores e mantém a validação cruzada
serial (n_jobs=1) para evitar paralelismo aninhado e excesso de processos.
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

# Limita threads de bibliotecas numéricas por processo; evita oversubscription.
# Ajuste OMP_NUM_THREADS/MKL_NUM_THREADS no script de submissão, se necessário.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica para HPC
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("default")
optuna.logging.set_verbosity(optuna.logging.INFO)

NOMES_COLUNAS = [
    "SpMax_L", "J_Dz_e", "nHM", "F01_N_N", "F04_C_N", "NssssC", "nCb",
    "C_percent", "nCp", "nO", "F03_C_N", "SdssC", "HyWi_B_m", "LOC",
    "SM6_L", "F03_C_O", "Me", "Mi", "nN_N", "nArNO2", "nCRX3",
    "SpPosA_B_p", "nCIR", "B01_C_Br", "B03_C_Cl", "N_073", "SpMax_A",
    "Psi_i_1d", "B04_C_Br", "SdO", "TI2_L", "nCrt", "C_026", "F02_C_N",
    "nHDon", "SpMax_B_m", "Psi_i_A", "nN", "SM6_B_m", "nArCOOR", "nX",
    "class",
]
ATRIBUTOS = NOMES_COLUNAS[:-1]
ALVO = "class"
SEMENTE_PADRAO = 73701

# Variável definida após a construção dos folds, para que o KNN nunca tente
# usar mais vizinhos do que o número de amostras disponível em um fold.
MAX_VIZINHOS_CV = 800
N_JOBS_MODELO = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Otimização e avaliação de modelos para biodegradabilidade."
    )
    parser.add_argument("--data", default="data/biodeg.csv",
                        help="Caminho do arquivo CSV separado por ponto e vírgula.")
    parser.add_argument("--output", default="resultados_hpc",
                        help="Diretório para resultados, modelos e bancos SQLite.")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Número de trials Optuna por algoritmo (padrão: 100).")
    parser.add_argument("--seed", type=int, default=SEMENTE_PADRAO,
                        help="Semente aleatória (padrão: 73701).")
    parser.add_argument("--n-jobs-modelo", type=int, default=None,
                        help="CPUs usadas internamente pelos estimadores. "
                             "Padrão: SLURM_CPUS_PER_TASK ou 1.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Número de folds na validação cruzada interna.")
    parser.add_argument("--split-folds", type=int, default=10,
                        help="Número de folds para separação treino/teste.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Tempo máximo de otimização por modelo, em segundos.")
    return parser.parse_args()


def sugere_pca(trial):
    """Escolhe se aplica PCA e, em caso afirmativo, sua variância explicada."""
    if trial.suggest_categorical("usar_pca", [True, False]):
        return PCA(
            n_components=trial.suggest_float("variancia_pca", 0.70, 0.99),
            svd_solver="full",
        )
    return "passthrough"


def cria_instancia_knn(trial):
    parametros = {
        "n_neighbors": trial.suggest_int(
            "num_vizinhos", 1, MAX_VIZINHOS_CV, log=True
        ),
        "weights": trial.suggest_categorical(
            "peso", ["uniform", "distance"]
        ),
        "metric": trial.suggest_categorical(
            "metrica", ["euclidean", "manhattan", "minkowski"]
        ),
        "n_jobs": N_JOBS_MODELO,
    }
    modelo = KNeighborsClassifier(**parametros)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", sugere_pca(trial)),
        ("classifier", modelo),
    ])


def cria_instancia_logreg(trial):
    parametros = {
        "C": trial.suggest_float("C", 0.001, 1000, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
        "max_iter": 5000,
        "n_jobs": N_JOBS_MODELO,
        "random_state": SEMENTE_ALEATORIA_GLOBAL,
    }
    modelo = LogisticRegression(**parametros)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", sugere_pca(trial)),
        ("classifier", modelo),
    ])


def cria_instancia_arvore(trial):
    parametros = {
        "criterion": trial.suggest_categorical(
            "criterio", ["gini", "entropy", "log_loss"]
        ),
        "splitter": trial.suggest_categorical(
            "estrategia_divisao", ["best", "random"]
        ),
        "max_depth": trial.suggest_int("profundidade_max", 2, 300),
        "min_samples_split": trial.suggest_int("min_exemplos_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_exemplos_folha", 1, 20),
        "max_features": trial.suggest_categorical(
            "max_atributos", [None, "sqrt", "log2"]
        ),
        "max_leaf_nodes": trial.suggest_int(
            "max_nos_folha", 10, 800, log=True
        ),
        "min_impurity_decrease": trial.suggest_float(
            "reducao_impureza_min", 0.0, 0.1
        ),
        "random_state": SEMENTE_ALEATORIA_GLOBAL,
    }
    modelo = DecisionTreeClassifier(**parametros)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", sugere_pca(trial)),
        ("classifier", modelo),
    ])


def cria_instancia_floresta(trial):
    parametros = {
        "n_estimators": trial.suggest_int("num_arvores", 100, 2000, log=True),
        "criterion": trial.suggest_categorical(
            "criterio", ["gini", "entropy", "log_loss"]
        ),
        "max_depth": trial.suggest_int("profundidade_max", 5, 800),
        "min_samples_split": trial.suggest_int("min_exemplos_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_exemplos_folha", 1, 20),
        "max_features": trial.suggest_float("num_max_atributos", 0.1, 1.0),
        "class_weight": trial.suggest_categorical(
            "peso_classes", [None, "balanced"]
        ),
        "max_samples": trial.suggest_float("fracao_amostras", 0.3, 1.0),
        "bootstrap": True,
        "n_jobs": N_JOBS_MODELO,
        "random_state": SEMENTE_ALEATORIA_GLOBAL,
    }
    modelo = RandomForestClassifier(**parametros)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", sugere_pca(trial)),
        ("classifier", modelo),
    ])


def cria_instancia_gaussnb(trial):
    parametros = {
        "var_smoothing": trial.suggest_float(
            "suavizacao_variancia", 1e-12, 1e-6, log=True
        )
    }
    modelo = GaussianNB(**parametros)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", sugere_pca(trial)),
        ("classifier", modelo),
    ])


INSTANCIAS = {
    "KNN": cria_instancia_knn,
    "Regressao_Logistica": cria_instancia_logreg,
    "Arvore_Decisao": cria_instancia_arvore,
    "Floresta_Aleatoria": cria_instancia_floresta,
    "Naive_Bayes_Gaussiano": cria_instancia_gaussnb,
}
SEMENTE_ALEATORIA_GLOBAL = SEMENTE_PADRAO


def carregar_dados(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {caminho.resolve()}\n"
            "Informe o caminho correto com --data."
        )

    df = pd.read_csv(
        caminho, sep=";", header=None, names=NOMES_COLUNAS
    )
    df = df.dropna(subset=NOMES_COLUNAS).copy()

    classes = set(df[ALVO].unique())
    if not classes.issubset({"RB", "NRB"}) or classes != {"RB", "NRB"}:
        raise ValueError(
            f"Esperadas as classes RB e NRB; classes encontradas: {classes}"
        )

    X = df[ATRIBUTOS].astype(float)
    y = (df[ALVO] == "RB").astype(int)  # RB=1 (positiva); NRB=0.
    # Mesmo agrupamento do notebook: descritores arredondados em uma casa.
    grupos = df[ATRIBUTOS].round(decimals=1).groupby(
        [col for col in ATRIBUTOS]
    ).ngroup()

    return df, X, y, grupos


def separar_treino_teste(X, y, grupos, n_splits, seed):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    indices_treino, indices_teste = next(
        splitter.split(X, y, groups=grupos)
    )
    return (
        X.iloc[indices_treino].to_numpy(),
        X.iloc[indices_teste].to_numpy(),
        y.iloc[indices_treino].to_numpy(),
        y.iloc[indices_teste].to_numpy(),
        grupos.iloc[indices_treino].to_numpy(),
        indices_treino,
        indices_teste,
    )


def funcao_objetivo(trial, X, y, grupos, nome_modelo, cv):
    modelo = INSTANCIAS[nome_modelo](trial)
    scores = cross_val_score(
        modelo,
        X,
        y,
        scoring="average_precision",
        cv=cv,
        groups=grupos,
        n_jobs=1,  # paralelismo fica dentro do estimador, se aplicável.
        error_score="raise",
    )
    return float(np.mean(scores))


def salvar_importancias(study, caminho_csv, nome_modelo):
    try:
        importancias = optuna.importance.get_param_importances(study)
        pd.DataFrame(
            [{"modelo": nome_modelo, "parametro": k, "importancia": v}
             for k, v in importancias.items()]
        ).to_csv(caminho_csv, index=False)
    except Exception as exc:
        print(f"[AVISO] Não foi possível calcular importâncias de "
              f"{nome_modelo}: {exc}")


def avaliar_modelo(nome, modelo, X_teste, y_teste, output_dir):
    modelo.fit(X_treino_global, y_treino_global)
    probabilidades = modelo.predict_proba(X_teste)[:, 1]
    previsoes = modelo.predict(X_teste)

    ap = average_precision_score(y_teste, probabilidades)
    matriz = confusion_matrix(y_teste, previsoes, labels=[0, 1])
    relatorio = classification_report(
        y_teste, previsoes, labels=[0, 1],
        target_names=["NRB", "RB"], output_dict=True, zero_division=0
    )

    nome_arquivo = nome.lower().replace(" ", "_")
    ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=["NRB", "RB"]
    ).plot(values_format="d")
    plt.title(f"Matriz de confusão - {nome}")
    plt.tight_layout()
    plt.savefig(output_dir / f"matriz_confusao_{nome_arquivo}.png", dpi=160)
    plt.close()

    precisao, recall, _ = precision_recall_curve(y_teste, probabilidades)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precisao, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title(f"Curva Precisão-Recall - {nome}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"curva_pr_{nome_arquivo}.png", dpi=160)
    plt.close()

    with open(output_dir / f"relatorio_{nome_arquivo}.json",
              "w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)

    return {
        "modelo": nome,
        "average_precision_teste": float(ap),
        "tn": int(matriz[0, 0]),
        "fp": int(matriz[0, 1]),
        "fn": int(matriz[1, 0]),
        "tp": int(matriz[1, 1]),
    }


def main():
    global MAX_VIZINHOS_CV, N_JOBS_MODELO
    global SEMENTE_ALEATORIA_GLOBAL, X_treino_global, y_treino_global

    args = parse_args()
    SEMENTE_ALEATORIA_GLOBAL = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    cpus_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if args.n_jobs_modelo is not None:
        N_JOBS_MODELO = args.n_jobs_modelo
    elif cpus_slurm:
        N_JOBS_MODELO = max(1, int(cpus_slurm))
    else:
        N_JOBS_MODELO = 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir = output_dir / "optuna_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    modelos_dir = output_dir / "modelos"
    modelos_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    df, X, y, grupos = carregar_dados(args.data)
    print(f"Dimensão após remoção de ausentes: {df.shape}")
    print(f"Distribuição das classes: {df[ALVO].value_counts().to_dict()}")
    print(f"Grupos únicos: {grupos.nunique()}")

    (X_treino_global, X_teste, y_treino_global, y_teste,
     grupos_treino, idx_treino, idx_teste) = separar_treino_teste(
        X, y, grupos, args.split_folds, args.seed
    )

    # Calcula o menor conjunto de treinamento da CV para restringir n_neighbors.
    cv = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed
    )
    tamanhos_treino_cv = [
        len(idx) for idx, _ in cv.split(
            X_treino_global, y_treino_global, groups=grupos_treino
        )
    ]
    MAX_VIZINHOS_CV = min(800, min(tamanhos_treino_cv))
    if MAX_VIZINHOS_CV < 1:
        raise ValueError("Não há amostras suficientes para a validação cruzada.")

    print(f"Treino: {len(y_treino_global)} | Teste: {len(y_teste)}")
    print(f"CPUs por estimador: {N_JOBS_MODELO}")
    print(f"Limite de n_neighbors: {MAX_VIZINHOS_CV}")

    # Salva os índices para que a partição seja reproduzível.
    pd.DataFrame({
        "indice_original": np.concatenate([idx_treino, idx_teste]),
        "particao": (["treino"] * len(idx_treino)) +
                    (["teste"] * len(idx_teste)),
    }).to_csv(output_dir / "indices_treino_teste.csv", index=False)

    # Baseline, usando a mesma classe positiva (RB=1).
    baseline = Pipeline([
        ("classifier", DummyClassifier(strategy="most_frequent"))
    ])
    baseline.fit(X_treino_global, y_treino_global)

    resultados_otimizacao = []
    estudos = {}
    modelos = {"Baseline": baseline}

    for nome_modelo in INSTANCIAS:
        print(f"\n{'=' * 70}\nOtimizando: {nome_modelo}\n{'=' * 70}")
        nome_estudo = f"biodegradabilidade_{nome_modelo.lower()}"
        caminho_db = (db_dir / f"{nome_estudo}.db").resolve()
        storage = f"sqlite:///{caminho_db.as_posix()}"

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            direction="maximize",
            study_name=nome_estudo,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )

        funcao = lambda trial, nm=nome_modelo: funcao_objetivo(
            trial, X_treino_global, y_treino_global,
            grupos_treino, nm, cv
        )
        study.optimize(
            funcao,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=1,  # uma trial por vez; evita contenção de recursos.
            gc_after_trial=True,
        )

        estudos[nome_modelo] = study
        melhor = study.best_trial
        print(f"Melhor AP CV: {melhor.value:.6f}")
        print(f"Melhores parâmetros: {melhor.params}")

        resultados_otimizacao.append({
            "modelo": nome_modelo,
            "melhor_ap_cv": float(melhor.value),
            "numero_trial": int(melhor.number),
            "parametros": json.dumps(melhor.params, ensure_ascii=False),
            "trials_completos": sum(
                t.state == optuna.trial.TrialState.COMPLETE
                for t in study.trials
            ),
        })

        # Reconstrói a pipeline usando exatamente os parâmetros do melhor trial.
        modelo_final = INSTANCIAS[nome_modelo](
            optuna.trial.FixedTrial(melhor.params)
        )
        modelos[nome_modelo] = modelo_final

        joblib.dump(
            modelo_final, modelos_dir / f"{nome_modelo.lower()}.joblib"
        )
        study.trials_dataframe().to_csv(
            output_dir / f"trials_{nome_modelo.lower()}.csv", index=False
        )
        salvar_importancias(
            study, output_dir / f"importancias_{nome_modelo.lower()}.csv",
            nome_modelo
        )

    df_otimizacao = pd.DataFrame(resultados_otimizacao).sort_values(
        "melhor_ap_cv", ascending=False
    )
    df_otimizacao.to_csv(
        output_dir / "resultados_otimizacao_cv.csv", index=False
    )

    # Avaliação final no conjunto de teste, somente após concluir a otimização.
    print("\nAvaliando modelos no conjunto de teste...")
    resultados_teste = []
    for nome, modelo in modelos.items():
        print(f"Avaliando: {nome}")
        resultado = avaliar_modelo(
            nome, modelo, X_teste, y_teste, output_dir
        )
        resultados_teste.append(resultado)
        joblib.dump(
            modelo, modelos_dir / f"{nome.lower().replace(' ', '_')}.joblib"
        )

    df_teste = pd.DataFrame(resultados_teste).sort_values(
        "average_precision_teste", ascending=False
    )
    df_teste.to_csv(output_dir / "resultados_teste.csv", index=False)

    resumo = {
        "arquivo_dados": str(Path(args.data).resolve()),
        "n_amostras": int(len(df)),
        "n_atributos": int(len(ATRIBUTOS)),
        "distribuicao_classes": {
            str(k): int(v) for k, v in df[ALVO].value_counts().items()
        },
        "n_grupos": int(grupos.nunique()),
        "n_treino": int(len(y_treino_global)),
        "n_teste": int(len(y_teste)),
        "seed": int(args.seed),
        "n_trials_por_modelo": int(args.n_trials),
        "n_jobs_modelo": int(N_JOBS_MODELO),
        "melhor_modelo_cv": str(df_otimizacao.iloc[0]["modelo"]),
        "melhor_ap_cv": float(df_otimizacao.iloc[0]["melhor_ap_cv"]),
        "melhor_modelo_teste": str(df_teste.iloc[0]["modelo"]),
        "melhor_ap_teste": float(df_teste.iloc[0]["average_precision_teste"]),
    }
    with open(output_dir / "resumo_execucao.json", "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    print("\nResultados da otimização (CV):")
    print(df_otimizacao.to_string(index=False))
    print("\nResultados no conjunto de teste:")
    print(df_teste.to_string(index=False))
    print(f"\nArquivos salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
