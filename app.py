# app.py

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import plotly.graph_objects as go
import plotly.io as pio
from prophet import Prophet
import warnings
import os
import torch
import numpy as np
import traceback
from chronos import ChronosPipeline
import timesfm
import ruptures as rpt
from tirex.base import load_model as load_tirex_model
from tirex.api_adapter.forecast import ForecastModel as TirexForecastModel
import ssl # Adicione este import
from datetime import datetime

# ======================================================================
# INÍCIO DO BLOCO PARA FORÇAR A IGNORAR A VERIFICAÇÃO SSL
# ======================================================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ======================================================================
# FIM DO BLOCO
# ======================================================================
# Ignora avisos comuns para uma saída mais limpa
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Habilita o CORS para permitir a comunicação com o React

# --- CONFIGURAÇÃO SEGURA DE TOKEN E AMBIENTE ---
# Carrega o token do Hugging Face a partir das variáveis de ambiente do sistema.
# ATENÇÃO: A variável de ambiente HF_TOKEN deve estar configurada no seu sistema.
if "HF_TOKEN" not in os.environ:
    raise ValueError("A variável de ambiente HF_TOKEN não foi encontrada. Configure-a antes de rodar a aplicação.")

# Seleciona o dispositivo (GPU se disponível, senão CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# CARREGAMENTO DO CHRONOS PIPELINE
print("Carregando ChronosPipeline (Small)...")
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
pipeline_chronos_small = None  # Inicia como None

try:
    pipeline_chronos_small = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        torch_dtype=torch_dtype
    )
    print("Pipeline Chronos (Small) carregado com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar ChronosPipeline: {e}")
    print("--- TRACEBACK DO ERRO DO PIPELINE CHRONOS ---")
    traceback.print_exc()
    print("-------------------------------------------")
    # pipeline_chronos_small já é None

# ---------------------------------------------------

# CARREGAMENTO DO MODELO TIREX
print("Carregando NX-AI/TiRex...")
tirex_model: TirexForecastModel = None # Inicia como None

try:
    # O TiRex requer GPU com compute capability >= 8.0, mas carregamos de qualquer forma
    # e o erro ocorrerá no uso se o hardware não for compatível.
    tirex_model = load_tirex_model("NX-AI/TiRex")
    print("Modelo TiRex carregado com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar TiRex: {e}")
    print("--- TRACEBACK DO ERRO DO TIREX ---")
    traceback.print_exc()
    print("----------------------------------")
    tirex_model = None
# ---------------------------------------------------


# CARREGAMENTO DO MODELO TIMESFM (SEGUINDO A NOVA DOCUMENTAÇÃO OFICIAL)
print("Carregando TimesFM (200m)...")
tfm = None

try:
    backend_device = "gpu" if torch.cuda.is_available() else "cpu"

    # Inicialização conforme a nova API do TimesFM
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend_device,
            # Parâmetros específicos para o modelo 1.0, conforme documentação
            context_len=512,
            horizon_len=7, # Nosso horizonte de previsão padrão
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    print("Modelo TimesFM carregado com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar TimesFM: {e}")
    print("--- TRACEBACK DO ERRO DO TIMESFM ---")
    traceback.print_exc()
    print("------------------------------------")
    tfm = None

# ---------------------------------------------------


import requests # Adicionar esta importação no início do arquivo app.py

def carregar_dados_job(nome_job, data_inicio, qtde_dias):
    """Carrega, transforma e processa os dados de um job específico a partir da API."""
    # Parâmetros da URL agora são recebidos dinamicamente.
    base_url = f"http://10.1.136.123:2601/nupro/ibm/api/v1/brb12/joblog/{nome_job}"
    params = {
        'f': 'json',
        'data_inicio': data_inicio,
        'qtde_dias': qtde_dias
    }

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()  # Lança exceção para códigos de erro HTTP (4xx ou 5xx)
        api_data = response.json()

        job_records = api_data.get("recordset", {}).get(nome_job.upper(), {})
        if not job_records:
            print(f"Nenhum registro encontrado para o job '{nome_job.upper()}' na resposta da API.")
            return None

        # Transforma os dados da API para o formato legado esperado
        transformed_data = []
        for job_id, details in job_records.items():
            transformed_data.append({
                "nome_job": details.get("jobname"),
                "data_execucao": details.get("dataExecucao"),
                "hora_execucao": details.get("horaExecucao", "").replace(":", "."),
                "tempo_cpu": str(details.get("tempoCPU", "0")).replace(".", ","),
                "tempo_sala": str(details.get("tempoSala", "0")).replace(".", ","),
                "return_code": float(details.get("RC", "0")),
                "usuario": details.get("usuario"),
                "jobid": float(details.get("jobID", "0"))
            })
        
        if not transformed_data:
             return None

        df = pd.DataFrame(transformed_data)

        # O restante do processamento permanece o mesmo
        df['TEMPO DE CPU'] = pd.to_numeric(df['tempo_cpu'].str.replace(',', '.'), errors='coerce')
        df['TEMPO DE SALA'] = pd.to_numeric(df['tempo_sala'].str.replace(',', '.'), errors='coerce')
        df.dropna(subset=['TEMPO DE CPU', 'TEMPO DE SALA'], inplace=True)
        
        df['TIMESTAMP'] = pd.to_datetime(
            df['data_execucao'] + ' ' + df['hora_execucao'], format='%d/%m/%Y %H.%M.%S', errors='coerce'
        )
        df.dropna(subset=['TIMESTAMP'], inplace=True)

        hora_parts = df['hora_execucao'].str.split('.', expand=True)
        df['MINUTOS_EXECUCAO'] = hora_parts[0].astype(int) * 60 + hora_parts[1].astype(int) + hora_parts[2].astype(int) / 60
        
        df = df.sort_values(by='TIMESTAMP').reset_index(drop=True)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Erro de rede ao buscar dados para o job '{nome_job}': {e}")
        return None
    except (ValueError, KeyError) as e:
        print(f"Erro ao processar o JSON da API para o job '{nome_job}': {e}")
        return None


def remover_outliers_inteligente(df, coluna, tipo):
    """Remove outliers de forma inteligente baseada no tipo de dado."""
    if len(df) < 10:  # Não remove outliers se tiver poucos dados
        return df

    # Usa IQR (Interquartile Range) para remoção mais eficaz de outliers
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1

    # Para outros tipos, usa lógica padrão
    multiplicador = 1.5

    limite_inferior = Q1 - multiplicador * IQR
    limite_superior = Q3 + multiplicador * IQR

    df_filtrado = df[
        (df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)
    ].copy()

    outliers_removidos = len(df) - len(df_filtrado)
    if outliers_removidos > 0:
        print(
            f"Para '{coluna}' ({tipo}), foram removidos {outliers_removidos} outliers do treinamento.")

    return df_filtrado

def detectar_changepoint_e_filtrar(df, coluna):
    """
    Detecta o ponto de mudança mais recente na série temporal e, se for significativo,
    filtra o DataFrame para usar apenas os dados após essa mudança.
    """
    # Só executa se tivermos um volume mínimo de dados para a análise fazer sentido
    if len(df) < 30:
        return df

    # Converte os dados da coluna para um array numpy, que a biblioteca utiliza
    pontos = df[coluna].values
    
    # A penalidade (pen) controla a sensibilidade. Valores maiores = menos mudanças detectadas.
    # pen=10 é um valor empírico que funciona bem para detectar mudanças significativas sem ser muito sensível a ruídos.
    algo = rpt.Pelt(model="l2").fit(pontos)
    pontos_de_mudanca = algo.predict(pen=10)

    # O resultado inclui o final da série, então o removemos se ele estiver lá.
    # Ex: [50, 134] significa que a primeira série vai de 0-49 e a nova começa em 50.
    if len(pontos_de_mudanca) > 1:
        # Pega o início do último segmento (o ponto de mudança mais recente)
        ultimo_ponto_de_mudanca = pontos_de_mudanca[-2]
        tamanho_ultimo_segmento = len(df) - ultimo_ponto_de_mudanca

        print(f"Ponto de mudança detectado no índice {ultimo_ponto_de_mudanca}. "
              f"O novo segmento tem {tamanho_ultimo_segmento} pontos.")

        # Define um tamanho mínimo para o novo segmento ser considerado válido para treinamento
        MIN_SEGMENTO_PARA_TREINO = 5
        if tamanho_ultimo_segmento >= MIN_SEGMENTO_PARA_TREINO:
            print(f"O último segmento é grande o suficiente. "
                  f"O treinamento usará apenas os dados a partir de {df['TIMESTAMP'].iloc[ultimo_ponto_de_mudanca].date()}.")
            return df.iloc[ultimo_ponto_de_mudanca:].reset_index(drop=True)
        else:
            print("O último segmento é muito curto. Usando o histórico completo para a análise.")
            
    return df

def configurar_prophet_por_tipo(tipo):
    """Configura os parâmetros do Prophet baseado no tipo de dado."""
    # Para CPU, Sala e Execução: growth linear livre, foco na tendência
    return {
            'growth': 'linear',
            'changepoint_range': 0.95,
            'changepoint_prior_scale': 0.5,
            'seasonality_prior_scale': 3.0,  # Sazonalidade moderada
            'weekly_seasonality': True,
            'interval_width': 0.95,
            'yearly_seasonality': False,
            'daily_seasonality': False
    }


def obter_ajuste_sazonal_prophet(df_treino_sazonal, coluna_y, prediction_length):
    """
    Usa o Prophet para modelar APENAS a sazonalidade MENSAL automática
    e retorna o valor de ajuste para os dias futuros.
    """
    # Só executa se tivermos dados suficientes para o Prophet aprender o padrão
    if len(df_treino_sazonal) < 60: # Pelo menos 2 meses de dados
        print("Dados insuficientes para análise sazonal mensal. Nenhum ajuste será feito.")
        return np.zeros(prediction_length)

    print("Calculando ajuste sazonal MENSAL com Prophet...")
    
    # Prepara o dataframe para o Prophet
    df_prophet = df_treino_sazonal[['TIMESTAMP', coluna_y]].rename(
        columns={'TIMESTAMP': 'ds', coluna_y: 'y'})

    # --- Configuração do Prophet para atuar como ESPECIALISTA SAZONAL MENSAL ---
    # Desligamos a tendência e outras sazonalidades para focar apenas na mensal.
    modelo_sazonal = Prophet(
        growth='flat',                  # Sem tendência de crescimento
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    # Adicionamos a sazonalidade mensal manualmente, o que é mais compatível
    modelo_sazonal.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    modelo_sazonal.fit(df_prophet)

    # Cria o dataframe futuro
    futuro = modelo_sazonal.make_future_dataframe(periods=prediction_length)
    
    # Faz a previsão dos componentes
    previsao_componentes = modelo_sazonal.predict(futuro)

    # Retorna APENAS o valor do componente 'monthly' para os dias futuros
    ajuste = previsao_componentes.tail(prediction_length)['monthly'].values
    print(f"Ajuste sazonal MENSAL para os próximos {prediction_length} dias: {ajuste}")
    return ajuste

# --- NOVA FUNÇÃO AUXILIAR (VERSÃO PIPELINE) ---

def gerar_previsao_chronos_small(contexto_historico, prediction_length=7, num_samples=20):
    """
    Gera previsões probabilísticas usando o ChronosPipeline.
    O pipeline retorna diretamente tensores numéricos (não tokens de texto).
    """
    # Garante que o pipeline tenha carregado. Se não, retorna NaNs
    if pipeline_chronos_small is None:
        print("ERRO: O Pipeline Chronos não foi carregado. Retornando previsão vazia.")
        nan_array = [np.nan] * prediction_length
        return nan_array, nan_array, nan_array

    # 1. Preparar o contexto: O pipeline espera um tensor PyTorch.
    # Nossa 'contexto_historico' é uma lista Python.
    context_tensor = torch.tensor(
        contexto_historico, dtype=torch.float32).to(device)

    # 2. Gerar previsão. O pipeline faz a amostragem (num_samples) internamente.
    # O output tem shape [batch_size (1), num_samples, prediction_length]
    forecast = pipeline_chronos_small.predict(
        context_tensor.unsqueeze(
            0),  # Adiciona a dimensão de batch (shape [1, context_length])
        prediction_length,
        num_samples=num_samples
    )

    # 3. Processar o output. O output JÁ É NUMÉRICO.
    # Pegamos o primeiro (e único) item do batch: forecast[0]
    # Isso nos dá um array de shape [num_samples, prediction_length]
    # Movemos para CPU (se estava na GPU) e convertemos para Numpy para calcular quantis
    samples_array = forecast[0].cpu().numpy()

    # 4. Calcular os quantis (Intervalo de 90% = 0.05 e 0.95. Mediana = 0.5)
    # axis=0 significa que calculamos os quantis "verticalmente" (através de todas as amostras)
    # para cada um dos 7 dias previstos.
    yhat_forecast = np.quantile(samples_array, 0.5, axis=0)
    yhat_lower_forecast = np.quantile(samples_array, 0.05, axis=0)
    yhat_upper_forecast = np.quantile(samples_array, 0.95, axis=0)

    return yhat_forecast, yhat_lower_forecast, yhat_upper_forecast

# --- FIM DA NOVA FUNÇÃO ---

# --- NOVA FUNÇÃO AUXILIAR PARA PREVISÃO COM TIMESFM ---
def gerar_previsao_timesfm(contexto_historico, prediction_length=7):
    """
    Gera previsões usando o modelo TimesFM.
    Retorna a previsão pontual (mediana) e um intervalo de confiança de 80%.
    """
    # Garante que o modelo TimesFM tenha carregado.
    if tfm is None:
        print("ERRO: O modelo TimesFM não foi carregado. Retornando previsão vazia.")
        nan_array = np.full(prediction_length, np.nan)
        return nan_array, nan_array, nan_array

    # 1. Preparar o contexto: TimesFM espera uma lista de arrays numpy.
    contexto_array = np.array(contexto_historico)
    forecast_input = [contexto_array]

    # 2. Definir a frequência. Como são dados de execuções diárias, usamos 0.
    frequency_input = [0]
    
    # 3. Gerar previsão.
    # CORREÇÃO: Ajustamos o 'horizon_len' do objeto de hiperparâmetros (hparams)
    # antes de chamar a previsão. Removemos o 'forecast_len'.
    tfm.hparams.horizon_len = prediction_length
    point_forecast, experimental_quantile_forecast = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    )

    # 4. Processar o output.
    # Os quantis padrão do TimesFM são [0.1, 0.2, ..., 0.9].
    # Usaremos q=0.5 (índice 4) para a mediana (yhat).
    # Usaremos q=0.1 (índice 0) e q=0.9 (índice 8) para o intervalo de confiança.
    # Isso resulta em um intervalo de confiança de 80%.
    quantiles_all = experimental_quantile_forecast[0]
    # O output dos quantis é um array. Para um intervalo de 80%, usamos os índices:
    # Índice 0: Quantil 0.1 (limite inferior)
    # Índice 4: Quantil 0.5 (mediana, nossa previsão principal)
    # Índice 8: Quantil 0.9 (limite superior)
    yhat_forecast = quantiles_all[:, 4]        # Mediana (Quantil 0.5)
    yhat_lower_forecast = quantiles_all[:, 0]  # Limite Inferior (Quantil 0.1)
    yhat_upper_forecast = quantiles_all[:, 8]  # Limite Superior (Quantil 0.9)

    return yhat_forecast, yhat_lower_forecast, yhat_upper_forecast
# --- FIM DA NOVA FUNÇÃO ---

# --- NOVA FUNÇÃO AUXILIAR PARA PREVISÃO COM TIREX ---
def gerar_previsao_tirex(contexto_historico, prediction_length=7, num_samples=100):
    """
    Gera previsões probabilísticas usando o modelo NX-AI/TiRex.
    """
    # Garante que o modelo TiRex tenha carregado.
    if tirex_model is None:
        print("ERRO: O modelo TiRex não foi carregado. Retornando previsão vazia.")
        nan_array = np.full(prediction_length, np.nan)
        return nan_array, nan_array, nan_array
        
    try:
        # 1. Preparar o contexto: TiRex espera um tensor PyTorch com dimensão de batch.
        # [batch_size, context_length]
        context_tensor = torch.tensor(
            contexto_historico, dtype=torch.float32).unsqueeze(0)

        # 2. Gerar previsão. O modelo retorna os quantis e a média diretamente.
        quantiles, mean = tirex_model.forecast(
            context=context_tensor,
            prediction_length=prediction_length
        )

        # 3. Processar o output. A saída 'quantiles' tem shape [batch_size (1), prediction_length, num_quantiles]
        # Pegamos o primeiro item do batch e convertemos para Numpy.
        quantiles_array = quantiles[0].cpu().numpy()

        # 4. Selecionar os quantis para o intervalo de confiança.
        # Assim como o TimesFM, vamos assumir que ele retorna 9 quantis (0.1, 0.2 ... 0.9)
        # Usaremos o quantil 0.5 (índice 4) para a mediana e 0.1/0.9 para o intervalo (80% de confiança).
        yhat_forecast = quantiles_array[:, 4]       # Mediana (Quantil 0.5)
        yhat_lower_forecast = quantiles_array[:, 0] # Limite Inferior (Quantil 0.1)
        yhat_upper_forecast = quantiles_array[:, 8] # Limite Superior (Quantil 0.9)

        return yhat_forecast, yhat_lower_forecast, yhat_upper_forecast

    except Exception as e:
        print(f"ERRO CRÍTICO durante a previsão com TiRex: {e}")
        print("--- TRACEBACK DO ERRO DO TIREX FORECAST ---")
        traceback.print_exc()
        print("-----------------------------------------")
        nan_array = np.full(prediction_length, np.nan)
        return nan_array, nan_array, nan_array
# --- FIM DA NOVA FUNÇÃO ---

def aplicar_filtros_avancados(df, args):
    """Aplica filtros dinâmicos a um DataFrame com base nos query parameters da requisição."""
    if df.empty:
        return df

    # Mapeia o nome do filtro na URL para o nome da coluna no DataFrame
    mapa_filtros = {
        'hora_execucao': 'MINUTOS_EXECUCAO',
        'tempo_cpu': 'TEMPO DE CPU',
        'tempo_sala': 'TEMPO DE SALA',
        'return_code': 'return_code',
        'usuario': 'usuario'
    }

    df_filtrado = df.copy()

    for key, value in args.items():
        if not key.startswith('filter_'):
            continue

        try:
            # Extrai o campo, operador e valor(es) do parâmetro
            campo = key.split('_', 1)[1]
            operador, valor_str = value.split(':', 1)
            
            if campo not in mapa_filtros:
                continue
            
            coluna = mapa_filtros[campo]

            # Converte os valores para os tipos corretos
            if campo == 'usuario':
                valor1 = valor_str
            elif campo == 'hora_execucao':
                # Converte HH:MM:SS para minutos totais
                parts = valor_str.split(',')
                h, m, s = map(int, parts[0].split(':'))
                valor1 = h * 60 + m + s / 60
                if operador == 'between' and len(parts) > 1:
                    h, m, s = map(int, parts[1].split(':'))
                    valor2 = h * 60 + m + s / 60
            else: # Campos numéricos (cpu, sala, rc)
                parts = valor_str.split(',')
                valor1 = float(parts[0])
                if operador == 'between' and len(parts) > 1:
                    valor2 = float(parts[1])

            # Aplica o filtro correspondente ao operador
            if operador == 'eq':
                df_filtrado = df_filtrado[df_filtrado[coluna] == valor1]
            elif operador == 'ne':
                df_filtrado = df_filtrado[df_filtrado[coluna] != valor1]
            elif operador == 'lt' and campo != 'usuario':
                df_filtrado = df_filtrado[df_filtrado[coluna] < valor1]
            elif operador == 'le' and campo != 'usuario':
                df_filtrado = df_filtrado[df_filtrado[coluna] <= valor1]
            elif operador == 'gt' and campo != 'usuario':
                df_filtrado = df_filtrado[df_filtrado[coluna] > valor1]
            elif operador == 'ge' and campo != 'usuario':
                df_filtrado = df_filtrado[df_filtrado[coluna] >= valor1]
            elif operador == 'between' and campo != 'usuario':
                df_filtrado = df_filtrado[df_filtrado[coluna].between(valor1, valor2)]

        except (ValueError, IndexError) as e:
            print(f"AVISO: Ignorando filtro malformado '{key}={value}'. Erro: {e}")
            continue
            
    return df_filtrado

@app.route("/api/grafico/<tipo>/<nome_job>")
def gerar_grafico_interativo(tipo, nome_job):
    """
    Rota principal que gera e retorna o JSON de um gráfico interativo com previsão.
    Aceita 'data_inicio' e 'data_fim' (DD/MM/YYYY) como query parameters.
    """
    data_inicio_str = request.args.get('data_inicio')
    data_fim_str = request.args.get('data_fim')

    # Validação de parâmetros de entrada
    if not data_inicio_str or not data_fim_str:
        return jsonify({"erro": "Os parâmetros 'data_inicio' e 'data_fim' são obrigatórios."}), 400

    try:
        # Converte as strings para objetos datetime
        start_date = datetime.strptime(data_inicio_str, '%d/%m/%Y')
        end_date = datetime.strptime(data_fim_str, '%d/%m/%Y')

        # Garante que a data final não seja anterior à inicial
        if end_date < start_date:
            return jsonify({"erro": "A 'data_fim' não pode ser anterior à 'data_inicio'."}), 400

        # Calcula a diferença de dias (inclusivo)
        delta = end_date - start_date
        qtde_dias = delta.days + 1

        # Impõe um limite máximo para evitar sobrecarga
        qtde_dias = min(qtde_dias, 365)

    except ValueError:
        return jsonify({"erro": "Formato de data inválido. Use DD/MM/YYYY."}), 400

    df_bruto = carregar_dados_job(nome_job, data_inicio_str, qtde_dias)

    if df_bruto is None:
        return jsonify({"erro": f"Job '{nome_job}' não encontrado para o período selecionado."}), 404
    
    # --- APLICAÇÃO DOS FILTROS AVANÇADOS ---
    df = aplicar_filtros_avancados(df_bruto, request.args)
    # ----------------------------------------

    if df.empty or len(df) < 5:
        return jsonify({"erro": f"Dados insuficientes para o job '{nome_job}' após a aplicação dos filtros."}), 404

    # Define qual coluna usar com base no tipo de gráfico solicitado
    mapa_colunas = {
        'cpu': ('TEMPO DE CPU', 'Tempo de CPU (min)', 'royalblue'),
        'sala': ('TEMPO DE SALA', 'Tempo de Sala (min)', 'darkorange'),
        'execucao': ('MINUTOS_EXECUCAO', 'Minutos desde 00:00 (min)', 'purple')
    }

    if tipo not in mapa_colunas:
        return jsonify({"erro": "Tipo de gráfico inválido."}), 400

    coluna_y, label_y, cor_principal = mapa_colunas[tipo]
    
    # --- INÍCIO DA ADIÇÃO ---
    # Define o Título e a Configuração do Eixo Y com base no tipo
    if tipo == 'execucao':
        title_grafico = f'Análise de Hora de Execução do Job: {nome_job.upper()}'
        yaxis_config = dict(
            range=[0, 1440],  # 24 horas * 60 minutos
            tickmode='array',
            tickvals=[h * 60 for h in range(0, 25, 4)],  # Marcadores a cada 4 horas
            ticktext=[f'{h:02d}:00' for h in range(0, 25, 4)],
            title='Horário da Execução'  # Título do EIXO Y
        )
    else:  # Padrão para 'cpu' e 'sala'
        title_grafico = f'Análise de {label_y.replace("(min)", "").strip()} do Job: {nome_job.upper()}'
        yaxis_config = dict(rangemode='tozero')
    # --- FIM DA ADIÇÃO ---

    # Detecta o "novo normal" antes de qualquer outra análise
    df_segmento = detectar_changepoint_e_filtrar(df, coluna_y)
    # Remove timestamps duplicados para evitar erros no pandas
    df_segmento.drop_duplicates(subset=['TIMESTAMP'], inplace=True)
    # Remove outliers de forma inteligente, AGORA USANDO O DATAFRAME CORRETO (df_segmento)
    df_treino = remover_outliers_inteligente(df_segmento, coluna_y, tipo)

    # --- LÓGICA DE PREVISÃO (MODELO SELETIVO) ---
    num_dados_treino = len(df_treino)

    # Lógica de horizonte de previsão dinâmico
    if num_dados_treino <= 100:
        # Para poucos dados, o horizonte é metade do histórico (mínimo 7, máximo 15 dias)
        horizonte_proporcional = int(num_dados_treino / 2)
        prediction_length = max(7, min(horizonte_proporcional, 15))
        print(f"Horizonte de previsão dinâmico para poucos dados ({num_dados_treino}): {prediction_length} dias.")
    else:
        # Para dados médios e grandes, usamos o horizonte completo de 30 dias
        prediction_length = 30
    
    previsao_gerada = False  # Flag para controlar se a previsão de IA já foi feita

    # Variáveis para armazenar o nome do modelo e a quantidade de dados
    nome_modelo_utilizado = "Prophet (Fallback)"
    quantidade_dados_modelo = num_dados_treino
    confianca_intervalo = 0.95 # Padrão para Prophet

    # --- ROTEAMENTO DE MODELO PARA CPU E SALA ---
    if tipo in ['cpu', 'sala', 'execucao']:
        modelo_usado = None # Esta variável não é mais usada, mas pode ser mantida ou removida
        yhat_futuro, yhat_lower_futuro, yhat_upper_futuro = None, None, None
        contexto_historico = df_treino[coluna_y].tolist()

        # --- ETAPA 1: CALCULAR AJUSTE SAZONAL MENSAL COM PROPHET ---
        # Usamos o df_segmento para dar ao Prophet o máximo de histórico possível
        # para aprender o padrão mensal, mesmo que o modelo principal use menos dados.
        ajuste_sazonal = obter_ajuste_sazonal_prophet(
            df_segmento, coluna_y, prediction_length
        )
        # ----------------------------------------------------------------

        # ETAPA 2: PREVISÃO PRINCIPAL COM CHRONOS/TIREX/TIMESFM
        if num_dados_treino <= 100: # POUCOS DADOS
            if pipeline_chronos_small is not None:
                nome_modelo_utilizado = "Chronos-T5-Small"
                quantidade_dados_modelo = num_dados_treino
                (yhat_futuro, yhat_lower_futuro, yhat_upper_futuro) = gerar_previsao_chronos_small(
                    contexto_historico, prediction_length=prediction_length
                )
                confianca_intervalo = 0.90 # Chronos retorna quantis, usamos 90%
            else:
                print(f"AVISO: Chronos solicitado para {tipo}, mas não está carregado.")
        
        elif 100 < num_dados_treino <= 200: # MÉDIOS DADOS (NOVO BLOCO)
            if tirex_model is not None:
                modelo_usado = f"NX-AI/TiRex (Médios Dados: {num_dados_treino})"
                (yhat_futuro, yhat_lower_futuro, yhat_upper_futuro) = gerar_previsao_tirex(
                    contexto_historico, prediction_length=prediction_length
                )
                confianca_intervalo = 0.80 # TiRex retorna quantis, usamos 90%
            else:
                print(f"AVISO: TiRex solicitado para {tipo}, mas não está carregado.")

        else: # GRANDES DADOS (ACIMA DE 200) - Mantendo TimesFM por enquanto
            if tfm is not None:
                # O modelo Moirai será implementado aqui posteriormente. Por enquanto, usamos TimesFM.
                modelo_usado = f"TimesFM-200M (Grandes Dados: {num_dados_treino} > 200)"
                (yhat_futuro, yhat_lower_futuro, yhat_upper_futuro) = gerar_previsao_timesfm(
                    contexto_historico, prediction_length=prediction_length
                )
                confianca_intervalo = 0.80
            else:
                print(f"AVISO: TimesFM solicitado para {tipo}, mas não está carregado.")

        # --- ETAPA 3: COMBINAR AS PREVISÕES ---
        # Se a previsão principal foi gerada com sucesso, somamos o ajuste sazonal
        if yhat_futuro is not None:
            print(f"Executando análise de '{tipo}' com {modelo_usado}")
            yhat_futuro += ajuste_sazonal
            yhat_lower_futuro += ajuste_sazonal
            yhat_upper_futuro += ajuste_sazonal
        # -----------------------------------------------

            # --- Lógica de construção do dataframe 'previsao' ---
            # O dataframe 'previsao' deve conter o histórico completo para a linha verde
            # E os valores futuros previstos.

            # Passo 1: Começamos com o histórico COMPLETO, mas sem valores de previsão.
            # Usamos o dataframe original (df_segmento) ANTES da remoção de outliers para pegar todas as datas.
            previsao = df_segmento[['TIMESTAMP']].copy()
            previsao.rename(columns={'TIMESTAMP': 'ds'}, inplace=True)
            
            # Adicionamos colunas de previsão, inicialmente vazias (NaN)
            previsao['yhat'] = np.nan
            previsao['yhat_lower'] = np.nan
            previsao['yhat_upper'] = np.nan

            # Passo 2: Preenchemos os valores 'yhat' para o período de TREINO
            # O 'yhat' do período histórico é simplesmente o valor real que o modelo viu.
            # Usamos 'df_treino' para isso.
            previsao.set_index('ds', inplace=True)
            df_treino_com_indice = df_treino.set_index('TIMESTAMP')
            previsao.update(df_treino_com_indice.rename(columns={coluna_y: 'yhat'}))
            previsao.reset_index(inplace=True)

            # Preenchemos também os intervalos de confiança para o período de treino
            previsao.loc[previsao['ds'].isin(df_treino['TIMESTAMP']), 'yhat_lower'] = previsao['yhat']
            previsao.loc[previsao['ds'].isin(df_treino['TIMESTAMP']), 'yhat_upper'] = previsao['yhat']
            
            # Passo 3: Criar e adicionar os dados da PREVISÃO FUTURA
            ultimo_timestamp_real = df_treino['TIMESTAMP'].max()
            datas_futuras = pd.date_range(
                start=ultimo_timestamp_real + pd.Timedelta(days=1),
                periods=prediction_length, freq='D'
            )
            df_previsao_futuro = pd.DataFrame({
                'ds': datas_futuras, 'yhat': yhat_futuro,
                'yhat_lower': yhat_lower_futuro, 'yhat_upper': yhat_upper_futuro,
            })
            
            # Passo 4: Combinar o histórico preenchido com o futuro
            previsao = pd.concat([previsao, df_previsao_futuro], ignore_index=True)
            previsao = previsao.dropna(subset=['yhat']).reset_index(drop=True) # Remove linhas sem previsão

            # Pós-processamento para garantir que não haja valores negativos
            minimo_real = df_treino.loc[df_treino[coluna_y] > 0, coluna_y].min()
            piso = max(minimo_real * 0.9, 0.01) if not pd.isna(
                minimo_real) else 0.01
            previsao['yhat'] = previsao['yhat'].clip(lower=piso)
            previsao['yhat_lower'] = previsao['yhat_lower'].clip(lower=piso)
            previsao['yhat_upper'] = previsao['yhat_upper'].clip(lower=piso)

            # Define a configuração de confiança para o hover do gráfico
            config_prophet = {'interval_width': confianca_intervalo}
            previsao_gerada = True

    # --- CAMINHO PADRÃO/FALLBACK: PROPHET ---
    # Será executado se:
    # 1. O tipo de gráfico NÃO for 'cpu'.
    # 2. O tipo for 'cpu', mas o modelo de IA correspondente (Chronos/TimesFM) falhou.
    if not previsao_gerada:
        if tipo == 'cpu':
            print(
                f"AVISO: Análise de CPU com {num_dados_treino} dados revertendo para Prophet (modelo de IA não disponível).")
        else:
            print(
                f"Executando análise de {tipo} com Prophet (Padrão) (Dados: {num_dados_treino})")
            
        # Prepara os dados para o formato do Prophet
        df_prophet = df_treino[['TIMESTAMP', coluna_y]].rename(
            columns={'TIMESTAMP': 'ds', coluna_y: 'y'})

        # Configura o Prophet baseado no tipo
        config_prophet = configurar_prophet_por_tipo(tipo)
        modelo = Prophet(**config_prophet)

        if tipo in ['cpu', 'sala', 'execucao']:
            modelo.add_seasonality(
                name='custom_weekly', period=7, fourier_order=3, prior_scale=0.1)

        modelo.fit(df_prophet)

        # Cria o dataframe futuro
        futuro = modelo.make_future_dataframe(
            periods=prediction_length, include_history=True)
        if tipo == 'horario':
            futuro['floor'] = 0
            futuro['cap'] = 24

        previsao = modelo.predict(futuro)

        # Pós-processamento para garantir valores não-negativos
        if tipo in ['cpu', 'sala', 'execucao']:
            minimo_real = df_prophet.loc[df_prophet['y'] > 0, 'y'].min()
            piso = max(minimo_real * 0.9, 0.01) if not pd.isna(
                minimo_real) else 0.01

            previsao['yhat'] = previsao['yhat'].clip(lower=piso)
            previsao['yhat_lower'] = previsao['yhat_lower'].clip(lower=piso)
            previsao['yhat_upper'] = previsao['yhat_upper'].clip(lower=piso)

    # --- FIM DA LÓGICA DE PREVISÃO ---

    # Montagem do Gráfico Interativo com Plotly Graph Objects
    fig = go.Figure()

    # --- CAMADA 1: ÁREA DE INCERTEZA ---
    fig.add_trace(go.Scatter(
        x=previsao['ds'].tolist() + previsao['ds'].tolist()[::-1],
        y=previsao['yhat_upper'].tolist() + previsao['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="none",
        name=f'Intervalo de Previsão'
    ))

    # --- CAMADA 2: LINHA DE PREVISÃO PRINCIPAL ---
    fig.add_trace(go.Scatter(
        x=previsao['ds'],
        y=previsao['yhat'],
        mode='lines',
        line=dict(color='rgba(0,100,80,0.8)', width=3),
        name='Previsão'
    ))

    # --- CAMADA 3: PONTOS DE DADOS (REAIS E IGNORADOS) ---

    # PRIMEIRO: Gera o texto do hover para TODOS os pontos do dataframe original
    hover_texts_full = []
    for index, row in df.iterrows():
        if tipo == 'cpu':
            hover_text = (
                f"<b>Job:</b> {row['nome_job']}<br>"
                f"<b>Data:</b> {row['data_execucao']}<br>"
                f"<b>Tempo CPU:</b> {row['tempo_cpu']}<br>"
                f"<b>Return Code:</b> {row['return_code']}<br>"
                f"<b>Usuário:</b> {row['usuario']}<br>"
                f"<b>Job ID:</b> {row['jobid']}"
            )
        elif tipo == 'sala':
            hover_text = (
                f"<b>Job:</b> {row['nome_job']}<br>"
                f"<b>Data:</b> {row['data_execucao']}<br>"
                f"<b>Tempo Sala:</b> {row['tempo_sala']}<br>"
                f"<b>Return Code:</b> {row['return_code']}<br>"
                f"<b>Usuário:</b> {row['usuario']}<br>"
                f"<b>Job ID:</b> {row['jobid']}"
            )
        elif tipo == 'execucao':
            hover_text = (
                f"<b>Job:</b> {row['nome_job']}<br>"
                f"<b>Data:</b> {row['data_execucao']}<br>"
                f"<b>Hora Execução:</b> {row['hora_execucao']}<br>"
                f"<b>Return Code:</b> {row['return_code']}<br>"
                f"<b>Usuário:</b> {row['usuario']}<br>"
                f"<b>Job ID:</b> {row['jobid']}"
            )
        hover_texts_full.append(hover_text)
    
    # Adiciona a lista de textos como uma nova coluna para facilitar a filtragem
    df['hover_text'] = hover_texts_full

    # SEGUNDO: Plota os pontos IGNORADOS (transparentes) com seu respectivo hover text
    df_ignorado = df[~df['TIMESTAMP'].isin(df_treino['TIMESTAMP'])].copy()
    if not df_ignorado.empty:
        fig.add_trace(go.Scatter(
            x=df_ignorado['TIMESTAMP'],
            y=df_ignorado[coluna_y],
            mode='markers',
            marker=dict(
                color=cor_principal,
                opacity=0.3,
                size=8,
                line=dict(width=1, color='white')
            ),
            name='Execuções Reais Ignoradas',
            hovertemplate='%{text}<extra></extra>',
            text=df_ignorado['hover_text']
        ))
    
    # CORREÇÃO: Adiciona a coluna 'hover_text' ao df_treino antes de usá-la
    df_treino = df[df['TIMESTAMP'].isin(df_treino['TIMESTAMP'])].copy()

    # TERCEIRO: Plota os pontos REAIS USADOS NO TREINO (sólidos) com seu hover text
    fig.add_trace(go.Scatter(
        x=df_treino['TIMESTAMP'],
        y=df_treino[coluna_y],
        mode='markers',
        marker=dict(color=cor_principal, size=8,
                    line=dict(width=1, color='white')),
        name='Execuções Reais (Treino)',
        hovertemplate='%{text}<extra></extra>',
        text=df_treino['hover_text']
    ))

    # Camada 4: Pontos de Previsão Futura
    # Identifica os dados futuros (após o último timestamp real)
    ultimo_timestamp_real = df['TIMESTAMP'].max()
    previsao_futura = previsao[previsao['ds'] > ultimo_timestamp_real]

    # Cria textos de hover para previsão futura
    hover_texts_previsao = []
    for index, row in previsao_futura.iterrows():
        data_formatada = row['ds'].strftime('%d/%m/%Y')
        # Novo: Informação do modelo e quantidade de dados
        info_modelo = f"<b>Modelo Utilizado:</b> {nome_modelo_utilizado} ({quantidade_dados_modelo} dados)"

        if tipo == 'cpu':
            valor_formatado = f"{row['yhat']:.2f}"
            hover_text_prev = (
                f"<b>PREVISÃO</b><br>"
                f"<b>Job:</b> {nome_job}<br>"
                f"<b>Data:</b> {data_formatada}<br>"
                f"<b>Tempo CPU Previsto:</b> {valor_formatado}<br>"
                f"<b>Intervalo de Previsão:</b> {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f}<br>"
                f"{info_modelo}"
            )
        elif tipo == 'sala':
            valor_formatado = f"{row['yhat']:.2f}"
            hover_text_prev = (
                f"<b>PREVISÃO</b><br>"
                f"<b>Job:</b> {nome_job}<br>"
                f"<b>Data:</b> {data_formatada}<br>"
                f"<b>Tempo Sala Previsto:</b> {valor_formatado}<br>"
                f"<b>Intervalo de Previsão:</b> {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f}<br>"
                f"{info_modelo}"
            )
        elif tipo == 'execucao':
            # Função auxiliar para formatar minutos em HH:MM
            def formatar_minutos_para_horario(minutos_float):
                if minutos_float < 0:
                    minutos_float = 0
                horas = int(minutos_float // 60) % 24
                minutos = int(minutos_float % 60)
                return f"{horas:02d}:{minutos:02d}"

            valor_formatado = formatar_minutos_para_horario(row['yhat'])
            intervalo_inicio = formatar_minutos_para_horario(
                row['yhat_lower'])
            intervalo_fim = formatar_minutos_para_horario(row['yhat_upper'])

            hover_text_prev = (
                f"<b>PREVISÃO</b><br>"
                f"<b>Job:</b> {nome_job}<br>"
                f"<b>Data:</b> {data_formatada}<br>"
                f"<b>Hora de Execução Prevista:</b> {valor_formatado}<br>"
                f"<b>Intervalo de Previsão:</b> {intervalo_inicio} - {intervalo_fim}<br>"
                f"{info_modelo}"
            )
        
        hover_texts_previsao.append(hover_text_prev)

    # Adiciona os pontos de previsão futura apenas se existirem
    if len(previsao_futura) > 0:
        fig.add_trace(go.Scatter(
            x=previsao_futura['ds'],
            y=previsao_futura['yhat'],
            mode='markers',
            marker=dict(
                color='rgba(255,99,71,0.7)',  # Vermelho tomate semi-transparente
                size=8,
                line=dict(width=2, color='white'),
                symbol='circle'  # Bolinha normal
            ),
            name='Previsões Futuras',
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts_previsao
        ))

    # --- Customização Final do Layout ---
    # DEBUG: Imprime o valor final do título no console para depuração.
    # Por favor, verifique o terminal onde o Flask está rodando ao carregar o gráfico.
    print("--- INFORMAÇÃO DE DEBUG ---")
    print(f"Tipo de Gráfico Solicitado: '{tipo}'")
    print(f"Título Final a ser Aplicado ao Gráfico: '{title_grafico}'")
    print("--------------------------")

    fig.update_layout(
        title=dict(
            text=title_grafico,
            y=0.98,
            x=0.05,          # Posição X movida para a ESQUERDA
            xanchor='left',  # Âncora do título na ESQUERDA
            yanchor='top'
        ),
        xaxis_title='Data da Execução',
        yaxis_title=label_y,
        yaxis=yaxis_config,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0.95,          # Posição X movida para a DIREITA
            xanchor="right"  # Âncora da legenda na DIREITA
        ),
        hovermode='closest'
    )


    # Converte a figura para JSON e a retorna
    return pio.to_json(fig)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
