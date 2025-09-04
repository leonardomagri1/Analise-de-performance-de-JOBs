# app.py

import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
import plotly.graph_objects as go
import plotly.io as pio
from prophet import Prophet
import warnings

# Ignora avisos comuns para uma saída mais limpa
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Habilita o CORS para permitir a comunicação com o React

def carregar_dados_job(nome_job):
    """Carrega e processa os dados de um job específico do arquivo JSON."""
    try:
        # Assume que dados_mock.json está na mesma pasta que app.py
        df_full = pd.read_json('dados_mock.json')
        df = df_full[df_full['nome_job'] == nome_job].copy()
        
        if df.empty: 
            return None

        # Converte as colunas de tempo para número, transformando erros em 'NaN' (Not a Number)
        df['TEMPO DE CPU'] = pd.to_numeric(df['tempo_cpu'].str.replace(',', '.'), errors='coerce')
        df['TEMPO DE SALA'] = pd.to_numeric(df['tempo_sala'].str.replace(',', '.'), errors='coerce')
        
        # Remove as linhas que resultaram em erro (agora são 'NaN')
        df.dropna(subset=['TEMPO DE CPU', 'TEMPO DE SALA'], inplace=True)
        df['TIMESTAMP'] = pd.to_datetime(df['data_execucao'] + ' ' + df['hora_execucao'], format='%d/%m/%Y %H.%M.%S')
        # Converte hora_execucao (formato HH.MM.SS) para minutos totais de execução
        hora_parts = df['hora_execucao'].str.split('.', expand=True)
        df['MINUTOS_EXECUCAO'] = hora_parts[0].astype(int) * 60 + hora_parts[1].astype(int) + hora_parts[2].astype(int) / 60
        df = df.sort_values(by='TIMESTAMP').reset_index(drop=True)
        return df
    except FileNotFoundError:
        return None

def remover_outliers_inteligente(df, coluna, tipo):
    """Remove outliers de forma inteligente baseada no tipo de dado."""
    if len(df) < 10:  # Não remove outliers se tiver poucos dados
        return df
    
    # Usa IQR (Interquartile Range) para remoção mais eficaz de outliers
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    
    # Para horário, usa lógica mais restritiva
    if tipo == 'horario':
        multiplicador = 1.3
    else:
        # Para outros tipos, usa lógica padrão
        multiplicador = 1.5
    
    limite_inferior = Q1 - multiplicador * IQR
    limite_superior = Q3 + multiplicador * IQR
    
    df_filtrado = df[
        (df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)
    ].copy()
    
    outliers_removidos = len(df) - len(df_filtrado)
    if outliers_removidos > 0:
        print(f"Para '{coluna}' ({tipo}), foram removidos {outliers_removidos} outliers do treinamento.")
    
    return df_filtrado

def configurar_prophet_por_tipo(tipo):
    """Configura os parâmetros do Prophet baseado no tipo de dado."""
    if tipo == 'horario':
        # Para horário: growth logistic com limites, sazonalidade baixa
        return {
            'growth': 'logistic',
            'changepoint_range': 0.9,
            'changepoint_prior_scale': 0.3,
            'seasonality_prior_scale': 0.5,
            'weekly_seasonality': True,
            'interval_width': 0.90,
            'yearly_seasonality': False,
            'daily_seasonality': False
        }
    elif tipo in ['cpu', 'sala', 'execucao']:
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

@app.route("/api/grafico/<tipo>/<nome_job>")
def gerar_grafico_interativo(tipo, nome_job):
    """
    Rota principal que gera e retorna o JSON de um gráfico interativo com previsão Prophet.
    """
    df = carregar_dados_job(nome_job)
    
    if df is None or len(df) < 5:
        return jsonify({"erro": "Job não encontrado ou dados insuficientes."}), 404

    # Define qual coluna usar com base no tipo de gráfico solicitado
    mapa_colunas = {
        'cpu': ('TEMPO DE CPU', 'Tempo de CPU (min)', 'royalblue'),
        'sala': ('TEMPO DE SALA', 'Tempo de Sala (min)', 'darkorange'),
        'horario': ('HORA_NUMERICA', 'Horário da Execução', 'green'),
        'execucao': ('MINUTOS_EXECUCAO', 'Minutos desde 00:00 (min)', 'purple')
    }
    
    if tipo not in mapa_colunas:
        return jsonify({"erro": "Tipo de gráfico inválido."}), 400
    
    coluna_y, label_y, cor_principal = mapa_colunas[tipo]
    
    # Processa dados específicos para horário
    if tipo == 'horario':
        df['HORA_NUMERICA'] = df['TIMESTAMP'].dt.hour + df['TIMESTAMP'].dt.minute / 60
        coluna_y = 'HORA_NUMERICA'

    # Remove outliers de forma inteligente
    df_treino = remover_outliers_inteligente(df, coluna_y, tipo)

    # Prepara os dados para o formato do Prophet
    df_prophet = df_treino[['TIMESTAMP', coluna_y]].rename(columns={'TIMESTAMP': 'ds', coluna_y: 'y'})
    
    # Adiciona limites apenas para horário
    if tipo == 'horario':
        df_prophet['floor'] = 0
        df_prophet['cap'] = 24

    # Configura o Prophet baseado no tipo
    config_prophet = configurar_prophet_por_tipo(tipo)
    modelo = Prophet(**config_prophet)
    
    # Adiciona sazonalidade customizada para dados não-horário
    if tipo in ['cpu', 'sala', 'execucao']:
        modelo.add_seasonality(name='custom_weekly', period=7, fourier_order=3, prior_scale=0.1)
    
    modelo.fit(df_prophet)

    # Cria o dataframe futuro
    futuro = modelo.make_future_dataframe(periods=7, include_history=True)
    if tipo == 'horario':
        futuro['floor'] = 0
        futuro['cap'] = 24
        
    previsao = modelo.predict(futuro)

    # --- PÓS-PROCESSAMENTO INTELIGENTE DA PREVISÃO ---
    # Garante que os valores de previsão (principal e incerteza) não sejam negativos
    if tipo in ['cpu', 'sala', 'execucao']:
        # Define um piso realista, um pouco abaixo do menor valor já visto
        minimo_real = df_prophet.loc[df_prophet['y'] > 0, 'y'].min()
        piso = max(minimo_real * 0.9, 0.01) # Usa 90% do mínimo, mas nunca menos que 0.01
        
        # 'clip' força todos os valores abaixo do piso a se tornarem o valor do piso
        previsao['yhat'] = previsao['yhat'].clip(lower=piso)
        previsao['yhat_lower'] = previsao['yhat_lower'].clip(lower=piso)
        previsao['yhat_upper'] = previsao['yhat_upper'].clip(lower=piso)

    # Montagem do Gráfico Interativo com Plotly Graph Objects
    fig = go.Figure()

    # Camada 1: Área de Incerteza
    fig.add_trace(go.Scatter(
        x=previsao['ds'].tolist() + previsao['ds'].tolist()[::-1],
        y=previsao['yhat_upper'].tolist() + previsao['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="none",
        name='Intervalo de Confiança'
    ))

    # Camada 2: Linha de Previsão Principal
    fig.add_trace(go.Scatter(
        x=previsao['ds'], 
        y=previsao['yhat'], 
        mode='lines',
        line=dict(color='rgba(0,100,80,0.8)', width=3), 
        name='Previsão'
    ))

    # Camada 3: Pontos dos Dados Reais com hover customizado
    # Cria textos de hover personalizados baseados no tipo
    hover_texts = []
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
        else:  # horario
            hover_text = (
                f"<b>Job:</b> {row['nome_job']}<br>"
                f"<b>Data:</b> {row['data_execucao']}<br>"
                f"<b>Horário:</b> {row['hora_execucao']}<br>"
                f"<b>Return Code:</b> {row['return_code']}<br>"
                f"<b>Usuário:</b> {row['usuario']}<br>"
                f"<b>Job ID:</b> {row['jobid']}"
            )
        hover_texts.append(hover_text)
    
    fig.add_trace(go.Scatter(
        x=df['TIMESTAMP'], 
        y=df[coluna_y], 
        mode='markers',
        marker=dict(color=cor_principal, size=8, line=dict(width=1, color='white')), 
        name='Execuções Reais',
        hovertemplate='%{text}<extra></extra>',
        text=hover_texts
    ))

    # Camada 4: Pontos de Previsão Futura
    # Identifica os dados futuros (após o último timestamp real)
    ultimo_timestamp_real = df['TIMESTAMP'].max()
    previsao_futura = previsao[previsao['ds'] > ultimo_timestamp_real]
    
    # Cria textos de hover para previsão futura
    hover_texts_previsao = []
    for index, row in previsao_futura.iterrows():
        data_formatada = row['ds'].strftime('%d/%m/%Y')
        if tipo == 'cpu':
            valor_formatado = f"{row['yhat']:.2f}"
            hover_text_prev = (
                f"<b>PREVISÃO</b><br>"
                f"<b>Job:</b> {nome_job}<br>"
                f"<b>Data:</b> {data_formatada}<br>"
                f"<b>Tempo CPU Previsto:</b> {valor_formatado}<br>"
                f"<b>Intervalo:</b> {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f}<br>"
                f"<b>Confiança:</b> {config_prophet['interval_width']*100:.0f}%"
            )
        elif tipo == 'sala':
            valor_formatado = f"{row['yhat']:.2f}"
            hover_text_prev = (
                f"<b>PREVISÃO</b><br>"
                f"<b>Job:</b> {nome_job}<br>"
                f"<b>Data:</b> {data_formatada}<br>"
                f"<b>Tempo Sala Previsto:</b> {valor_formatado}<br>"
                f"<b>Intervalo:</b> {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f}<br>"
                f"<b>Confiança:</b> {config_prophet['interval_width']*100:.0f}%"
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
                intervalo_inicio = formatar_minutos_para_horario(row['yhat_lower'])
                intervalo_fim = formatar_minutos_para_horario(row['yhat_upper'])
                
                hover_text_prev = (
                    f"<b>PREVISÃO</b><br>"
                    f"<b>Job:</b> {nome_job}<br>"
                    f"<b>Data:</b> {data_formatada}<br>"
                    f"<b>Hora de Execução Prevista:</b> {valor_formatado}<br>"
                    f"<b>Intervalo:</b> {intervalo_inicio} - {intervalo_fim}<br>"
                    f"<b>Confiança:</b> {config_prophet['interval_width']*100:.0f}%"
                )
        else:  # horario
            # Função auxiliar para formatar horário
            def formatar_horario(valor_decimal):
                horas = int(valor_decimal)
                minutos = int((valor_decimal % 1) * 60)
                return f"{horas:02d}:{minutos:02d}"
            
            valor_formatado = formatar_horario(row['yhat'])
            intervalo_inicio = formatar_horario(row['yhat_lower'])
            intervalo_fim = formatar_horario(row['yhat_upper'])
            
            hover_text_prev = (
                f"<b>PREVISÃO</b><br>"
                f"<b>Job:</b> {nome_job}<br>"
                f"<b>Data:</b> {data_formatada}<br>"
                f"<b>Horário Previsto:</b> {valor_formatado}<br>"
                f"<b>Intervalo:</b> {intervalo_inicio} - {intervalo_fim}<br>"
                f"<b>Confiança:</b> {config_prophet['interval_width']*100:.0f}%"
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

    # Configuração específica do eixo Y e título do gráfico
    yaxis_config = dict(rangemode='tozero')
    title_grafico = f'Análise de {coluna_y.replace("_", " ").title()} do Job: {nome_job.upper()}'

    if tipo == 'horario':
        # Formatar o eixo Y para mostrar horários (lógica existente)
        yaxis_config = dict(
            range=[0, 24],
            tickmode='array',
            tickvals=list(range(0, 25, 2)),  # A cada 2 horas
            ticktext=[f'{h:02d}:00' for h in range(0, 25, 2)],
            title='Horário da Execução'
        )
    
    # Adiciona a formatação para o gráfico de execução
    if tipo == 'execucao':
        # Formatar o eixo Y para mostrar horários a partir dos minutos
        yaxis_config = dict(
            range=[0, 1440],  # 24 horas * 60 minutos
            tickmode='array',
            tickvals=[h * 60 for h in range(0, 25, 4)],  # Marcadores a cada 4 horas
            ticktext=[f'{h:02d}:00' for h in range(0, 25, 4)],
            title='Horário da Execução'
        )
        # Ajusta o título principal do gráfico para refletir o conteúdo
        title_grafico = f'Análise de Hora de Execução do Job: {nome_job.upper()}'

    # Customização Final do Layout
    fig.update_layout(
        title=title_grafico,
        xaxis_title='Data da Execução', 
        yaxis_title=label_y,
        yaxis=yaxis_config, 
        template='plotly_white',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        ),
        hovermode='closest'
    )
    
    # Converte a figura para JSON e a retorna
    return pio.to_json(fig)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

