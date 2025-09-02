# visualizacao.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def processar_dados_log(caminho_arquivo):
    """
    Lê o arquivo de log TXT, processa e limpa os dados para análise.
    Retorna um DataFrame do Pandas.
    """
    try:
        print(f"Lendo dados do arquivo: {caminho_arquivo}")
        # Lê o arquivo de texto, usando TAB como separador
        df = pd.read_csv(caminho_arquivo, sep='\t')

        # --- LIMPEZA E CONVERSÃO DOS DADOS (COM TRATAMENTO DE ERRO) ---
        # Substitui vírgula por ponto nos campos de tempo
        df['TEMPO DE CPU'] = df['TEMPO DE CPU'].str.replace(',', '.', regex=False)
        df['TEMPO DE SALA'] = df['TEMPO DE SALA'].str.replace(',', '.', regex=False)

        # Converte para número de forma segura. Se um valor não for numérico (ex: 'USRIWPRD'),
        # ele será transformado em 'NaN' (Not a Number) em vez de causar um erro.
        df['TEMPO DE CPU'] = pd.to_numeric(df['TEMPO DE CPU'], errors='coerce')
        df['TEMPO DE SALA'] = pd.to_numeric(df['TEMPO DE SALA'], errors='coerce')

        # Remove (descarta) quaisquer linhas que tenham dados inválidos (NaN) nas colunas de tempo
        df.dropna(subset=['TEMPO DE CPU', 'TEMPO DE SALA'], inplace=True)

        # Combina data e hora em uma única coluna de 'Timestamp' para o gráfico
        df['TIMESTAMP'] = pd.to_datetime(df['DATA DE EXECUÇÃO'] + ' ' + df['HORA DE EXECUÇÃO'], format='%d/%m/%Y %H.%M.%S')

        # Ordena os dados pela data/hora para que o gráfico de linha faça sentido
        df = df.sort_values(by='TIMESTAMP')

        print("Dados processados com sucesso.")
        return df

    except FileNotFoundError:
        print(f"ERRO: O arquivo de log '{caminho_arquivo}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao processar os dados: {e}")
        return None


def gerar_grafico_desempenho(df, nome_job):
    """
    Gera um gráfico de desempenho com eixos duplos (tempo e horário) e médias móveis.
    """
    if df is None or df.empty:
        print("Não há dados para gerar o gráfico.")
        return

    print("Gerando o gráfico de desempenho aprimorado...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(17, 9))

    # --- EIXO Y ESQUERDO (ax1): TEMPOS DE EXECUÇÃO EM MINUTOS ---
    cor_cpu = 'royalblue'
    cor_sala = 'darkorange'
    ax1.plot(df['TIMESTAMP'], df['TEMPO DE CPU'], marker='o', linestyle='-', label='Tempo de CPU (min)', color=cor_cpu)
    ax1.plot(df['TIMESTAMP'], df['TEMPO DE SALA'], marker='s', linestyle='--', label='Tempo de Sala (min)', color=cor_sala)
    
    # --- TENDÊNCIA COM PREVISÃO DE 7 DIAS ---
    # 1. Prepara os dados para o cálculo da regressão
    x_numeric = mdates.date2num(df['TIMESTAMP'])
    
    # 2. Cria um novo intervalo de datas que se estende 7 dias no futuro
    data_final_previsao = df['TIMESTAMP'].iloc[-1] + pd.Timedelta(days=7)
    datas_previsao = pd.date_range(start=df['TIMESTAMP'].iloc[0], end=data_final_previsao)
    x_previsao_numeric = mdates.date2num(datas_previsao)
    
    # 3. Calcula e plota a previsão para o Tempo de CPU
    z_cpu = np.polyfit(x_numeric, df['TEMPO DE CPU'], 1)
    p_cpu = np.poly1d(z_cpu)
    ax1.plot(datas_previsao, p_cpu(x_previsao_numeric), color=cor_cpu, linestyle=':', label='Previsão CPU')

    # 4. Calcula e plota a previsão para o Tempo de Sala
    z_sala = np.polyfit(x_numeric, df['TEMPO DE SALA'], 1)
    p_sala = np.poly1d(z_sala)
    ax1.plot(datas_previsao, p_sala(x_previsao_numeric), color=cor_sala, linestyle=':', label='Previsão Sala')

    ax1.set_xlabel('Data da Execução', fontsize=12)
    ax1.set_ylabel('Tempo de Execução (em minutos)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # --- EIXO Y DIREITO (ax2): HORÁRIO DA EXECUÇÃO ---
    ax2 = ax1.twinx() # Cria um segundo eixo Y que compartilha o mesmo eixo X
    cor_hora = 'green'
    # Converte a hora para um formato numérico (horas a partir da meia-noite) para plotar
    df['HORA_NUMERICA'] = df['TIMESTAMP'].dt.hour + df['TIMESTAMP'].dt.minute / 60
    ax2.plot(df['TIMESTAMP'], df['HORA_NUMERICA'], color=cor_hora, marker='^', linestyle='-.', label='Horário de Execução')
    ax2.set_ylabel('Horário da Execução', fontsize=12, color=cor_hora)
    
    # Formata o eixo Y direito para mostrar as horas no formato "HH:MM"
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f"{int(y):02d}:{int((y*60)%60):02d}"))
    ax2.tick_params(axis='y', labelcolor=cor_hora)
    ax2.grid(False) # Desativa a grade para o eixo secundário para não poluir

    # --- FORMATAÇÃO FINAL ---
    fig.suptitle(f'Análise de Desempenho do Job: {nome_job}', fontsize=16, weight='bold')
    
    # Coleta as legendas de ambos os eixos para uni-las
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # MUDANÇA: Altera a localização da legenda para o canto superior direito
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Formata o eixo X para mostrar apenas a data
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    fig.autofmt_xdate()

    # Salva o gráfico
    nome_arquivo_grafico = f"grafico_avancado_{nome_job}_{pd.Timestamp.now().strftime('%Y-%m-%d')}.png"
    try:
        plt.savefig(nome_arquivo_grafico, bbox_inches='tight')
        print(f"Gráfico salvo com sucesso como: {nome_arquivo_grafico}")
    except Exception as e:
        print(f"Ocorreu um erro ao salvar o gráfico: {e}")