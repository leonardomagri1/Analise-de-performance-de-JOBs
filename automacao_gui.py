# automacao_gui.py

import pyautogui
import time
import sys
import pyperclip
import ctypes
from datetime import datetime
import re
import visualizacao

def garantir_capslock_desativado():
    """
    Verifica se o Caps Lock está ativo. Se estiver, pressiona a tecla para desativá-lo.
    Funciona apenas em Windows.
    """
    # HTONE é 0x14 em hexadecimal, o código da tecla Caps Lock no Windows
    if ctypes.WinDLL("User32.dll").GetKeyState(0x14) == 1:
        print("ALERTA: Caps Lock está ativo. Desativando...")
        pyautogui.press('capslock')

def fazer_login(matricula, senha):
    """
    Abre o cliente IBM OnDemand, espera ele carregar e tenta fazer o login.
    """
    try:
        # --- ETAPA 1: ABRIR O PROGRAMA ---
        print("Iniciando o programa IBM OnDemand...")
        pyautogui.press('win')
        time.sleep(2)
        pyautogui.write('OnDemand')
        time.sleep(2)
        pyautogui.press('enter')
        print("Programa iniciado. Aguardando a janela de login carregar...")

        time.sleep(3)

        # --- ETAPA 2: INSERIR CREDENCIAIS ---
        print("Movendo para o campo de usuário...")
        pyautogui.click(x=608, y=371)
        garantir_capslock_desativado()
        pyautogui.write(matricula, interval=0.1)
        time.sleep(1)

        print("Movendo para o campo de senha...")
        pyautogui.press('tab')
        pyautogui.write(senha, interval=0.1)
        time.sleep(1)

        print("Pressionando Enter para fazer login...")
        pyautogui.press('enter')
        print("Login enviado com sucesso!")
        return True

    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o login: {e}")
        return False

def abrir_area_de_logs():
    """
    Após o login, clica na opção para abrir a pesquisa de logs.
    """
    try:
        # Espera a janela principal do programa carregar completamente após o login.
        # Se a próxima tela demorar para aparecer, AUMENTE este valor.
        print("Aguardando a tela principal carregar...")
        time.sleep(2) 
        
        # --- ATENÇÃO: ATUALIZE AS COORDENADAS ABAIXO ---
        # Use o script 'descobrir_coordenadas.py' para encontrar as coordenadas
        # do botão/ícone/menu que abre a pesquisa de logs.
        print("Clicando na opção para abrir a pesquisa de logs...")
        pyautogui.doubleClick(x=654, y=326)

        print("Área de logs aberta com sucesso!")
        return True

    except Exception as e:
        print(f"Ocorreu um erro ao tentar abrir a área de logs: {e}")
        return False

# ===================================================================
# --- SPRINT 3: FUNÇÃO PARA PESQUISAR O JOB E INSERIR DATAS ---
# ===================================================================
def pesquisar_job_por_data(nome_job, data_inicio, data_fim, filtro_hora, filtro_cpu, filtro_sala, filtro_rc):
    """
    Preenche os campos de pesquisa, incluindo os condicionais opcionais.
    """
    try:
        # Espera a tela de pesquisa carregar. Ajuste o tempo se necessário.
        print("Aguardando a tela de pesquisa carregar...")
        time.sleep(1)

        # --- ATENÇÃO: ATUALIZE AS COORDENADAS ABAIXO ---
        # Use o 'descobrir_coordenadas.py' para achar os locais exatos.

        # 1. Clica e digita o nome do job
        print(f"Pesquisando pelo job: {nome_job}")
        pyautogui.click(x=365, y=145) # <-- MUDE AQUI (coordenada do campo NOME DO JOB)
        pyautogui.write(nome_job, interval=0.1)
        time.sleep(1)

        # 2. Clica e cola a DATA DE INÍCIO
        print(f"Inserindo data de início: {data_inicio}")
        pyautogui.doubleClick(x=339, y=175) # Apenas um clique para dar foco ao campo
        time.sleep(0.5)
        pyperclip.copy(data_inicio) # Copia a data para a área de transferência
        pyautogui.hotkey('ctrl', 'v') # Usa Ctrl+V para colar a data
        time.sleep(1)

        # 3. Clica e cola a DATA DE FIM
        print(f"Inserindo data de fim: {data_fim}")
        pyautogui.doubleClick(x=514, y=178) # Apenas um clique para dar foco ao campo
        time.sleep(0.5)
        pyperclip.copy(data_fim) # Copia a data para a área de transferência
        pyautogui.hotkey('ctrl', 'v') # Usa Ctrl+V para colar a data
        time.sleep(1)
        
        # --- NOVOS CAMPOS OPCIONAIS COM CONDIÇÃO ---
        escolha_hora, valor_hora = filtro_hora
        if escolha_hora:
            print(f"Inserindo filtro de Hora de Execução...")
            # 1. Clica na seta do dropdown de condição da HORA
            pyautogui.click(x=234, y=207) # <-- MUDE AQUI (coordenada da SETA do dropdown de HORA)
            time.sleep(1)

            # 2. Clica na OPÇÃO correta baseada na escolha do usuário
            # ATENÇÃO: Você precisa mapear a coordenada Y de cada opção!
            if escolha_hora == '1': # Igual a
                pyautogui.click(x=356, y=263) # Exemplo de coordenada para "Igual a"
            elif escolha_hora == '2': # Menor que
                pyautogui.click(x=362, y=288) # Exemplo de coordenada para "Menor que"
            elif escolha_hora == '3': # Maior que
                pyautogui.click(x=371, y=312) # Exemplo de coordenada para "Maior que"
            elif escolha_hora == '4': # Menor ou igual a
                pyautogui.click(x=489, y=285) # Exemplo de coordenada para "Menor ou igual a"
            elif escolha_hora == '5': # Maior ou igual a
                pyautogui.click(x=481, y=313) # Exemplo de coordenada para "Maior ou igual a"
            elif escolha_hora == '6': # Entre
                pyautogui.click(x=368, y=341) # Exemplo de coordenada para "Entre"
            elif escolha_hora == '7': # Diferente de
                pyautogui.click(x=495, y=264) # Exemplo de coordenada para "Diferente de"
            
            time.sleep(1)

            # 3. Insere o valor (ou valores, no caso de "Entre")
            if escolha_hora == '6': # Se for "Entre"
                valor1, valor2 = valor_hora
                pyautogui.doubleClick(x=355, y=209) # <-- MUDE AQUI (Coordenada do PRIMEIRO campo de valor HORA)
                pyautogui.write(valor1, interval=0.1)
                pyautogui.doubleClick(x=499, y=210) # <-- MUDE AQUI (Coordenada do SEGUNDO campo de valor HORA)
                pyautogui.write(valor2, interval=0.1)
            else: # Para todas as outras opções
                pyautogui.doubleClick(x=344, y=210) # <-- MUDE AQUI (Coordenada do campo de valor HORA)
                pyautogui.write(valor_hora, interval=0.1)

        escolha_cpu, valor_cpu = filtro_cpu
        if escolha_cpu:
            print(f"Inserindo filtro de Tempo de CPU...")
            # 1. Clica na seta do dropdown de condição do CPU
            pyautogui.click(x=227, y=239) # <-- MUDE AQUI (coordenada da SETA do dropdown de CPU)
            time.sleep(1)

            # 2. Clica na OPÇÃO correta
            if escolha_cpu == '1': # Igual a
                pyautogui.click(x=367, y=298) # Exemplo de coordenada
            elif escolha_cpu == '2': # Menor ou igual a
                pyautogui.click(x=478, y=297) # Exemplo de coordenada
            elif escolha_cpu == '3': # Maior ou igual a
                pyautogui.click(x=359, y=324) # Exemplo de coordenada
            
            time.sleep(1)
            
            # 3. Insere o valor
            pyautogui.doubleClick(x=391, y=244) # <-- MUDE AQUI (Coordenada do campo de valor CPU)
            pyautogui.write(valor_cpu, interval=0.1)

        escolha_sala, valor_sala = filtro_sala
        if escolha_sala:
            print(f"Inserindo filtro de Tempo de Sala...")
            # 1. Clica na seta do dropdown de condição da SALA
            pyautogui.click(x=239, y=272) # <-- MUDE AQUI (coordenada da SETA do dropdown de SALA)
            time.sleep(1)

            # 2. Clica na OPÇÃO correta
            if escolha_sala == '1': # Igual a
                pyautogui.click(x=361, y=332) # Exemplo de coordenada
            elif escolha_sala == '2': # Menor ou igual a
                pyautogui.click(x=482, y=327) # Exemplo de coordenada
            elif escolha_sala == '3': # Maior ou igual a
                pyautogui.click(x=365, y=352) # Exemplo de coordenada
            
            time.sleep(1)

            # 3. Insere o valor
            pyautogui.doubleClick(x=340, y=275) # <-- MUDE AQUI (Coordenada do campo de valor SALA)
            pyautogui.write(valor_sala, interval=0.1)

        escolha_rc, valor_rc = filtro_rc
        if escolha_rc:
            print(f"Inserindo filtro de Return Code...")
            # 1. Clica na seta do dropdown de condição do RC
            pyautogui.click(x=232, y=304) # <-- MUDE AQUI (coordenada da SETA do dropdown de RC)
            time.sleep(1)

            # 2. Clica na OPÇÃO correta
            if escolha_rc == '1': # Igual a
                pyautogui.click(x=355, y=357) # Exemplo de coordenada
            elif escolha_rc == '2': # Menor que
                pyautogui.click(x=357, y=385) # Exemplo de coordenada
            elif escolha_rc == '3': # Maior que
                pyautogui.click(x=501, y=385) # Exemplo de coordenada
            elif escolha_rc == '4': # Diferente de
                pyautogui.click(x=495, y=357) # Exemplo de coordenada
            
            time.sleep(1)

            # 3. Insere o valor
            pyautogui.doubleClick(x=376, y=308) # <-- MUDE AQUI (Coordenada do campo de valor RC)
            pyautogui.write(valor_rc, interval=0.1)

        # 4. Clica no botão de pesquisar/buscar
        print("Iniciando a pesquisa...")
        pyautogui.click(x=1792, y=150) # <-- MUDE AQUI (coordenada do botão PESQUISAR)
        
        print("Pesquisa de logs iniciada com sucesso!")
        return True

    except Exception as e:
        print(f"Ocorreu um erro ao pesquisar o job: {e}")
        return False

def obter_filtro_opcional(nome_filtro, opcoes):
    """
    Exibe um menu de opções para um filtro, valida a escolha e solicita o valor.
    """
    print(f"\n--- {nome_filtro} ---")
    for i, opcao in enumerate(opcoes, 1):
        print(f"  {i} - {opcao}")
    
    while True: # Inicia um loop para garantir uma escolha válida
        escolha = input(f"Escolha uma opção de 1 a {len(opcoes)} (ou Enter para pular): ")
        
        if not escolha:
            return None, None # Se o usuário pressionar Enter, sai e retorna None

        try:
            escolha_int = int(escolha) # Tenta converter a escolha para um número inteiro
            if 1 <= escolha_int <= len(opcoes):
                break # Se for um número válido dentro do intervalo, quebra o loop
            else:
                # Se for um número fora do intervalo
                print(f"ERRO: Opção inválida. Por favor, digite um número entre 1 e {len(opcoes)}.")
        except ValueError:
            # Se a conversão para inteiro falhar (ex: usuário digitou 'abc')
            print("ERRO: Entrada inválida. Por favor, digite apenas o número da opção.")

    # Se o loop terminou, significa que 'escolha' é uma opção válida
    opcao_selecionada = opcoes[int(escolha)-1]
    valor1 = input(f"Digite o valor para '{opcao_selecionada}': ")
    
    if opcao_selecionada.lower() == 'entre':
        valor2 = input(f"Digite o segundo valor para 'Entre': ")
        return escolha, (valor1, valor2)
    
    return escolha, valor1

# ===================================================================
# --- SPRINT 4: FUNÇÃO PARA SELECIONAR E COPIAR OS RESULTADOS ---
# ===================================================================
def copiar_resultados_pesquisa():
    """
    Após a pesquisa, seleciona todos os resultados da lista e os copia para a área de transferência.
    """
    try:
        # Espera os resultados da pesquisa aparecerem. AUMENTE este valor se necessário!
        print("Aguardando resultados da pesquisa...")
        time.sleep(10)

        # --- ATENÇÃO: MAPEIE TODAS AS COORDENADAS ABAIXO ---
        # Use 'descobrir_coordenadas.py' para cada passo.

        # 1. Clica com o botão direito em qualquer item da lista de resultados
        print("Abrindo menu de contexto...")
        pyautogui.rightClick(x=400, y=504) # <-- MUDE AQUI (coordenada de um job na lista)
        time.sleep(2)

        # 2. Clica no botão "Selecionar" (ou similar) do menu
        print("Clicando em 'Selecionar'...")
        pyautogui.click(x=501, y=941) # <-- MUDE AQUI (coordenada do botão 'Selecionar')
        time.sleep(2)

        # 3. Clica no botão "Selecionar Todos" do submenu
        print("Clicando em 'Selecionar Todos'...")
        pyautogui.click(x=966, y=940) # <-- MUDE AQUI (coordenada do botão 'Selecionar Todos')
        time.sleep(2) # Espera a interface atualizar a seleção

        # 4. Clica novamente com o botão direito para reabrir o menu
        print("Reabrindo menu para copiar...")
        pyautogui.rightClick(x=400, y=504) # <-- MUDE AQUI (mesma coordenada do passo 1)
        time.sleep(2)

        # 5. Clica no botão "Copiar"
        print("Copiando dados para a área de transferência...")
        pyautogui.click(x=470, y=695) # <-- MUDE AQUI (coordenada do botão 'Copiar')
        time.sleep(2)

        # 6. Pega os dados da área de transferência
        dados_brutos = pyperclip.paste()
        print("Dados copiados com sucesso!")
        return dados_brutos

    except Exception as e:
        print(f"Ocorreu um erro ao copiar os resultados: {e}")
        return None

def limpar_e_formatar_dados(dados_brutos):
    """
    Recebe os dados brutos e os formata, substituindo múltiplos
    espaços por um único TAB para alinhar com o cabeçalho.
    """
    print("Limpando e formatando os dados copiados...")
    linhas_processadas = []
    # Divide o texto original em linhas individuais
    linhas_originais = dados_brutos.strip().split('\n')

    for linha in linhas_originais:
        # A expressão regular r'\s{2,}' encontra 2 ou mais espaços/tabs em sequência
        # e os substitui por um único TAB ('\t')
        linha_limpa = re.sub(r'\s{2,}', '\t', linha.strip())
        linhas_processadas.append(linha_limpa)
    
    # Junta todas as linhas limpas de volta em um texto único
    return "\n".join(linhas_processadas)

def salvar_dados_em_arquivo(dados, nome_base):
    """
    Adiciona um cabeçalho e salva os dados brutos em um arquivo de texto.
    """
    try:
        # Cria um nome de arquivo único com a data atual
        data_hoje = datetime.now().strftime('%Y-%m-%d')
        nome_arquivo = f"logs_{nome_base}_{data_hoje}.txt"

        # Define o cabeçalho com os nomes das colunas, separados por TAB (\t)
        # O \n no final garante que os dados comecem na linha seguinte
        cabecalho = "NOME DO JOB\tDATA DE EXECUÇÃO\tHORA DE EXECUÇÃO\tTEMPO DE CPU\tTEMPO DE SALA\tRETURN CODE\tUSUARIO\tJOBID\n"

        # Junta o cabeçalho com os dados que foram copiados
        conteudo_final = cabecalho + dados
        
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            f.write(conteudo_final)
        
        print(f"Dados salvos com sucesso no arquivo: {nome_arquivo}")
        return nome_arquivo # <-- ADICIONE ESTA LINHA
    except Exception as e:
        print(f"Ocorreu um erro ao salvar o arquivo: {e}")
        return None

if __name__ == '__main__':
    MINHA_MATRICULA = "u856682"
    MINHA_SENHA = "analista123eL*"
    
    # --- SOLICITA AS INFORMAÇÕES AO USUÁRIO ---
    NOME_DO_JOB = input("Digite o nome do JOB a ser pesquisado: ")
    
    # Loop para validar as datas de início e fim
    while True:
        DATA_INICIAL = input("Digite a data de INÍCIO (DD/MM/AAAA): ")
        DATA_FINAL = input("Digite a data de FIM (DD/MM/AAAA): ")
        try:
            # Converte as strings de data para objetos de data que o Python entende
            data_inicio_obj = datetime.strptime(DATA_INICIAL, '%d/%m/%Y')
            data_fim_obj = datetime.strptime(DATA_FINAL, '%d/%m/%Y')

            # Verifica a condição lógica: a data de início não pode ser maior que a de fim
            if data_inicio_obj <= data_fim_obj:
                break # Se as datas estiverem corretas, sai do loop
            else:
                print("\nERRO: A data de início não pode ser posterior à data de fim. Por favor, tente novamente.\n")
        except ValueError:
            # Este erro acontece se o usuário digitar um formato incorreto (ex: "20-06-2025" ou "abc")
            print("\nERRO: Formato de data inválido. Por favor, use o formato DD/MM/AAAA.\n")

    # --- Filtros Opcionais com Condições ---
    opcoes_hora = ["Igual a", "Menor que", "Maior que", "Menor ou igual a", "Maior ou igual a", "Entre", "Diferente de"]
    opcoes_cpu_sala = ["Igual a", "Menor ou igual a", "Maior ou igual a"]
    opcoes_rc = ["Igual a", "Menor que", "Maior que", "Diferente de"]

    escolha_hora, valor_hora = obter_filtro_opcional("Hora de Execução", opcoes_hora)
    escolha_cpu, valor_cpu = obter_filtro_opcional("Tempo de CPU", opcoes_cpu_sala)
    escolha_sala, valor_sala = obter_filtro_opcional("Tempo de Sala", opcoes_cpu_sala)
    escolha_rc, valor_rc = obter_filtro_opcional("Return Code", opcoes_rc)

    print("\nObrigado! As informações foram recebidas.")
    print("O script começará em 5 segundos. Não mexa no mouse ou teclado!")
    time.sleep(5)
    
    login_sucesso = fazer_login(MINHA_MATRICULA, MINHA_SENHA)
    
    if login_sucesso:
        area_logs_aberta = abrir_area_de_logs()
        
        if area_logs_aberta:
            pesquisa_sucesso = pesquisar_job_por_data(
                NOME_DO_JOB, DATA_INICIAL, DATA_FINAL,
                (escolha_hora, valor_hora),
                (escolha_cpu, valor_cpu),
                (escolha_sala, valor_sala),
                (escolha_rc, valor_rc)
            )

            # Se a pesquisa foi iniciada, tenta copiar os dados
            if pesquisa_sucesso:
                dados_copiados = copiar_resultados_pesquisa()
                
            # Se os dados foram copiados, PRIMEIRO limpa e formata, DEPOIS salva.
            if dados_copiados:
                dados_limpos = limpar_e_formatar_dados(dados_copiados)
                # Captura o nome do arquivo que foi salvo
                caminho_arquivo_salvo = salvar_dados_em_arquivo(dados_limpos, NOME_DO_JOB)

                # Se o arquivo foi salvo com sucesso, gera o gráfico
                if caminho_arquivo_salvo:
                    print("\n--- Iniciando Geração do Gráfico ---")
                    df_dados = visualizacao.processar_dados_log(caminho_arquivo_salvo)
                    if df_dados is not None:
                        visualizacao.gerar_grafico_desempenho(df_dados, NOME_DO_JOB)