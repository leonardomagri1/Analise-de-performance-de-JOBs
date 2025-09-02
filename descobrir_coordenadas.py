# descobrir_coordenadas.py
import pyautogui
import time

print("Posicione o mouse sobre o local desejado em 5 segundos...")
time.sleep(5)

# Pega e exibe a posição atual do mouse
posicao_atual = pyautogui.position()
print(f"A posição atual do mouse é: {posicao_atual}")
print("Pressione Ctrl+C no terminal para sair.")