import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime

print("=== TESTE SIMPLES DO TENSORFLOW ===")
print(f"Versão TensorFlow: {tf.__version__}")

# Nome do arquivo PDF com timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_filename = f'resultado_tensorflow_{timestamp}.pdf'

# Criar um arquivo PDF para armazenar resultados
pdf = PdfPages(pdf_filename)

# Função para adicionar texto ao PDF
def adicionar_texto_ao_pdf(texto):
    fig = plt.figure(figsize=(8, 1))
    plt.axis('off')
    plt.text(0.01, 0.5, texto, fontsize=10, va='center')
    pdf.savefig(fig)
    plt.close(fig)

adicionar_texto_ao_pdf(f"=== TESTE SIMPLES DO TENSORFLOW ===\nVersão TensorFlow: {tf.__version__}\nData: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# 1. CARREGAR DADOS - Dígitos escritos à mão (MNIST)
print("\n[1] Carregando o dataset MNIST...")
mnist = tf.keras.datasets.mnist
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

# Normalizar os dados (converter de 0-255 para 0-1)
x_treino, x_teste = x_treino / 255.0, x_teste / 255.0

# 2. VISUALIZAR ALGUNS EXEMPLOS
print("\n[2] Visualizando alguns exemplos do dataset...")
fig = plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_treino[i], cmap='gray')
    plt.title(f"Dígito: {y_treino[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('exemplos_mnist.png')
pdf.savefig(fig)  # Salvar no PDF
plt.close(fig)
print("    → Imagem com exemplos salva como 'exemplos_mnist.png'")

adicionar_texto_ao_pdf("\n[3] Criando modelo simples...")

# 3. CRIAR UM MODELO SIMPLES
print("\n[3] Criando modelo simples...")
modelo = tf.keras.models.Sequential([
    # Achatar a imagem 28x28 em um vetor de 784 elementos
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Camada oculta com 128 neurônios
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Camada de saída com 10 neurônios (um para cada dígito)
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. COMPILAR O MODELO
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. TREINAR O MODELO
print("\n[4] Treinando o modelo...")
print("    Época | Precisão | Perda   | Precisão Teste | Perda Teste")
print("    -----------------------------------------------------")

# Lista para armazenar os resultados de treinamento
resultados_treinamento = []

class MostrarProgresso(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoca, logs=None):
        if epoca % 1 == 0:  # mostrar a cada 1 época
            linha = f"    {epoca+1:5d} | {logs['accuracy']*100:6.2f}% | {logs['loss']:7.4f} | " \
                   f"{logs['val_accuracy']*100:7.2f}% | {logs['val_loss']:7.4f}"
            print(linha)
            resultados_treinamento.append(linha)

# Treinar por apenas 5 épocas para ser rápido
historico = modelo.fit(
    x_treino, y_treino,
    epochs=5,
    batch_size=128,
    validation_data=(x_teste, y_teste),
    verbose=0,
    callbacks=[MostrarProgresso()]
)

# Adicionar resultados do treinamento ao PDF
adicionar_texto_ao_pdf("\n[4] Resultados do treinamento:")
adicionar_texto_ao_pdf("    Época | Precisão | Perda   | Precisão Teste | Perda Teste")
adicionar_texto_ao_pdf("    -----------------------------------------------------")
for linha in resultados_treinamento:
    adicionar_texto_ao_pdf(linha)

# 6. Plotar gráficos de treinamento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, len(historico.history['accuracy']) + 1)

# Gráfico de precisão
ax1.plot(epochs_range, [acc * 100 for acc in historico.history['accuracy']], label='Treino')
ax1.plot(epochs_range, [acc * 100 for acc in historico.history['val_accuracy']], label='Teste')
ax1.set_title('Precisão durante o Treinamento')
ax1.set_ylabel('Precisão (%)')
ax1.set_xlabel('Época')
ax1.legend()
ax1.grid(True)

# Gráfico de perda
ax2.plot(epochs_range, historico.history['loss'], label='Treino')
ax2.plot(epochs_range, historico.history['val_loss'], label='Teste')
ax2.set_title('Perda durante o Treinamento')
ax2.set_ylabel('Perda')
ax2.set_xlabel('Época')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
pdf.savefig(fig)  # Salvar no PDF
plt.close(fig)

# 7. AVALIAR O MODELO
print("\n[5] Avaliação final do modelo:")
perda, precisao = modelo.evaluate(x_teste, y_teste, verbose=0)
resultado_final = f"    Precisão: {precisao*100:.2f}%"
print(resultado_final)
adicionar_texto_ao_pdf("\n[5] Avaliação final do modelo:")
adicionar_texto_ao_pdf(resultado_final)

# 8. FAZER ALGUMAS PREVISÕES
print("\n[6] Fazendo algumas previsões...")
adicionar_texto_ao_pdf("\n[6] Algumas previsões do modelo:")

# Seleciona 5 imagens aleatórias para previsão
indices = np.random.randint(0, len(x_teste), 5)
fig = plt.figure(figsize=(12, 5))

for i, idx in enumerate(indices):
    # Previsão
    imagem = x_teste[idx:idx+1]
    previsao = modelo.predict(imagem, verbose=0)
    digito_previsto = np.argmax(previsao)
    confianca = previsao[0][digito_previsto] * 100
    
    # Visualizar
    plt.subplot(1, 5, i+1)
    plt.imshow(x_teste[idx], cmap='gray')
    plt.title(f"Real: {y_teste[idx]}\nPrevisto: {digito_previsto}\nConfiança: {confianca:.1f}%")
    plt.axis('off')

plt.tight_layout()
plt.savefig('previsoes_mnist.png')
pdf.savefig(fig)  # Salvar no PDF
plt.close(fig)
print("    → Imagem com previsões salva como 'previsoes_mnist.png'")

# 9. Matriz de confusão
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Gerar previsões para todo o conjunto de teste
previsoes = modelo.predict(x_teste, verbose=0)
previsoes_classes = np.argmax(previsoes, axis=1)

# Criar matriz de confusão
cm = confusion_matrix(y_teste, previsoes_classes)

# Visualizar matriz de confusão
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Previsão')
ax.set_ylabel('Valor Real')
ax.set_title('Matriz de Confusão')
plt.tight_layout()
pdf.savefig(fig)  # Salvar no PDF
plt.close(fig)

# Finalizar PDF
adicionar_texto_ao_pdf("\n=== TESTE CONCLUÍDO COM SUCESSO ===")
adicionar_texto_ao_pdf(f"PDF gerado em: {pdf_filename}")
pdf.close()

print("\n=== TESTE CONCLUÍDO COM SUCESSO ===")
print(f"O TensorFlow está funcionando corretamente no seu ambiente!")
print(f"Resultados salvos em PDF: {pdf_filename}")