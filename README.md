# TensorFlow-project

Este código implementa um modelo simples de aprendizado de máquina usando TensorFlow para reconhecimento de dígitos escritos à mão. Aqui estão as principais funcionalidades:

Configuração e inicialização:

Importa bibliotecas necessárias (TensorFlow, NumPy, Matplotlib)
Cria um arquivo PDF com timestamp para salvar resultados
Carregamento de dados:

Utiliza o dataset MNIST (dígitos manuscritos)
Normaliza os dados de entrada para valores entre 0 e 1
Visualização de exemplos:

Mostra 5 imagens de exemplo do dataset
Salva estas visualizações no PDF
Criação do modelo neural:

Implementa uma rede neural simples com:
Uma camada de achatamento (flatten) para transformar imagens 28x28 em vetores
Uma camada oculta com 128 neurônios e função de ativação ReLU
Uma camada de saída com 10 neurônios (um para cada dígito 0-9)
Treinamento:

Treina o modelo por 5 épocas
Registra e exibe métricas de precisão e perda a cada época
Utiliza callbacks para mostrar progresso
Visualização do treinamento:

Gera gráficos para mostrar a evolução da precisão e perda
Compara desempenho entre conjunto de treino e teste
Avaliação do modelo:

Calcula a precisão final no conjunto de teste
Mostra o resultado quantitativo
Demonstração de previsões:

Seleciona 5 imagens aleatórias do conjunto de teste
Faz previsões e mostra os resultados visualmente
Exibe o dígito real, o previsto e o nível de confiança
Matriz de confusão:

Gera uma matriz de confusão para avaliar o desempenho do modelo
Visualiza os acertos e erros entre as diferentes classes (dígitos)
Conclusão:

Finaliza o PDF com um resumo
Exibe mensagem confirmando que o TensorFlow está funcionando corretamente
