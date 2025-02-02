# Deep Learning and Neural Networks and more

This GitHub repository, is a collection of projects and tools designed to support students and professionals exploring the exciting fields of data science and artificial intelligence.  Here's what you'll find:

- **Diverse Project Examples:** The repository showcases practical applications of various deep learning techniques, offering hands-on learning experiences.
- Focus Areas: The projects span a wide range of topics, including:
  - Machine Learning
  - Artificial Neural Networks
  - Deep Learning
  - Computer Vision
  - Image Processing and Treatment
  - Image Detection and Classification
  - Product Recommendation Networks
  - Large Language Models (LLMs)
  - Natural Language Processing (NLP)
- **Educational Resources:** The repository aims to be a valuable resource for those learning and working in these areas, providing code examples and potentially tutorials to help users understand the concepts.
- **Community Contributions:** Contributions and feedback are welcome, fostering a collaborative environment for learning and development.

## Transfer Learning
### PT-BR
Este projeto, inicialmente desenvolvido com método transfer-learning com o Dataset da MNIST, foi adaptado para classificar imagens de gatos e cachorros Dataset cats_vs_dogs. A base de código é flexível e pode ser facilmente ajustada para trabalhar com diferentes conjuntos de dados, demonstrando os conceitos aprendidos através de Redes Neurais.

Com ele é possível realizar um upload de imagem(s) de exemplo para ser(em) categorizado(s) de acordo com a probabilidade definida pelo modelo.

### EN
This project, initially developed using a transfer learning approach with the MNIST dataset, has been adapted to classify images of cats and dogs from the cats_vs_dogs dataset. The codebase is flexible and can be easily adjusted to work with different datasets, demonstrating the concepts learned through Neural Networks.
With it, it is possible to upload sample image(s) to be categorized according to the probability defined by the model.

## Redução de Dimensionalidade / Dimensionality Reduction
### PT-BR
Notebook com a função de upload para aplicar a redução de dimensionalidade em uma imagem, com aplicação de binarização, conversão para cinza, preto e branco (e inversão) e contornos à imagem. A redução de dimensionalidade é muito usada em algoritmos de redes neurais para muitos propósitos, um deles é redução do peso de imagens sem que a máquina perca a capacidade da leitura e interpretação da imagem.

### EN
A Colab Notebook with an upload function to apply dimensionality reduction to an image. It includes features for binarization, grayscale conversion, black and white inversion, and contour detection. Dimensionality reduction is commonly used in neural network algorithms for various purposes, including reducing image weight without compromising the machine's ability to read and interpret the image.

## Notebook Matriz de Confusão / Confusion Matrix Notebook
### PT-BR
Criação da Matriz de Confusão:

Criação da matriz de confusão para avaliar o desempenho do modelo em uma tarefa de classificação.
Análise detalhada dos componentes da matriz de confusão (verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos).

Integração do TensorBoard:
 - Configuração do TensorBoard para registrar métricas e visualizar o processo de treinamento.
 - Visualização da matriz de confusão como um mapa de calor no TensorBoard.
 - Exploração de outros recursos do TensorBoard para obter insights sobre o desempenho do modelo.

Público-Alvo:
Este notebook é adequado para profissionais de aprendizado de máquina, cientistas de dados e estudantes que desejam:
 - Aprender a construir e treinar modelos de aprendizado de máquina.
 - Avaliar o desempenho do modelo usando matrizes de confusão.
 - Utilizar o TensorBoard para visualizar o progresso do treinamento e os resultados.

Pré-requisitos:
 - Implementação de modelos de aprendizado de máquina.
 - Avaliação do desempenho do modelo.
 - Utilização do TensorBoard para visualizações informativas.

### EN
This Colab Notebook provides a step-by-step guide to training a machine learning model, evaluating its performance using a confusion matrix, and visualizing the results with TensorBoard.

Confusion Matrix Creation:
Creation of the confusion matrix to assess the model's performance on a classification task.
In-depth analysis of the confusion matrix's components (true positives, true negatives, false positives, and false negatives).

TensorBoard Integration:
 - Configuration of TensorBoard to log metrics and visualize the training process.
 - Visualization of the confusion matrix as a heatmap in TensorBoard.
 - Exploration of other TensorBoard features to gain insights into model performance.

Target Audience:
This notebook is suitable for machine learning practitioners, data scientists, and students who want to:
 - Learn how to build and train machine learning models.
 - Evaluate model performance using confusion matrices.
 - Utilize TensorBoard to visualize training progress and results.

Prerequisites:
 - Implementing machine learning models.
 - Evaluating model performance.
 - Leveraging TensorBoard for insightful visualizations.

## YOLOv3 Object Detection
### PT-BR
Este notebook demonstra como treinar um detector de objetos YOLOv3 usando o framework Darknet.
Utilizaremos o conjunto de dados COCO para treinar o modelo em uma variedade de objetos.
O notebook irá guiá-lo através do processo de configuração do ambiente, configuração do processo de treinamento e avaliação do modelo treinado.

### EN
This notebook demonstrates how to train a YOLOv3 object detector using the Darknet framework.
We will use the COCO dataset to train the model on a variety of objects.
The notebook will guide you through the process of setting up the environment, configuring the training process, and evaluating the trained model.

## Face Detection Using OpenCV and ResNet
### PT-BR
Este notebook demonstra um sistema de detecção e classificação facial utilizando técnicas de Visão Computacional. Ele utiliza manipulação de imagens e um modelo ResNet pré-treinado para analisar imagens capturadas por uma webcam. O objetivo é melhorar a precisão e confiabilidade da detecção facial por meio de:
- Detecção de faces: Identificar e localizar faces dentro do fluxo de vídeo da webcam.
- Classificação de faces: Determinar a identidade ou categoria das faces detectadas.
- Pontuação de Confiança: Atribuir uma pontuação de confiança a cada detecção, filtrando previsões com baixa confiança e garantindo que apenas detecções confiáveis sejam consideradas.

### EN
This notebook demonstrates a face detection and classification system using Computer Vision techniques. It leverages image manipulation and a pre-trained ResNet model to analyze images captured from a webcam. The goal is to enhance face detection accuracy and reliability by:

- Detecting faces: Identifying and locating faces within the webcam feed.
- Classifying faces: Determining the identity or category of detected faces.
- Confidence Scoring: Assigning a confidence score to each detection, filtering out low-confidence predictions and ensuring only reliable detections are considered.

## Image Similarity Recommendations Notebook
### PT-BR
Este notebook apresenta um sistema de recomendação visual que, a partir de uma imagem de entrada, sugere produtos com características visuais similares. O sistema utiliza uma rede neural profunda treinada para identificar e classificar objetos em diversas categorias, como roupas, acessórios e eletrônicos. A recomendação é baseada na similaridade visual entre a imagem de entrada e os produtos em nosso banco de dados, desconsiderando informações textuais como preço e marca.

### EN
This notebook demonstrates a visual recommendation system. Given an input image, it suggests similar products based on visual attributes. A deep neural network, trained on various object categories, powers the system. Recommendations are derived from visual comparisons, ignoring textual data like price and brand.
