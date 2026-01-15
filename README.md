<div align="center">

# Studio Multimodal AI

*Um framework Python abrangente para análise de IA multimodal em imagens, vídeos e texto*

[![Python](https://img.shields.io/badge/Python->=3.13-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square)](https://github.com/psf/black)

[Funcionalidades](#funcionalidades) • [Instalação](#instalação) • [Início Rápido](#início-rápido) • [Módulos](#módulos) • [Exemplos](#exemplos)

</div>

Um framework Python modular projetado para análise abrangente de IA multimodal. Este projeto fornece ferramentas organizadas e fluxos de trabalho para processamento e análise de dados de imagens, vídeos e texto usando técnicas de aprendizado de máquina de última geração.

> [!TIP]
> Este projeto é estruturado como módulos independentes, permitindo que você use apenas os componentes necessários para suas tarefas específicas de IA multimodal.

## Funcionalidades

- 🖼️ **Processamento de Imagens** - Visão computacional, extração de características, detecção de objetos e classificação
- 🎥 **Análise de Vídeos** - Extração de quadros, detecção de movimento, reconhecimento de ações e análise temporal  
- 📝 **Processamento de Texto** - PLN, análise de sentimento, reconhecimento de entidades e modelagem de linguagem
- 🧩 **Arquitetura Modular** - Módulos independentes que podem ser usados separadamente ou em conjunto
- 🔬 **Pronto para Pesquisa** - Notebooks Jupyter para experimentação e análise
- 🧪 **Cobertura de Testes** - Conjunto abrangente de testes para desenvolvimento confiável
- 📊 **Visualização** - Recursos integrados de plotagem e visualização de dados
- 🚀 **Configuração Fácil** - Processo simples de instalação e configuração

## Instalação

### Pré-requisitos

- Python >= 3.13
- pip ou gerenciador de pacotes conda

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/yourusername/studio-multimodal-ai.git
cd studio-multimodal-ai

# Instale as dependências
pip install -r requirements.txt

# Ou instale no modo de desenvolvimento
pip install -e .
```

### Ambiente Virtual (Recomendado)

```bash
# Crie um ambiente virtual
python -m venv .venv

# Ative o ambiente virtual
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

## Início Rápido

```python
# Exemplo de processamento de imagens
from images_module.src.preprocessing import preprocess_image
from images_module.src.features import extract_features

# Carregue e processe uma imagem
image = preprocess_image('path/to/image.jpg')
features = extract_features(image)

# Exemplo de processamento de vídeo
from videos_module.src.preprocessing import extract_frames
from videos_module.src.analysis import detect_motion

# Extraia quadros e analise movimento
frames = extract_frames('path/to/video.mp4')
motion_data = detect_motion(frames)

# Exemplo de processamento de texto
from text_module.src.preprocessing import clean_text, tokenize_text
from text_module.src.analysis import sentiment_analysis

# Processe e analise texto
clean_content = clean_text('Seu conteúdo de texto aqui')
tokens = tokenize_text(clean_content)
sentiment = sentiment_analysis(clean_content)
```

## Módulos

### 🖼️ Módulo de Imagens

Localizado em [`images_module/`](images_module/), este módulo fornece capacidades abrangentes de processamento de imagens:

- **Pré-processamento**: Carregamento, redimensionamento, normalização e melhoria de imagens
- **Extração de Características**: Características tradicionais de CV e embeddings de deep learning  
- **Detecção de Objetos**: YOLO, R-CNN e outras frameworks de detecção
- **Classificação**: Categorização de imagens usando modelos pré-treinados e personalizados

### 🎥 Módulo de Vídeos

Localizado em [`videos_module/`](videos_module/), este módulo lida com análise de vídeo:

- **Processamento de Quadros**: Extração, filtragem e amostragem temporal
- **Análise de Movimento**: Fluxo óptico, rastreamento de objetos e detecção de movimento
- **Reconhecimento de Ações**: Classificação de atividades e detecção de eventos temporais
- **Resumo de Vídeo**: Extração de quadros-chave e resumo de conteúdo

### 📝 Módulo de Texto

Localizado em [`text_module/`](text_module/), este módulo fornece capacidades de PLN:

- **Pré-processamento de Texto**: Limpeza, tokenização e normalização
- **Análise**: Análise de sentimento, reconhecimento de entidades e modelagem de tópicos
- **Modelos de Linguagem**: Integração com transformers e modelos personalizados
- **Classificação**: Categorização de texto e detecção de intenções

## Estrutura do Projeto

```
studio-multimodal-ai/
├── images_module/          # Processamento de imagens e visão computacional
│   ├── data/              # Datasets de imagens
│   ├── notebooks/         # Notebooks Jupyter para experimentação
│   ├── src/              # Código principal de processamento de imagens
│   └── tests/            # Testes unitários para funcionalidades de imagem
├── videos_module/          # Processamento e análise de vídeo
│   ├── data/              # Datasets de vídeo
│   ├── notebooks/         # Notebooks de análise de vídeo
│   ├── src/              # Código principal de processamento de vídeo
│   └── tests/            # Testes unitários para funcionalidades de vídeo
├── text_module/           # Processamento de texto e PLN
│   ├── data/              # Datasets de texto
│   ├── notebooks/         # Notebooks de experimentos de PLN
│   ├── src/              # Código principal de processamento de texto
│   └── tests/            # Testes unitários para funcionalidades de texto
├── docs/                  # Documentação e guias
├── requirements.txt       # Dependências do projeto
└── setup.py              # Configuração do pacote
```

## Exemplos

### Pipeline de Classificação de Imagens

```python
from images_module.src.preprocessing import preprocess_image
from images_module.src.classification import ImageClassifier

# Inicialize o classificador
classifier = ImageClassifier(model_type='resnet50')

# Processe e classifique a imagem
image = preprocess_image('sample.jpg', target_size=(224, 224))
prediction = classifier.predict(image)
print(f"Classe predita: {prediction}")
```

### Detecção de Movimento em Vídeo

```python
from videos_module.src.preprocessing import extract_frames
from videos_module.src.analysis import MotionDetector

# Extraia quadros e detecte movimento
frames = extract_frames('video.mp4', frame_interval=5)
detector = MotionDetector()
motion_regions = detector.detect(frames)
```

### Análise de Sentimento de Texto

```python
from text_module.src.preprocessing import TextPreprocessor
from text_module.src.analysis import SentimentAnalyzer

# Inicialize os componentes
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()

# Analise o sentimento
text = "Este é um framework de IA multimodal incrível!"
clean_text = preprocessor.clean_text(text)
sentiment = analyzer.analyze(clean_text)
print(f"Sentimento: {sentiment}")
```

## Desenvolvimento

### Executando Testes

```bash
# Execute todos os testes
pytest

# Execute testes de módulos específicos
pytest images_module/tests/
pytest videos_module/tests/
pytest text_module/tests/
```

### Formatação de Código

```bash
# Formate o código com black
black .

# Verifique o estilo do código
flake8 .
```

### Notebooks Jupyter

Lance o Jupyter para explorar os notebooks de exemplo:

```bash
jupyter notebook
# Navegue até a pasta notebooks/ de qualquer módulo
```

## Dependências

O projeto inclui dependências abrangentes para IA multimodal:

- **Núcleo**: NumPy, Pandas, SciPy
- **Visão Computacional**: OpenCV, Pillow, scikit-image
- **Processamento de Vídeo**: MoviePy, imageio
- **PLN**: NLTK, spaCy, transformers
- **Aprendizado de Máquina**: scikit-learn, PyTorch, TensorFlow
- **Visualização**: Matplotlib, Seaborn, Plotly
- **Desenvolvimento**: pytest, black, flake8

Consulte [`requirements.txt`](requirements.txt) para a lista completa.

## Recursos

- [Visão Computacional com OpenCV](https://opencv.org/)
- [Processamento de Vídeo com MoviePy](https://zulko.github.io/moviepy/)
- [PLN com spaCy](https://spacy.io/)
- [Deep Learning com PyTorch](https://pytorch.org/)
- [Biblioteca Transformers](https://huggingface.co/transformers/)

## FAQ

**P: Posso usar módulos individuais separadamente?**
R: Sim! Cada módulo (`images_module`, `videos_module`, `text_module`) foi projetado para ser independente e pode ser importado separadamente.

**P: Quais versões do Python são suportadas?**
R: Este projeto requer Python 3.13 ou superior para desempenho e compatibilidade ideais.

**P: Como adiciono modelos personalizados?**
R: Cada módulo possui arquitetura extensível. Adicione seus modelos personalizados aos respectivos diretórios `src/` e siga os padrões existentes.

## Solução de Problemas

**Problemas de Instalação:**
- Certifique-se de ter o Python 3.13+ instalado
- Use um ambiente virtual para evitar conflitos de dependências
- No Windows, instale o Visual Studio Build Tools para compilação

**Problemas de Memória com Arquivos Grandes:**
- Processe dados em lotes para datasets grandes
- Use tamanhos de chunk apropriados para processamento de vídeo
- Monitore o uso de memória durante o processamento

**Suporte a GPU:**
- Instale versões compatíveis com CUDA do PyTorch/TensorFlow
- Verifique se os drivers da GPU estão instalados corretamente
- Verifique a compatibilidade CUDA com seu hardware
