
# Integração AWS S3 com Classificação de Imagens usando YOLO

Este projeto é um estudo para integrar a análise de imagens com um modelo pré-treinado YOLO e armazenar os resultados no AWS S3. O fluxo envolve o processamento de imagens, a detecção de objetos e o envio dos arquivos processados e das inferências para um bucket no AWS S3.

## 📌 Tecnologias Utilizadas

- **Python**: Linguagem principal para o processamento de imagens e integração com AWS S3.
- **OpenCV** (`cv2`): Usado para manipulação e pré-processamento de imagens.
- **Ultralytics YOLO**: Modelo pré-treinado para detecção de objetos.
- **Boto3**: Biblioteca para interagir com serviços da AWS, incluindo S3.
- **JSON**: Formato para armazenar os resultados da detecção de objetos.

## 📁 Estrutura do Projeto

```
/projeto-yolo-s3
│── imgs/                   # Diretório com as imagens brutas
│── processed_imgs/          # Diretório para armazenar imagens processadas
│── detection_results.json   # JSON com os resultados da detecção
│── main.py                  # Script principal de processamento e upload
│── requirements.txt         # Dependências do projeto
│── README.md                # Documentação do projeto
```

## 🔧 Configuração e Execução

### 1️⃣ Pré-requisitos

Antes de executar o projeto, certifique-se de ter:

- Uma conta na AWS e um **bucket S3** criado.
- Credenciais da AWS configuradas no seu ambiente (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
- Python instalado (`>=3.8`).
- Dependências instaladas:

```bash
pip install -r requirements.txt
```

### 2️⃣ Configuração do AWS S3

No código, substitua `"bucket_name"` pelo nome do seu bucket no AWS S3:

```python
AWS_BUCKET_NAME = "bucket_name"
```

Se necessário, configure suas credenciais localmente com:

```bash
aws configure
```

### 3️⃣ Execução do Script

Após configurar tudo corretamente, execute o script principal:

```bash
python script.py
```

## 🚀 Fluxo do Processo

1. **Pré-processamento**:
   - O script carrega as imagens da pasta `imgs/`, redimensiona para 256x256 pixels e salva na pasta `processed_imgs/`.

2. **Upload das Imagens**:
   - As imagens processadas são enviadas para o **AWS S3** na pasta `imgs/`.

3. **Inferência com YOLO**:
   - O modelo pré-treinado YOLO (`yolo11n.pt`) realiza a detecção de objetos nas imagens processadas.

4. **Geração do JSON**:
   - O resultado da detecção é armazenado em `detection_results.json`, incluindo:
     - **Coordenadas dos objetos detectados**
     - **Classe do objeto**
     - **Confiança do modelo**

5. **Upload do JSON para S3**:
   - O arquivo `detection_results.json` é enviado para o AWS S3 na pasta `inferences/`.

## 📜 Exemplo de Saída (JSON)

O JSON gerado terá um formato semelhante a este:

```json
[
    {
        "image_path": "./processed_imgs/image1.jpg",
        "detections": [
            {
                "class_id": 0,
                "bounding_box": [50, 30, 200, 180],
                "confidence": 0.95
            }
        ]
    }
]
```

## 📩 Contato

Caso tenha dúvidas ou sugestões, fique à vontade para contribuir! 🚀
