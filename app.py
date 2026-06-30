import streamlit as str
from cv2 import imdecode, IMREAD_COLOR, cvtColor, COLOR_BGR2RGB
from numpy import frombuffer, uint8
from ultralytics import YOLO

# Configuração da página Streamlit
str.set_page_config(page_title="Vision Engine | Object Seg", layout="wide")

str.title("👁️ Detector e Segmentador de Objetos em Tempo Real")
str.write("Faça o upload de uma imagem para que o motor de Visão Computacional processe o conteúdo.")

# Cache do modelo para evitar recarregamento a cada interação do usuário
@str.cache_resource
def load_yolo_model():
    # Carrega a versão Nano de segmentação (leve e otimizada para CPU)
    return YOLO("yolov8n-seg.pt")

try:
    model = load_yolo_model()
except Exception as e:
    str.error(f"Erro ao carregar o modelo de visão: {e}")
    str.stop()

# Componente de Upload
uploaded_file = str.file_uploader("Escolha uma imagem do seu dispositivo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Conversão do arquivo de upload para formato legível pelo OpenCV
    file_bytes = frombuffer(uploaded_file.read(), uint8)
    opencv_img = imdecode(file_bytes, IMREAD_COLOR)
    
    # Layout em duas colunas para comparação
    col1, col2 = str.columns(2)
    
    with col1:
        str.subheader("Imagem Original")
        str.image(cvtColor(opencv_img, COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        str.subheader("Resultado da Segmentação")
        
        # Execução da inferência (Roda bem em CPU)
        with str.spinner("Processando tensores e gerando máscaras..."):
            results = model(opencv_img, conf=0.25)  # threshold de confiança de 25%
            
            # Extração da imagem anotada nativa do objeto Results
            annotated_frame = results[0].plot()
            
            # Conversão de BGR (OpenCV) para RGB (Streamlit)
            annotated_frame_rgb = cvtColor(annotated_frame, COLOR_BGR2RGB)
            
        str.image(annotated_frame_rgb, use_container_width=True)
        
        # Exibição analítica das classes encontradas
        detected_boxes = results[0].boxes
        if len(detected_boxes) > 0:
            str.success(f"Detectado(s) {len(detected_boxes)} objeto(s) com sucesso!")
            found_classes = [model.names[int(cls)] for cls in detected_boxes.cls]
            str.info(f"Classes identificadas: {', '.join(set(found_classes))}")
        else:
            str.warning("Nenhum objeto conhecido foi detectado com a confiança mínima de 25%.")