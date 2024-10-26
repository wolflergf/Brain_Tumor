import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Configuração da página Streamlit
st.set_page_config(
    page_title="Classificador de Tumor Cerebral", page_icon="🧠", layout="wide"
)


# Função para carregar o modelo com tratamento de erro
@st.cache_resource
def load_classification_model():
    try:
        return load_model("brain_tumor_model.keras", compile=False)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None


# Função para preprocessar e classificar a imagem
def predict_tumor(img, model):
    try:
        # Preprocessamento
        img = img.convert("RGB")  # Garantir que a imagem está em RGB
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predição
        with st.spinner("Processando a imagem..."):
            result = model.predict(img_array)

        return result
    except Exception as e:
        st.error(f"Erro durante o processamento da imagem: {str(e)}")
        return None


# Função para exibir as probabilidades como barras de progresso
def display_probabilities(probabilities, classes):
    for idx, (prob, class_name) in enumerate(zip(probabilities[0], classes)):
        col1, col2, col3 = st.columns([2, 6, 2])
        with col1:
            st.write(f"{class_name}:")
        with col2:
            st.progress(float(prob))
        with col3:
            st.write(f"{prob:.1%}")


def main():
    # Título e descrição
    st.title("🧠 Classificador de Tumor Cerebral")
    st.markdown(
        """
    Este aplicativo utiliza inteligência artificial para identificar diferentes tipos de tumores cerebrais
    em imagens de ressonância magnética.
    """
    )

    # Carregar o modelo
    model = load_classification_model()

    if model is None:
        st.error(
            "Não foi possível iniciar o aplicativo devido a um erro no carregamento do modelo."
        )
        return

    # Interface de upload
    st.markdown("### Upload de Imagem")
    uploaded_file = st.file_uploader(
        "Faça o upload de uma imagem de ressonância magnética",
        type=["jpg", "png", "jpeg"],
        help="Formatos suportados: JPG, PNG, JPEG",
    )

    if uploaded_file is not None:
        try:
            # Criar colunas para organizar o layout
            col1, col2 = st.columns(2)

            with col1:
                # Mostrar a imagem carregada
                img = Image.open(uploaded_file)
                st.image(img, caption="Imagem carregada", use_column_width=True)

            with col2:
                # Realizar a classificação
                result = predict_tumor(img, model)

                if result is not None:
                    classes = ["Glioma", "Meningioma", "Pituitário", "Sem Tumor"]
                    pred_class = np.argmax(result)

                    # Exibir resultado principal
                    st.markdown("### Resultado da Análise")
                    st.markdown(f"**Classificação:** {classes[pred_class]}")

                    # Exibir probabilidades
                    st.markdown("### Probabilidades por Classe")
                    display_probabilities(result, classes)

                    # Adicionar nota de aviso
                    st.info(
                        """
                    ⚠️ Nota: Este é um sistema de suporte à decisão e não deve ser usado
                    como única fonte para diagnóstico. Sempre consulte um profissional de saúde qualificado.
                    """
                    )

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar a imagem: {str(e)}")
            st.info("Por favor, tente novamente com outra imagem.")

    # Adicionar informações adicionais
    with st.expander("ℹ️ Sobre o Classificador"):
        st.markdown(
            """
        Este classificador foi treinado para identificar quatro categorias diferentes:
        - **Glioma**: Um tipo de tumor que ocorre no cérebro e na medula espinhal
        - **Meningioma**: Tumor geralmente benigno que surge nas meninges
        - **Pituitário**: Tumor que se desenvolve na glândula pituitária
        - **Sem Tumor**: Imagens sem presença de tumores
        
        O modelo utiliza deep learning e foi treinado com milhares de imagens de ressonância magnética.
        """
        )


if __name__ == "__main__":
    main()
