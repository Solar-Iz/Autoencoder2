# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from our_models import ImprovedConvAutoencoder, preprocess_image

def transforming(image):
    target_size = (440, 440)
    # Преобразование изображений в соответствующий размер (! на всякий случай, выборочно проверяла, вроде одинаковые.. но мало ли!)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    transformed_image = transform(image).usqueeze(0)
    return transforms.ToPILImage()(transformed_image)


def page_preprocessing():
    st.title('Preprocessing')
 # Добавим первый подзаголовок
    st.header('**class ImprovedConvAutoencoder():**')
    st.markdown("""
        Сверточный автоэнкодер, состоящий из четырех сверточных слоев в энкодере и декодере.
        Предназначен для кодирования и восстановления изображений с использованием сверточных и
        транспонированных сверточных операций, а также функций активации ReLU и Sigmoid для
        обеспечения нелинейности и корректного масштабирования значений пикселей.</big>
    """)

    # Загрузим и отобразим изображение архитектуры модели
    st.image('samples/Model.png', caption='Model Architecture')


    st.header('Loss Function')
    
    st.image('samples/4LLoss.png')
    """
    ***Best validation loss: 0.043 at epoch 34***
    """

    st.header('Application of the ImprovedConvAutoencoder() on the test images')

    # Загрузим и отобразим несколько примеров изображений
    example_images = [
        'samples/4Lpred.png',
        'samples/4Lpred_1.png',
        'samples/4Lpred_2.png',
    ]

    for example_image_path in example_images:
        example_image = Image.open(example_image_path)
        st.image(example_image, caption=f'Example: {example_image_path}')

  

def page_clearing():
    st.title('Clear your docs')
    model_cleaning = ImprovedConvAutoencoder()
    model_cleaning.load_state_dict(torch.load('best_weights4L.pth', map_location=torch.device('cpu')))
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with torch.no_grad():
            clean_doc = model_cleaning(preprocess_image(image))
        col3, col4 = st.columns(2)
        with col3:
            st.image(transforming(image), caption='Before')

        with col4:
            st.image(transforms.ToPILImage()(clean_doc.squeeze(0)), caption='After')

# Главная часть приложения
def main():
    st.set_page_config(page_title='Document Cleaning App')

    # Создаем боковое меню навигации
    pages = {'Preprocessing': page_preprocessing, 'Clearing': page_clearing}
    page = st.sidebar.selectbox('Select a page', tuple(pages.keys()))

    # Запускаем выбранную страницу
    pages[page]()

# Запуск приложения
if __name__ == '__main__':
    main()