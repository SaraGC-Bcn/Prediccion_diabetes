'''==== ARCHIVO PARA EL DEPLOYMENT EN STREAMLIT PARA LA APLICACIÓN DE PREDICCIÓN DE DIABETES ====
Este archivo contiene el código para el despliegue de la aplicación en Streamlit, llamando al 
pipeline de inference que habíamos preparado. '''

import streamlit as st
import pandas as pd

from src.inference_pipeline import predict_diabetes 


def main():
    #configuración de la página
    st.set_page_config(
        page_title="Predicción de diabetes",
        page_icon="🩺",
        layout="wide"   # o "centered" para ocupar el modo centralizado
    )

    #título y descripción de lo que hace la pagina (titulo coloreado usando markdown HTML)
    st.markdown("<h1 style='color:#00aeef;'>Predicción del riesgo de tener diabetes</h1>", unsafe_allow_html=True)
    st.write(""" Esta aplicación permite conocer el riesgo de un paciente de tener diabetes en un futuro.
            Para conocer el riesgo, es necesario completar los siguientes campos y pulsar el botón de predicción. 
    """)

    #incluimos el subtítulo para introducir los datos a completar por el usuario
    st.markdown("<h2 style='color:#00aeef;'>Datos a completar</h2>", unsafe_allow_html=True)

    #Dado que hemos de recoger datos de 8 variables, organizamos el formulario en 4 columnas con
    # 4 variables cada una para mayor calidad y mejor distribución en la pantalla.
    col1, col2, col3, col4, col5= st.columns([1,3,2,3,3]) #columnas vacías para separar 
    
    #Abrimos cada una de las columnas para colocar los campos correspondientes a cada una de las variables.
    with col2:
        age = st.number_input('Edad (años)', value=0)
        pregnancies = st.number_input('Número de embarazos', value=0)
        bmi = st.number_input('Índice de masa corporal', value=0.0)
        skin_thickness = st.number_input('Grosor del pliegue cutáneo del triceps (mm)', value=0)
        
    with col4:
        blood_pressure = st.number_input('Presión arterial diastólica (la más baja)(mmHg)', value=0)
        glucose = st.number_input('Concentración de glucosa (pasadas 2h)', value=0)
        diabetes_pedigree_function = st.number_input('Función de pedigrí de diabetes', value=0.0)
        insulin = st.number_input('Insulina en serum a las 2h (mu U/ml)', value=0)

    st.write('---') #línea divisoria entre las variables de cada columna

    #incluimos un botón para realizar la predicción una vez el usuario haya introducido los datos
    #Primero: se crea un diccionario con los datos introducidos por el usuario y lo convertimos a un dataframe
    columnas = st.columns([1,2.5,4,3,1.5]) 
    if columnas[2].button('¿Tiene el paciente riesgo de diabetes?', icon = '🔎', icon_position = 'left'):
        try:
            input_data = pd.DataFrame({
                "Pregnancies": [pregnancies], 
                "Glucose": [glucose],
                "BloodPressure": [blood_pressure],
                "SkinThickness": [skin_thickness],
                "Insulin": [insulin],
                "BMI": [bmi],
                "DiabetesPedigreeFunction": [diabetes_pedigree_function],
                "Age": [age]
                })  
                
            #Segundo: con la función de predicción importada del pipeline de inference,
            #obtenemos la predicción y la probabilidad
            label, probability = predict_diabetes(input_data)

            st.write('') #espacio entre el botón y el resultado
            st.write('') #espacio entre el botón y el resultado

            #Tercero: mostramos el resultado al usuario
            col4, col5, col6 = st.columns([1,3,2]) #columna vacía para separar las otras dos
        
            if label =='Diabetes':
                col5.subheader('Es posible que el paciente :red[tenga Diabetes]',divider="red", text_alignment="center")
            else:
                col5.subheader('Es posible que el paciente :green[no tenga Diabetes]', divider="green", text_alignment="center")


            if probability > 0.5:
                col5.subheader(f'Probabilidad de tener diabetes: :red[{probability:.1%}]', text_alignment="center")
            else:
                col5.subheader(f'Probabilidad de tener diabetes: :green[{probability:.1%}]', text_alignment="center")
            
            st.write('---') #línea divisoria entre el resultado y la advertencia sobre la base de datos 
            
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")   

    st.write('') #espacio entre el resultado y la advertencia sobre la base de datos
    st.markdown('''🚨La base de datos con la que se ha entrenado el modelo solo contempla mujeres 
                    mayores de 21 años de ascendencia indígena Pima, por lo que **la predicción puede no ser 
                    tan precisa para otros grupos de población**. Además, si no se completan correctamente todos
                    los campos, también puede disminuir la fiabilidad del modelo. ''', text_alignment="center")
        

if __name__ == "__main__":
    main()
