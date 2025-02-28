# FirstGenerativeAI
My First ever Generative AI model

# 🚀 Mi primera aplicación de IA Generativa en Español con GPT-2

## 📌 Descripción
Esta aplicación permite entrenar un modelo de IA Generativa basado en **GPT-2 en español** y generar texto a partir de un prompt. Además, permite **subir archivos PDF o TXT** para entrenar el modelo con contenido personalizado.

La aplicación está desarrollada en **Python** utilizando la biblioteca **Streamlit** para la interfaz gráfica y la librería **Hugging Face Transformers** para el modelo de lenguaje.

---

## 📦 Instalación
### 1️⃣ **Clonar el repositorio**
```bash
git clone https://github.com/tuusuario/mi-app-ia-generativa.git
cd mi-app-ia-generativa
```

### 2️⃣ **Crear y activar un entorno virtual**
```bash
python3 -m venv venv
source venv/bin/activate  # Para macOS/Linux
# En Windows usa: venv\Scripts\activate
```

### 3️⃣ **Instalar dependencias**
```bash
pip install -r requirements.txt
```

---

## 🚀 Uso
### **Ejecutar la aplicación**
Ejecuta el siguiente comando para iniciar la aplicación en el navegador:
```bash
streamlit run app.py
```

### **Funciones principales**
✅ **Subir un archivo PDF o TXT** para entrenar el modelo.
✅ **Entrenar el modelo** con el contenido proporcionado.
✅ **Ingresar un prompt** y generar texto con el modelo.

---

## 📂 Estructura del Proyecto
```
mi-app-ia-generativa/
│── app.py                 # Código principal de la aplicación
│── requirements.txt       # Lista de dependencias
│── venv/                  # Entorno virtual (excluido en Git)
│── gpt2_finetuned/        # Modelo entrenado (se genera tras entrenar)
```

---

## 📌 Dependencias
- **Python 3.8+**
- **Streamlit** (Interfaz gráfica)
- **Transformers** (Hugging Face para GPT-2)
- **Torch** (PyTorch para entrenar y ejecutar modelos)
- **PyPDF2** (Para extraer texto de PDFs)

Si necesitas instalar una dependencia manualmente, usa:
```bash
pip install nombre-paquete
```

---

## 🛠 Posibles Errores y Soluciones
### 🔴 `ImportError: Using the Trainer with PyTorch requires accelerate>=0.26.0`
✅ Solución:
```bash
pip install --upgrade accelerate
```

### 🔴 `ModuleNotFoundError: No module named 'transformers'`
✅ Solución:
```bash
pip install transformers
```

---

## 📌 Mejoras Futuras
- [ ] Integrar otros modelos de lenguaje como GPT-3.
- [ ] Permitir la descarga del texto generado en un archivo.
- [ ] Implementar almacenamiento de modelos en la nube.

---

## 👨‍💻 Autor
**Alvaro Yuste Valles** - [LinkedIn](www.linkedin.com/in/álvaro-yuste-valles-499a30b3) 

---

## 📝 Licencia
Este proyecto está bajo la licencia **MIT**. ¡Úsalo y mejora el código libremente! 🎉

