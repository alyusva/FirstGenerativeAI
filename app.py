import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os
import PyPDF2  

# Configurar la aplicación
st.title("🚀 Mi primera aplicación de IA Generativa en Español con GPT-2")

# Cargar el modelo en español
MODEL_NAME = "DeepESP/gpt2-spanish"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_path = "gpt2_finetuned"

if os.path.exists(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cpu")
    st.write("✅ **Modelo cargado desde el entrenamiento previo.**")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
    st.write("🔄 **Modelo base en español cargado.**")

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# **Sección para subir archivos PDF o TXT**
st.subheader("📂 Subir un archivo de texto para entrenar el modelo")

uploaded_file = st.file_uploader("Sube un archivo PDF o TXT", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.getvalue().decode("utf-8")

    with open("training_text.txt", "w") as f:
        f.write(text)

    st.success("✅ Archivo cargado y listo para el entrenamiento.")

# Función para entrenar el modelo
def train_model(dataset_path="training_text.txt", output_dir="gpt2_finetuned"):
    st.write("⚡ Entrenando modelo... Esto puede tardar unos minutos.")

    # Crear dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128
    )

    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=1,
        prediction_loss_only=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=AutoModelForCausalLM.from_pretrained(MODEL_NAME),
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(output_dir)
    st.success("🎉 ¡Entrenamiento completado! Modelo guardado en 'gpt2_finetuned'.")

# Botón para entrenar el modelo solo si hay un archivo cargado
if uploaded_file is not None and st.button("Entrenar Modelo"):
    train_model()

# **Sección de generación de texto**
st.subheader("📝 Generación de Texto")
prompt = st.text_area("Introduce tu prompt:", "Había una vez en un mundo futurista...")

# Función para generar texto
def generate_text(prompt, model, tokenizer, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,  # Controla la creatividad
        top_k=top_k,              # Filtra palabras poco probables
        top_p=top_p,              # Generación más coherente
        repetition_penalty=1.2,    # Evita repeticiones
        do_sample=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Botón para generar texto
if st.button("Generar Texto"):
    with st.spinner("✍️ Generando texto..."):
        generated_text = generate_text(prompt, model, tokenizer)
        st.write("### ✨ Texto generado:")
        st.write(generated_text)
