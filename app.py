import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

# Configurar la app
st.title("🚀 Aplicación de IA Generativa en Español con GPT-2")

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

# Función para entrenar el modelo
def train_model(dataset_path="sample_text.txt", output_dir="gpt2_finetuned"):
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

# Botón para entrenar el modelo
if st.button("Entrenar Modelo"):
    # Crear un dataset de muestra
    sample_text = """Había una vez un mundo donde la inteligencia artificial escribía historias increíbles.
    En ese mundo, la gente usaba la IA para crear literatura, noticias y aventuras.
    """
    
    with open("sample_text.txt", "w") as f:
        f.write(sample_text)

    train_model()

# Interfaz para ingresar un prompt y generar texto
st.subheader("📝 Generación de Texto")
prompt = st.text_area("Introduce tu prompt:", "Había una vez un mundo futurista...")

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

