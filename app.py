import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

# Configuración de la aplicación
st.title("Aplicación de IA Generativa con GPT-2")

# Cargar el tokenizador y modelo base
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_path = "gpt2_finetuned"

# Función para entrenar el modelo con un dataset personalizado
def train_model(dataset_path="sample_text.txt", output_dir="gpt2_finetuned"):
    st.write("Entrenando modelo... Esto puede tardar unos minutos.")

    # Crear dataset a partir de texto
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
        model=GPT2LMHeadModel.from_pretrained("gpt2"),
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(output_dir)
    st.success("¡Entrenamiento finalizado! El modelo ha sido guardado.")

# Botón para entrenar el modelo
if st.button("Entrenar Modelo"):
    sample_text = """Hola, bienvenido a mi aplicación de IA generativa.
    Esta IA es capaz de generar textos basados en tus entradas.
    Puedes preguntarle cosas o pedirle que escriba historias, chistes, o código.
    """
    
    with open("sample_text.txt", "w") as f:
        f.write(sample_text)

    train_model()

# Cargar el modelo entrenado si existe
if os.path.exists(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path).to("cpu")
else:
    model = GPT2LMHeadModel.from_pretrained("gpt2").to("cpu")

# Interfaz para generar texto
st.subheader("Generación de Texto")
prompt = st.text_area("Introduce tu prompt:", "Érase una vez en un futuro lejano...")

if st.button("Generar Texto"):
    if model:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generación de texto con el modelo
        output = model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("**Texto generado:**")
        st.write(generated_text)
    else:
        st.error("El modelo aún no ha sido entrenado.")

