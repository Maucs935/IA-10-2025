
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

print("="*70)
print("🎓 ENTRENAMIENTO DE TUTOR DE ALGORITMOS")
print("="*70)

# Verificar GPU
if not torch.cuda.is_available():
    print("❌ ERROR: No se detectó GPU")
    exit()

print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}\n")

# 1. CARGAR DATASET
print("📚 Paso 1/6: Cargando dataset...")

if not os.path.exists("dataset_algoritmos.jsonl"):
    print("❌ ERROR: No se encontró dataset_algoritmos.jsonl")
    exit()

dataset = load_dataset('json', data_files='dataset_algoritmos.jsonl')
print(f"   ✓ {len(dataset['train'])} ejemplos cargados")

def format_chat_template(example):
    messages = example['messages']
    formatted_text = ""
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'system':
            formatted_text += f"<|system|>\n{content}\n"
        elif role == 'user':
            formatted_text += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            formatted_text += f"<|assistant|>\n{content}\n<|end|>\n"
    
    return {"text": formatted_text}

dataset = dataset.map(format_chat_template, remove_columns=dataset["train"].column_names)
print("   ✓ Dataset formateado")

# 2. CARGAR MODELO
print("\n📦 Paso 2/6: Cargando modelo (Phi-2)...")
print("   ⏳ 3-5 minutos...\n")

model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

print("   ✓ Modelo cargado")

# 3. CONFIGURAR LORA
print("\n⚙️ Paso 3/6: Configurando LoRA...")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"   ✓ Parámetros entrenables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# 4. TOKENIZAR
print("\n🔤 Paso 4/6: Tokenizando...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("   ✓ Listo")

# 5. CONFIGURAR ENTRENAMIENTO
print("\n🎯 Paso 5/6: Configurando...")

training_args = TrainingArguments(
    output_dir="./resultados",
    num_train_epochs=15,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

print("   ✓ Listo")

# 6. ENTRENAR
print("\n🔥 Paso 6/6: ENTRENANDO")
print("="*70)
print("⏳ Tiempo estimado: 15-25 minutos")
print("="*70 + "\n")

try:
    trainer.train()
    
    print("\n" + "="*70)
    print("✅ ¡COMPLETADO!")
    print("="*70)
    
    output_dir = "./tutor-algoritmos-final"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n💾 Guardado en: {os.path.abspath(output_dir)}")
    print("\n📝 Ejecuta: python probar_tutor.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")