from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

"""
C1: LoRA on mlp.c_fc + mlp.c_proj in block 11 only
"""

# =========================================
# Load separate base model and LoRA adapter
# =========================================

BASE_MODEL = "openai-community/gpt2"
ADAPTER_DIR = "./gpt2-childes-lora-c1"
MERGED_DIR = "./gpt2-childes-lora-c1-merged"

base_model =AutoModelForCausalLM.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

####################################################
# now, we can inspect modules and do inference ... #
#   - LoRA modules                                 #
#   - LoRA matrices/tensors shapes                 #
#   - LoRA configuration                           #
#       r, alpha, dropout, target_moduels          #
#   - direct access LoRA module                    #
####################################################

# inpsect
model
# eye-ball lora related modules @ c_fc & c_proj @ the 11th, i.e. the last, GPT2Block
"""
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-10): 11 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): lora.Linear(
                (base_layer): Conv1D()
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=768, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (c_proj): lora.Linear(
                (base_layer): Conv1D()
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=768, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )
  )
)

"""
# this will expand all blocks/modules
print(model)

# show lora-related modules only
for name, module in model.named_modules():
    if "lora" in name.lower():
        print(name, "@", type(module))

"""
base_model.model.transformer.h.11.mlp.c_fc.lora_dropout @ <class 'torch.nn.modules.container.ModuleDict'>
base_model.model.transformer.h.11.mlp.c_fc.lora_dropout.default @ <class 'torch.nn.modules.dropout.Dropout'>
base_model.model.transformer.h.11.mlp.c_fc.lora_A @ <class 'torch.nn.modules.container.ModuleDict'>
base_model.model.transformer.h.11.mlp.c_fc.lora_A.default @ <class 'torch.nn.modules.linear.Linear'>
base_model.model.transformer.h.11.mlp.c_fc.lora_B @ <class 'torch.nn.modules.container.ModuleDict'>
base_model.model.transformer.h.11.mlp.c_fc.lora_B.default @ <class 'torch.nn.modules.linear.Linear'>
base_model.model.transformer.h.11.mlp.c_fc.lora_embedding_A @ <class 'torch.nn.modules.container.ParameterDict'>
base_model.model.transformer.h.11.mlp.c_fc.lora_embedding_B @ <class 'torch.nn.modules.container.ParameterDict'>
base_model.model.transformer.h.11.mlp.c_proj.lora_dropout @ <class 'torch.nn.modules.container.ModuleDict'>
base_model.model.transformer.h.11.mlp.c_proj.lora_dropout.default @ <class 'torch.nn.modules.dropout.Dropout'>
base_model.model.transformer.h.11.mlp.c_proj.lora_A @ <class 'torch.nn.modules.container.ModuleDict'>
base_model.model.transformer.h.11.mlp.c_proj.lora_A.default @ <class 'torch.nn.modules.linear.Linear'>
base_model.model.transformer.h.11.mlp.c_proj.lora_B @ <class 'torch.nn.modules.container.ModuleDict'>
base_model.model.transformer.h.11.mlp.c_proj.lora_B.default @ <class 'torch.nn.modules.linear.Linear'>
base_model.model.transformer.h.11.mlp.c_proj.lora_embedding_A @ <class 'torch.nn.modules.container.ParameterDict'>
base_model.model.transformer.h.11.mlp.c_proj.lora_embedding_B @ <class 'torch.nn.modules.container.ParameterDict'>
"""

# show lora-related matrices shapes
for name, param in model.named_parameters():
    if "lora_" in name:
        print(name, "@", tuple(param.shape), param.requires_grad)
"""
base_model.model.transformer.h.11.mlp.c_fc.lora_A.default.weight @ (8, 768) False
base_model.model.transformer.h.11.mlp.c_fc.lora_B.default.weight @ (3072, 8) False
base_model.model.transformer.h.11.mlp.c_proj.lora_A.default.weight @ (8, 3072) False
base_model.model.transformer.h.11.mlp.c_proj.lora_B.default.weight @ (768, 8) False
"""

# inspect LoRA configuration itself
print(model.peft_config)
"""#formatted
{
  'default': LoraConfig(peft_type=<PeftType.LORA: 'LORA'>,
  auto_mapping=None,
  base_model_name_or_path='openai-community/gpt2',
  revision=None,
  task_type='CAUSAL_LM',
  inference_mode=True,
  r=8,
  target_modules={
    'transformer.h.11.mlp.c_fc',
    'transformer.h.11.mlp.c_proj'
  },
  lora_alpha=16,
  lora_dropout=0.05,
  fan_in_fan_out=True,
  bias='none',
  use_rslora=False,
  modules_to_save=None,
  init_lora_weights=True,
  layers_to_transform=None,
  layers_pattern=None,
  rank_pattern={
    
  },
  alpha_pattern={
    
  },
  megatron_config=None,
  megatron_core='megatron.core',
  loftq_config={
    
  },
  use_dora=False,
  layer_replication=None)
}
"""
#pretty print
import pprint
pprint.pprint(model.peft_config)
"""
{'default': LoraConfig(peft_type=<PeftType.LORA: 'LORA'>,
                       auto_mapping=None,
                       base_model_name_or_path='openai-community/gpt2',
                       revision=None,
                       task_type='CAUSAL_LM',
                       inference_mode=True,
                       r=8,
                       target_modules={'transformer.h.11.mlp.c_fc',
                                       'transformer.h.11.mlp.c_proj'},
                       lora_alpha=16,
                       lora_dropout=0.05,
                       fan_in_fan_out=True,
                       bias='none',
                       use_rslora=False,
                       modules_to_save=None,
                       init_lora_weights=True,
                       layers_to_transform=None,
                       layers_pattern=None,
                       rank_pattern={},
                       alpha_pattern={},
                       megatron_config=None,
                       megatron_core='megatron.core',
                       loftq_config={},
                       use_dora=False,
                       layer_replication=None)}
"""
# insepct default adapter
config = model.peft_config["default"]

print("r:", config.r)
print("alpha:", config.lora_alpha)
print("dropout:", config.lora_dropout)
print("target modules:", config.target_modules)
"""
r: 8
alpha: 16
dropout: 0.05
target modules: {'transformer.h.11.mlp.c_fc', 'transformer.h.11.mlp.c_proj'}

"""

# understand how LoRA weights are merged into those of the target_modules

lora_c_fc = model.base_model.model.transformer.h[11].mlp.c_fc
# lora related properties in lora_c_fc module
for p in dir(lora_c_fc):
     if "lora_" in p.lower():
         print(p)
"""
lora_A
lora_B
lora_alpha
lora_dropout
lora_embedding_A
lora_embedding_B
lora_magnitude_vector
reset_lora_parameters
"""

#lora_c_fc.reset_lora_parameters
""" 
<bound method LoraLayer.reset_lora_parameters of lora.Linear(
  (base_layer): Conv1D()
  (lora_dropout): ModuleDict(
    (default): Dropout(p=0.05, inplace=False)
  )
  (lora_A): ModuleDict(
    (default): Linear(in_features=768, out_features=8, bias=False)
  )
  (lora_B): ModuleDict(
    (default): Linear(in_features=8, out_features=3072, bias=False)
  )
  (lora_embedding_A): ParameterDict()
  (lora_embedding_B): ParameterDict()
)>
"""

A = lora_c_fc.lora_A["default"].weight
B = lora_c_fc.lora_B["default"].weight

print("A:", A.shape)
#A: torch.Size([8, 768])

print("B:", B.shape)
#B: torch.Size([3072, 8])

delta_W = B @ A

print("delta_W:", delta_W.shape)
#delta_W: torch.Size([3072, 768])

#base layer 
lora_c_fc.base_layer
#Conv1D()
lora_c_fc.base_layer.weight.shape
#torch.Size([768, 3072])

#merged weights W' = W + alpha/r * B @ A, where alpha=16, r=8, alpha / r = 2
#so, LoRA contribution is scaled accordingly


###################################
# save merged model and tokenizer #
#################################=#

merged_model = model.merge_and_unload()

merged_model.save_pretrained(MERGED_DIR)
#-rw-r--r--  1 myaccount  staff   924B Sep 19 14:46 config.json
#-rw-r--r--  1 myaccount  staff   119B Sep 19 14:46 generation_config.json
#-rw-------  1 myaccount  staff   475M Sep 19 14:46 model.safetensors

tokenizer.save_pretrained(MERGED_DIR)
#('./gpt2-childes-lora-c1-merged/tokenizer_config.json',
# './gpt2-childes-lora-c1-merged/special_tokens_map.json',
# './gpt2-childes-lora-c1-merged/vocab.json',
# './gpt2-childes-lora-c1-merged/merges.txt',
# './gpt2-childes-lora-c1-merged/added_tokens.json',
# './gpt2-childes-lora-c1-merged/tokenizer.json')

# ========================================
# Compare base and adapted weights
# ========================================
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
peft_model = PeftModel.from_pretrained(
     base_model,
     ADAPTER_DIR,
 )

base_weight = (
     base_model.
     transformer.
     h[11].
     mlp.
     c_fc.
     weight
)

lora_layer = (
     peft_model.
     base_model.
     model.
     transformer.
     h[11].
     mlp.
     c_fc
)


A = lora_layer.lora_A["default"].weight
B = lora_layer.lora_B["default"].weight

delta_W = (lora_layer.scaling["default"]) * (B @ A)

print("base weight:", base_weight.shape)
#base weight: torch.Size([768, 3072])

print("LoRA delta:", delta_W.shape)
#LoRA delta: torch.Size([3072, 768])

"""#as stated from above
                 c_fc

       pretrained W
       3072 × 768
            │
            │ frozen
            │
            ├───────────────┐
            │               │
            │          LoRA delta
            │               │
            │      alpha  / r * (B @ A)
            │               │
            └───────┬───────┘
                    +
                    |
                    ▼
             effective W'
"""
# trainable parameter count, this should be done at time of training (see code there)
# model = get_peft_model(model, lora_config)
# print("\nTrainable parameters:")
# model.print_trainable_parameters()
"""where 61440 = 2 x (3072 x 8+768 x 8) in LoRA module and all base weights are frozen! 
Trainable parameters:
trainable params: 61,440 || all params: 124,501,248 || trainable%: 0.0493
"""

# alternatively, this is how we can calculate this ratio manually at time of trianing
#total_params = sum(
#    p.numel() for p in model.parameters()
#)

#trainable_params = sum(
#    p.numel()
#    for p in model.parameters()
#    if p.requires_grad
#)

#print("Total:", total_params)
#print("Trainable:", trainable_params)
#print("Percentage:", 100 * trainable_params / total_params) #this should give us the same numbers as above


# confirm every non-LoRA parameter is frozen, which should be done in training code
# trainable = []
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        trainable.append(name)
#print("\n".join(trainable))
#You should see only LoRA parameters.

# or verify in this way at time of training (see training code)
# ============================================================
# Verify exactly which parameters are trainable
# ============================================================
#print("\nTrainable parameter names:")
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(f"  {name}  {tuple(param.shape)}")
"""
Trainable parameter names:
  base_model.model.transformer.h.11.mlp.c_fc.lora_A.default.weight  (8, 768)
  base_model.model.transformer.h.11.mlp.c_fc.lora_B.default.weight  (3072, 8)
  base_model.model.transformer.h.11.mlp.c_proj.lora_A.default.weight  (8, 3072)
  base_model.model.transformer.h.11.mlp.c_proj.lora_B.default.weight  (768, 8)
"""

# =========================================
#  Load merged SFT model
# =========================================
model = AutoModelForCausalLM.from_pretrained(
    "./gpt2-childes-lora-c1-merged"
)
tokenzier = AutoTokenizer.from_pretrained(BASE_MODEL)


# ====================================================
# compare base model inference to SFT model using LoRA
# ====================================================

prompt = "Adult: waht did you do today?\nChild:"

inputs = tokenizer(
    prompt,
    return_tensors='pt',
)

# generation with merged model: SFT w/ LoRA
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
"""
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Adult: waht did you do today?
Child: the ones in the van.<|endoftext|>
"""

# generation with base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
base_model.config.pad_token_id = tokenizer.pad_token_id
#base_model.eval()

with torch.no_grad():
    output_ids = base_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
"""
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Adult: waht did you do today?
Child: hmm no. I don't know if he got out of the car, the driver was in the car, but I'm sure they'd never seen him.
S: Yeah, but that's just him.
Child: i am sorry
"""
