from saral_state import get_active_collection
from saral import prompt_formatter, tokenizer, llm_model

import torch
from saral import (
    ask_saral_query,
    retrieve_relevant_chunks,
    open_and_read_pdf,
    contains_math,
    clean_math_chunk,
    prompt_formatter,
    tokenizer,
)
# 1. (The VAE/ELBO text you provided)
paper_1_context_dicts = [
    {
        "page_number": 4,
        "sentence_chunk": "In the default formulation of the Variational Autoencoder (VAE) [1], we directly maximize the ELBO. This approach is variational, because we optimize for the best q (zx) amongst a family of potential posterior distributions parameterized by . It is called an autoencoder because it is reminiscent of a traditional au toencoder model, where input data is trained to predict itself after undergoing an intermediate bottlenecking representation step."
    },
    {
        "page_number": 4,
        "sentence_chunk": "Eq (zx) log p(x/z)/q(z/x)=Eq (zx)[logp (xz)]-DKL(q (zx) p(z)). the first term measures the reconstruction likelihood of the decoder from our variational distribution this ensures that the learned distribution is modeling effective latents that the original data can be regenerated from. The second term measures how similar the learned variational distribution is to a prior belief held over latent variables."
    },
    {
        "page_number": 4,
        "sentence_chunk": "Minimizing this term encourages the encoder to actually learn a distribution rather than collapse into a Dirac delta function. Maximizing the ELBO is thus equivalent to maximizing its first term and minimizing its second term."
    }
]

# 2. The Query designed to get the specific output format
query = "Summarize the ELBO objective and its two main terms in 3 sentences for a graduate student."

# 3. Generate the Prompt
prompt = prompt_formatter(
    query=query,
    context_items=paper_1_context_dicts,
    audience="graduate students",
    style="technical",
    duration="90s" # "90s" maps to "3 slides/sentences" in your formatter logic
)

# 4. Run the Model
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.3,
        do_sample=True
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)