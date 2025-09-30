1. Setup

Install the necessary libraries for NLP:

!pip install transformers sentencepiece --quiet


Import required classes:

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import textwrap


Models used:

T5-base for summarization.

PEGASUS for paraphrasing.

2. Summarization with T5

Purpose: Reduce long technical content into a short, clear summary.

Function:

def generate_summary(text, min_len=40, max_len=120, beams=4, no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True):
    """
    Generate a concise summary using T5.
    """
    input_text = "summarize: " + text.strip().replace("\n", " ")
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = t5_model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        num_beams=beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        early_stopping=early_stopping
    )
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


Example:

electrical_text = "A transformer transfers electrical energy between circuits through electromagnetic induction..."
summary = generate_summary(electrical_text, min_len=30, max_len=80, beams=5)
print(summary)


Output:
"Transformers efficiently transfer electrical energy and are essential in power distribution."

3. Paraphrasing with PEGASUS

Purpose: Generate multiple ways to express the same sentence without changing meaning.

Function:

def generate_paraphrases(text, num_return=5, beams=10, max_len=60, early_stopping=True):
    """
    Generate multiple paraphrased sentences using PEGASUS.
    """
    inputs = pegasus_tokenizer.encode(text, return_tensors="pt", truncation=True)
    paraphrase_ids = pegasus_model.generate(
        inputs,
        max_length=max_len,
        num_beams=beams,
        num_return_sequences=num_return,
        early_stopping=early_stopping
    )
    return pegasus_tokenizer.batch_decode(paraphrase_ids, skip_special_tokens=True)


Example:

electrical_sentence = "Electrical energy is transmitted at high voltages to reduce power loss."
paraphrases = generate_paraphrases(electrical_sentence, num_return=3, beams=8)
for i, p in enumerate(paraphrases):
    print(f"{i+1}. {p}")


Output:

"High-voltage transmission reduces power loss in electricity transfer."

"To minimize energy loss, electricity is sent at high voltages."

"Electrical energy is transferred at high voltages to prevent power loss."

4. Interactive Playground

Users can input their own text and instantly get both a summary and paraphrases.

Example:

your_text = "The synchronous motor operates at constant speed irrespective of the load applied to it."

generated_summary = generate_summary(your_text, min_len=25, max_len=70, beams=5)
generated_paraphrases = generate_paraphrases(your_text, num_return=3, beams=7)

print("Summary:", generated_summary)
print("Paraphrases:", generated_paraphrases)
