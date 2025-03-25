import asyncio
import json
import math
import random
import re
from datetime import datetime
import concurrent.futures
from utils.inference import generate_text

# Global settings for variant generation
MODEL = "deepseek-chat"
TIMEOUT_SECONDS = 1  # Maximum allowed seconds for any task if needed

# Configurable parameters
NUM_VARIANTS = 3          # Number of final variants per difficulty
RECURSION_DEPTH = 0       # Set >0 if you want to generate variants recursively
DIFFICULTIES = ["easier"] # Options could include "easier", "equivalent", "harder"

# Generic transformations by difficulty.
# You can change these lists to suit different types of prompts.
TRANSFORMATIONS_BY_DIFFICULTY = {
    "easier": [
        "simplify the language",
        "reduce the complexity",
        "remove unnecessary details",
        "simplify the instructions",
     
    ],
    "equivalent": [
        "make minor adjustments",
        "change wording slightly",
        "rephrase without altering meaning",
        "substitute synonyms",
    ],
    "harder": [
        "add additional details",
        "increase complexity",
        "introduce more technical language",
        "expand the prompt",
        "add challenging constraints",
    ]
}

# Generic personas that may inspire creative variants.
PERSONAS = [
    "an expert in the field",
    "a creative thinker",
    "a seasoned professional",
    "an enthusiastic beginner",
    "a visionary strategist",
    "a technical specialist",
    "a pragmatic problem-solver"
]

# This function returns a prompt template for the LLM.
def get_random_prompt_template(prompt: str, difficulty: str, count: int, transforms_text: str, personas_str: str) -> str:
    template_options = [
        (
            f"Assume you can adopt various personas such as {personas_str}.\n\n"
            f"Given the prompt/task: {prompt}\n"
            f"Your task is to generate {count} creative variant(s) that are {difficulty} than the original.\n\n"
            "Important constraints:\n"
            "- Maintain the original intent of the prompt.\n"
            "- Avoid introducing arbitrary or irrelevant changes.\n"
            "- All modifications should be specific and meaningful.\n\n"
            "Follow these steps:\n"
            "1. Analyze the original prompt deeply, looking for hidden simplifications and opportunities for improvement.\n"
            "2. Think outside conventional approaches – consider alternative phrasings, simplifications, or restructuring.\n"
            f"3. Draw inspiration from various fields. Some ideas: {transforms_text}\n"
            "4. Provide a detailed explanation of your creative reasoning process.\n"
            "5. Present each variant in the following exact format:\n"
            "====\n"
            "Variant <number>:\n"
            "Reasoning: <your creative chain-of-thought explanation>\n"
            "Variant: <the new prompt variant>\n"
            "====\n\n"
            "Generate truly novel variants that might surprise even experienced practitioners."
        ),
        (
            f"Channel the creative spirit of professionals like {personas_str}.\n\n"
            f"For this task: {prompt}\n"
            f"Create {count} variant(s) that are {difficulty} than the original prompt.\n\n"
            "Key points:\n"
            "- Do not change the core intent of the prompt.\n"
            "- All modifications must be specific and justified.\n\n"
            "Steps:\n"
            "1. Examine the prompt carefully and identify aspects that can be simplified or modified.\n"
            f"2. Experiment with ideas such as: {transforms_text}\n"
            "3. Explain your reasoning in detail.\n"
            "4. Provide your answer in the following exact format:\n"
            "====\n"
            "Variant <number>:\n"
            "Reasoning: <your creative chain-of-thought explanation>\n"
            "Variant: <the new prompt variant>\n"
            "====\n\n"
            "Aim to create variants that reveal new perspectives on the original prompt."
        )
    ]
    return random.choice(template_options)

# This function parses the LLM response to extract variants.
def parse_variants(text: str) -> list:
    variants = []
    blocks = re.split(r"====\s*", text)
    for block in blocks:
        if "Variant:" in block and "Reasoning:" in block:
            reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Variant:", block, re.DOTALL)
            reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
            
            variant_expr = None
            for line in block.splitlines():
                if line.strip().startswith("Variant:"):
                    variant_expr = line.strip()[len("Variant:"):].strip()
                    break
            if variant_expr:
                variants.append({"reasoning": reasoning_text, "variant": variant_expr})
    return variants

# This function calls the LLM (via generate_text) to produce a chunk of variants.
async def generate_variant_chunk(prompt: str, difficulty: str, count: int) -> list:
    transformations = TRANSFORMATIONS_BY_DIFFICULTY.get(difficulty.lower(), ["make a small change"])
    num_choices = random.choice(range(3, 7))
    chosen_transforms = random.sample(transformations, min(num_choices, len(transformations)))
    transforms_text = ", ".join(chosen_transforms)
    personas_str = ", ".join(PERSONAS)
    
    prompt_variant = get_random_prompt_template(prompt, difficulty, count, transforms_text, personas_str)
    temperature_choice = random.choice([0.8, 1.0, 1.2, 1.4])
    response_text = await generate_text(MODEL, prompt_variant, temperature=temperature_choice)
    
    parsed_variants = parse_variants(response_text)
    for variant in parsed_variants:
        variant["transformations_used"] = chosen_transforms
    tasks = [process_single_variant(prompt, difficulty, variant) for variant in parsed_variants]
    processed_variants = await asyncio.gather(*tasks)
    return [v for v in processed_variants if v is not None]

# This function processes a single variant.
async def process_single_variant(original_prompt: str, difficulty: str, variant_data: dict) -> dict:
    variant_prompt = variant_data.get("variant")
    if not variant_prompt:
        return None
    return {
        "original": original_prompt,
        "requested_difficulty": difficulty,
        "variant": variant_prompt,
        "reasoning": variant_data.get("reasoning"),
        "transformations_used": variant_data.get("transformations_used", []),
        "evaluation": None,  # Placeholder for any evaluation metric if needed
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# This function orchestrates the variant generation for a given prompt.
async def process_prompt(base_prompt: str, difficulties: list, num_variants: int = NUM_VARIANTS, recursion_depth: int = RECURSION_DEPTH) -> list:
    final_results = []
    seen_variants = set()
    buffer_multiplier = 3  # Request extra variants to allow for filtering
    tasks = []

    for difficulty in difficulties:
        total_to_request = num_variants * buffer_multiplier
        num_chunks = math.ceil(total_to_request / 10)
        for i in range(num_chunks):
            count = 10 if (i < num_chunks - 1) else (total_to_request - 10 * (num_chunks - 1))
            tasks.append((difficulty, generate_variant_chunk(base_prompt, difficulty, count)))

    chunk_results = await asyncio.gather(*[t[1] for t in tasks])
    difficulty_dict = {d: [] for d in difficulties}
    for idx, (difficulty, _) in enumerate(tasks):
        for variant in chunk_results[idx]:
            variant_expr = variant.get("variant")
            if variant_expr and variant_expr not in seen_variants:
                seen_variants.add(variant_expr)
                difficulty_dict[difficulty].append(variant)

    for difficulty in difficulties:
        final_results.extend(difficulty_dict[difficulty][:num_variants])
    
    # If recursion is enabled, recursively generate variants of the variants.
    if recursion_depth > 0:
        recursive_results = []
        for variant in final_results:
            new_prompt = variant["variant"]
            sub_variants = await process_prompt(new_prompt, difficulties, num_variants, recursion_depth - 1)
            recursive_results.extend(sub_variants)
        final_results.extend(recursive_results)
    
    return final_results

# Main entry point: you can change the base prompt, difficulties, or recursion depth here.
async def main():
    base_prompt = "Describe how to optimize a robotics task for efficiency."
    difficulties = ["easier"]  # Change or add difficulties such as "equivalent" or "harder" if desired
    variants = await process_prompt(base_prompt, difficulties, num_variants=NUM_VARIANTS, recursion_depth=RECURSION_DEPTH)
    
    with open("variants.json", "w") as outfile:
        json.dump(variants, outfile, indent=2)
    
    for idx, v in enumerate(variants, start=1):
        print(f"\n--- Variant {idx} ---")
        print("Requested difficulty:", v["requested_difficulty"])
        print("Transformations used:", v["transformations_used"])
        print("Variant prompt:", v["variant"])
        print("Reasoning:", v["reasoning"])
        print("Evaluation:", v["evaluation"])
        print("Timestamp:", v["timestamp"])

if __name__ == "__main__":
    asyncio.run(main())
