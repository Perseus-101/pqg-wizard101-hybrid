import os
import json
import jsonschema
from pathlib import Path

# --- 1. IMPORTS FROM THE MAIN GENERATION SCRIPT ---
try:
    from main import get_llm, generate_quest_in_sequence, generate_baseline_quest
except ImportError:
    print("FATAL: Could not import from main.py. Make sure it's in the same directory.")
    exit()

# --- 2. CONFIGURATION FOR SURVEY GENERATION ---
SYSTEMS_TO_TEST = {
    "advanced_system": "gpt-4o",
    "baseline_system": "gpt-4o"
}
NUM_QUESTS_TO_GENERATE = 5 # Increased to run all scenarios
OUTPUT_DIR = "human_survey_outputs"

# --- 3. WIZARD101-ACCURATE PLAYER SCENARIOS & TEMPLATE ---
# This data includes both main storyline quests and more unique, complex side quests.
PLAYER_SCENARIOS = [
    {
        "id": "prompt_01",
        "level": 3,
        "location": "The Commons",
        "last_quest_summary": "The player has just enrolled in Ravenwood after meeting Headmaster Ambrose.",
        "trigger": "Headmaster Ambrose is concerned about a disturbance in Unicorn Way and asks the player to speak with Private Stillson to gain entry and investigate."
    },
    {
        "id": "prompt_02",
        "level": 6,
        "location": "Unicorn Way",
        "last_quest_summary": "The player has helped Ceren Nightchant and Lady Oriel by defeating several of Rattlebones' undead minions.",
        "trigger": "Lady Oriel reveals that the source of the corruption is the skeletal mastermind, Rattlebones, and asks the player to enter his tower and defeat him."
    },
    {
        "id": "prompt_03",
        "level": 8,
        "location": "The Commons",
        "last_quest_summary": "The player has defeated Rattlebones, but Ambrose is now worried about General Foulwind's cyclops army.",
        "trigger": "Headmaster Ambrose directs the player to Cyclops Lane to assist the wizard guard, General Akilles, with the cyclops threat."
    },
    {
        "id": "prompt_04",
        "level": 12,
        "location": "Triton Avenue",
        "last_quest_summary": "The player has dealt with the major threats on the main streets of Wizard City.",
        "trigger": "Ambrose reveals his former student, Malistaire, is behind the city's troubles and has sent his agent, Lord Nightshade, to the Haunted Cave to steal a powerful artifact. The player must stop him."
    },
    {
        "id": "prompt_05",
        "level": 15,
        "location": "The Commons",
        "last_quest_summary": "The player has successfully defeated Lord Nightshade, dealing a major blow to Malistaire's plans.",
        "trigger": "As a new challenge, Headmaster Ambrose suggests the player seek out Prospector Zeke, who has a special exploration task: finding the lost Smiths of Wizard City, starting with the one in the Commons."
    }
]

def format_prompt_from_scenario(scenario: dict) -> str:
    """
    Formats a structured player scenario into a consistent, natural language prompt.
    This mimics how a game engine would request a quest, providing a standardized
    but dynamic input for the LLM.
    """
    # This template ensures every prompt sent to the LLM has the exact same structure.
    prompt_template = (
        "Generate a quest for a level {level} player, whose last known location was {location}. "
        "The player's most recent accomplishment was: '{last_quest_summary}'. "
        "The specific task for this new quest is: {trigger}"
    )
    return prompt_template.format(**scenario)


# --- 4. LOAD SCHEMA FOR VALIDATION ---
try:
    with open('./knowledge_base/wizard101_quest_schema.json', 'r', encoding='utf-8') as f:
        QUEST_SCHEMA = json.load(f)
    print("Successfully loaded quest schema for validation.")
except FileNotFoundError as e:
    print(f"FATAL: A required knowledge base file was not found. Error: {e}")
    exit()

# --- 5. SCHEMA VALIDATION FUNCTION ---
def validate_schema(quest_json: dict) -> (bool, str):
    """Validates the quest against the JSON schema."""
    if "error" in quest_json:
        return False, f"Generation Error: {quest_json.get('details', 'Unknown')}"
    try:
        jsonschema.validate(instance=quest_json, schema=QUEST_SCHEMA)
        return True, "Valid"
    except jsonschema.exceptions.ValidationError as e:
        return False, f"Schema Error on property '{'.'.join(str(p) for p in e.path)}': {e.message}"
    except Exception as e:
        return False, f"Unknown validation error: {str(e)}"

# --- 6. MAIN GENERATION ORCHESTRATOR ---
def run_generation_for_survey():
    """Runs the generation process for all defined systems and prompts."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for system_name, model_id in SYSTEMS_TO_TEST.items():
        print("\n" + "="*60)
        print(f"--- GENERATING QUESTS FOR: {system_name.upper()} ---")
        print(f"--- Using Model: {model_id} ---")
        print("="*60)
        
        system_output_dir = Path(OUTPUT_DIR) / system_name
        system_output_dir.mkdir(exist_ok=True)
        llm = get_llm(model_id)
        
        for i, scenario_data in enumerate(PLAYER_SCENARIOS):
            if i >= NUM_QUESTS_TO_GENERATE: break
            
            prompt_id = scenario_data["id"]
            prompt_text = format_prompt_from_scenario(scenario_data)
            
            print(f"\n--- Generating Quest {i+1}/{NUM_QUESTS_TO_GENERATE} (ID: {prompt_id}) ---")
            print(f"    Formatted Prompt: \"{prompt_text}\"")

            quest_json = {}
            if system_name == "advanced_system":
                initial_state = {
                    "last_quest_summary": scenario_data["last_quest_summary"],
                    "player_level_estimate": scenario_data["level"]
                }
                quest_json, _ = generate_quest_in_sequence(prompt_text, initial_state, llm)
            elif system_name == "baseline_system":
                quest_json, _ = generate_baseline_quest(prompt_text, llm)
            
            output_path = system_output_dir / f"{prompt_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(quest_json, f, indent=4)
            
            is_valid, schema_msg = validate_schema(quest_json)
            print(f"  - Schema Valid: {is_valid} ({schema_msg})")
    
    print("\n" + "="*60)
    print("All quest generation for the survey is complete.")
    print(f"Output saved in: '{OUTPUT_DIR}'")
    print("="*60)

if __name__ == '__main__':
    run_generation_for_survey()
