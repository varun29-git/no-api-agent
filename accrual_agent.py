import mlx_lm
import json
import sqlite3
import re
from mlx_lm import load, generate

# 1. Load the Sovereign Brain (Gemma 3n E2B)
model_path = "mlx-community/gemma-3-4b-it-4bit"
model, tokenizer = load(model_path)

def process_transaction(raw_text):
    # System Prompt with 'Pre-fill' to force JSON output
    prompt = f"""<start_of_turn>user
You are the logic engine for Accrual. Extract data into JSON.
Format: {{"name": "string", "total": int, "paid": int, "udhaar": int}}
Text: {raw_text}<end_of_turn>
<start_of_turn>model
{{"""

    # We start with '{' in the prompt to anchor the model
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
    full_str = "{" + response.strip()
    
    # Regex Shield to find the JSON block
    match = re.search(r'(\{.*\})', full_str, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            save_to_ledger(data)
            return data
        except:
            print(f"❌ JSON Parse Error. Raw: {full_str}")
    else:
        print("❌ No JSON found.")
    return None

def save_to_ledger(data):
    conn = sqlite3.connect('accrual_ledger.db')
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO customers (name) VALUES (?)", (data['name'],))
    cursor.execute("UPDATE customers SET total_udhaar = total_udhaar + ? WHERE name = ?", 
                   (data['udhaar'], data['name']))
    
    cursor.execute("SELECT id FROM customers WHERE name = ?", (data['name'],))
    c_id = cursor.fetchone()[0]
    cursor.execute("INSERT INTO transactions (customer_id, amount, note) VALUES (?, ?, ?)",
                   (c_id, data['udhaar'], f"Purchased items. Paid {data['paid']}"))
    
    conn.commit()
    conn.close()
    print(f"✅ Accrual complete. {data['name']}'s ledger updated by ₹{data['udhaar']}.")

if __name__ == "__main__":
    print("\n--- Accrual Autonomous Agent Active ---")
    user_input = input("Enter transaction (e.g. Suresh bought milk for 50, paid 20): ")
    process_transaction(user_input)