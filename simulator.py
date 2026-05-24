import os
import pandas as pd
import numpy as np
import time
import random
from datetime import timedelta

DATA_FILE = "data/huxley_mock_access_logs_with_users.csv"
CSV_COLUMNS = ['timestamp', 'pc_id', 'state', 'user_id', 'session_ttl_remaining']


def ensure_csv_header():
    """Recreate the header row if the file is missing or empty."""
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(DATA_FILE, index=False)


def load_initial_state():
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    latest_df = df.sort_values('timestamp').groupby('pc_id').last().reset_index()

    state_memory = {}
    for _, row in latest_df.iterrows():
        ttl_val = row['session_ttl_remaining']
        state_memory[row['pc_id']] = {
            'state': row['state'],
            'user_id': row['user_id'],
            'ttl': int(ttl_val) if ttl_val != 'N/A' else 0
        }

    return state_memory, latest_df['timestamp'].max()


def process_tick(state_memory, current_time, available_users):
    """Processes a single minute of time. Extracted for testability."""
    current_time += timedelta(minutes=1)
    new_entries = []

    for pc_id, data in state_memory.items():
        # 1. Handle Active Sessions
        if data['ttl'] > 0:
            data['ttl'] -= 1

            if data['ttl'] <= 0:
                # TTL Expired: Return user to pool and set PC to Offline
                available_users.append(data['user_id'])
                data['state'] = 'Offline'
                data['user_id'] = 'N/A'
                data['ttl'] = 0
            else:
                data['state'] = np.random.choice(['In Use', 'Idle'], p=[0.90, 0.10])

        # 2. Handle Empty PCs
        else:
            if random.random() < 0.05 and available_users:
                new_user = random.choice(available_users)
                available_users.remove(new_user)

                data['state'] = 'In Use'
                data['user_id'] = new_user
                data['ttl'] = random.randint(15, 120)
            else:
                data['state'] = 'Offline'

        new_entries.append({
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'pc_id': pc_id,
            'state': data['state'],
            'user_id': data['user_id'],
            'session_ttl_remaining': data['ttl'] if data['user_id'] != 'N/A' else 'N/A'
        })

    return new_entries, current_time


def run_simulation():
    state_memory, current_time = load_initial_state()
    total_pcs = len(state_memory)
    all_possible_users = set([f"AnonUser_{i:04d}" for i in range(1, total_pcs + 1)])
    active_users = set([data['user_id'] for data in state_memory.values() if data['user_id'] != 'N/A'])
    available_users = list(all_possible_users - active_users)

    print(f"Simulation started. Resuming from {current_time}.")

    while True:
        try:
            new_entries, current_time = process_tick(state_memory, current_time, available_users)

            ensure_csv_header()
            new_df = pd.DataFrame(new_entries, columns=CSV_COLUMNS)
            try:
                new_df.to_csv(DATA_FILE, mode='a', header=False, index=False)
            except OSError as e:
                print(f"[{current_time.strftime('%H:%M:%S')}] CSV write failed: {e}. Retrying in 5s.")
                time.sleep(5)
                continue

            print(f"[{current_time.strftime('%H:%M:%S')}] Tick completed.")
            time.sleep(60)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_simulation()