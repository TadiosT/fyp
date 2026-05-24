import unittest
from datetime import datetime
from simulator import process_tick


class TestSimulatorTTL(unittest.TestCase):

    def setUp(self):
        """Sets up a clean slate of data before every single test."""
        self.current_time = datetime(2026, 4, 1, 14, 0, 0)
        self.available_users = ["AnonUser_0003", "AnonUser_0004"]

        # A mocked dictionary of PCs to feed into the function
        self.state_memory = {
            "Lab_202_PC001": {"state": "In Use", "user_id": "AnonUser_0001", "ttl": 5},
            "Lab_202_PC002": {"state": "Idle", "user_id": "AnonUser_0002", "ttl": 1},  # About to expire!
        }

    def test_ttl_decrements_normally(self):
        """Test that a TTL of 5 drops to 4 and user stays active."""
        process_tick(self.state_memory, self.current_time, self.available_users)

        pc_data = self.state_memory["Lab_202_PC001"]
        self.assertEqual(pc_data["ttl"], 4)
        self.assertEqual(pc_data["user_id"], "AnonUser_0001")
        self.assertIn(pc_data["state"], ["In Use", "Idle"])

    def test_ttl_expires_correctly(self):
        """Test that a TTL of 1 drops to 0, kicks user out, and returns them to pool."""
        # PC002 starts with a TTL of 1 and AnonUser_0002
        process_tick(self.state_memory, self.current_time, self.available_users)

        pc_data = self.state_memory["Lab_202_PC002"]

        # 1. PC Memory should be wiped
        self.assertEqual(pc_data["ttl"], 0)
        self.assertEqual(pc_data["user_id"], "N/A")
        self.assertEqual(pc_data["state"], "Offline")

        # 2. User should be returned to the available pool
        self.assertIn("AnonUser_0002", self.available_users)

    def test_csv_output_format_on_expiry(self):
        """Test that the generated CSV rows perfectly reflect the expiry."""
        new_entries, _ = process_tick(self.state_memory, self.current_time, self.available_users)

        # Find the row generated for PC002
        pc002_row = next(row for row in new_entries if row["pc_id"] == "Lab_202_PC002")

        self.assertEqual(pc002_row["state"], "Offline")
        self.assertEqual(pc002_row["user_id"], "N/A")
        self.assertEqual(pc002_row["session_ttl_remaining"], "N/A")


if __name__ == '__main__':
    unittest.main()