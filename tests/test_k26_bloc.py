import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from products.coalition_simulator.model import k26_bloc


class K26BlocTests(unittest.TestCase):
    def dimension(self):
        rows = [{"ballot_symbol": symbol, "party_id": name, "bloc_core": "Netanyahu bloc"}
                for symbol, name in k26_bloc.canonical_names().items()]
        rows.append({"ballot_symbol": "דרך", "party_id": "Yashar", "bloc_core": "Other"})
        return pd.DataFrame(rows)

    def test_confirmed_minor_lists_are_coalition(self):
        self.assertTrue({"ני", "נר", "זך", "בי"} <= k26_bloc.coalition_symbols())
        self.assertEqual(k26_bloc.canonical_names()["נר"], "Tzeva Shachor")
        k26_bloc.validate_dimension(self.dimension())

    def test_stale_minor_list_assignment_is_rejected(self):
        for symbol in ["ני", "נר", "זך", "בי"]:
            frame = self.dimension()
            frame.loc[frame.ballot_symbol.eq(symbol), "bloc_core"] = "Other"
            with self.assertRaisesRegex(ValueError, "differs from the bloc contract"):
                k26_bloc.validate_dimension(frame)

    def test_hash_tracks_policy_not_line_endings(self):
        contract = k26_bloc.load_contract()
        original = k26_bloc.contract_sha256()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "contract.json"
            path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=4).replace("\n", "\r\n").encode())
            with patch.object(k26_bloc, "CONTRACT_PATH", path):
                self.assertEqual(k26_bloc.contract_sha256(), original)
                contract["coalition_parties"] = [row for row in contract["coalition_parties"] if row["symbol"] != "ני"]
                path.write_text(json.dumps(contract), encoding="utf-8")
                self.assertNotEqual(k26_bloc.contract_sha256(), original)
