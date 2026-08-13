"""检查本轮新增策略模块是否具备用户要求的中文学习说明入口。"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


POLICY_ROOT = Path(__file__).resolve().parents[1]
EXTRA_SCRIPTS = [
    POLICY_ROOT.parents[0] / "scripts" / "build_inventory.py",
    POLICY_ROOT.parents[0] / "scripts" / "freeze_formal_experiment.py",
    POLICY_ROOT.parents[0] / "scripts" / "export_result_tables.py",
    POLICY_ROOT.parents[0] / "scripts" / "render_selected_cases.py",
    POLICY_ROOT.parents[0] / "scripts" / "render_software_replay.py",
    POLICY_ROOT.parents[0] / "scripts" / "select_report_cases.py",
    POLICY_ROOT.parents[0] / "scripts" / "verify_formal_bundle.py",
]


class AdvancedDocumentationTest(unittest.TestCase):
    """验证模块、类和函数都不是无说明黑盒。"""

    def test_new_modules_classes_and_functions_have_docstrings(self):
        """扫描advanced_policy及正式冻结脚本，报告所有缺失docstring的符号。"""
        paths = sorted(POLICY_ROOT.rglob("*.py")) + EXTRA_SCRIPTS
        missing = []
        for path in paths:
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if ast.get_docstring(tree) is None:
                missing.append(f"{path}:module")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node) is None:
                    missing.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
