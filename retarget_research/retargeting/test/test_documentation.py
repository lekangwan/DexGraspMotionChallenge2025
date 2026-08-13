"""自动检查正式重定向Python代码的模块与函数说明是否缺失。

输入：`retargeting/`目录下所有Python源文件。
输出：unittest通过，或列出缺少docstring的模块/函数及行号。
内部逻辑：使用标准库AST读取源码，不真正导入运动学或仿真模块。
作用：防止后续开发遗漏供审查学习使用的输入、输出、逻辑与作用说明。
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


RETARGET_ROOT = Path(__file__).resolve().parents[1]


def find_missing_docstrings(root: Path):
    """查找目录内缺少模块或函数docstring的位置。

    输入：需要递归检查的源码根目录。
    输出：字符串列表，每项包含相对文件、对象名称和代码行号。
    逻辑：解析每个Python文件的AST，读取模块和所有函数的首个字符串。
    作用：为单元测试提供不依赖第三方包的文档完整性结果。
    """
    missing = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(root)
        if ast.get_docstring(tree) is None:
            missing.append(f"{relative}: module docstring")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{relative}:{node.lineno} function {node.name}")
    return missing


class DocumentationTest(unittest.TestCase):
    """正式重定向源码的文档规范测试集合。"""

    def test_all_modules_and_functions_have_docstrings(self):
        """确认每个Python模块和函数都有可供学习的说明入口。

        输入：无测试参数；固定检查整个`retargeting/`目录。
        输出：没有缺失项则通过，否则一次报告完整缺失列表。
        逻辑：调用AST扫描函数，并把所有缺失项合并成一个断言信息。
        作用：让文档遗漏在代码提交前暴露，而不是等人工审查时发现。
        """
        missing = find_missing_docstrings(RETARGET_ROOT)
        self.assertEqual([], missing, "\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()

