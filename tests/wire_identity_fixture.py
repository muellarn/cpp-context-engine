"""Small real fixture for wire identity, external storage, and capped points-to."""

import json
from pathlib import Path


def create_wire_identity_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "external.hpp").write_text(
        "extern int external_storage;\n"
        "struct ExternalRecord { int field; };\n"
        "extern ExternalRecord external_record;\n",
        encoding="utf-8",
    )
    project = directory / "project"
    project.mkdir()
    declarations = "\n".join(f"int storage_{index};" for index in range(70))
    branches = "\n".join(f"case {index}: pointer = &storage_{index}; break;" for index in range(70))
    (project / "identity.cpp").write_text(
        '#include "../external.hpp"\n'
        "namespace wire_identity_long_namespace_for_repeatable_scope {\n"
        f"{declarations}\n"
        "int capped_external_and_fields(int choice) {\n"
        "int *pointer = &external_storage;\n"
        f"switch (choice) {{ {branches} }}\n"
        "*pointer = external_record.field;\n"
        "external_record.field = *pointer;\n"
        "return external_storage + external_record.field;\n}\n}\n",
        encoding="utf-8",
    )
    (project / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": ".",
                    "file": "identity.cpp",
                    "arguments": ["clang++", "-std=c++20", "-c", "identity.cpp"],
                }
            ]
        ),
        encoding="utf-8",
    )
    return project
