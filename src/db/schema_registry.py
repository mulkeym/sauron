from dataclasses import dataclass, field


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    description: str = ""


@dataclass
class TableSchema:
    database: str
    table: str
    columns: list[ColumnSchema]
    description: str = ""
    acl_groups: list[str] = field(default_factory=list)


class SchemaRegistry:
    def __init__(self):
        self._schemas: dict[str, TableSchema] = {}

    def register(self, schema: TableSchema) -> None:
        key = f"{schema.database}.{schema.table}"
        self._schemas[key] = schema

    def get_schema(self, database: str, table: str) -> TableSchema | None:
        return self._schemas.get(f"{database}.{table}")

    def list_for_user(self, user_groups: list[str]) -> list[TableSchema]:
        return [s for s in self._schemas.values() if any(g in s.acl_groups for g in user_groups)]

    def schemas_to_prompt(self, user_groups: list[str]) -> str:
        schemas = self.list_for_user(user_groups)
        if not schemas:
            return "No database schemas available."
        parts = []
        for s in schemas:
            cols = "\n".join(f"  - {c.name} ({c.dtype}): {c.description}" for c in s.columns)
            parts.append(f"Database: {s.database}\nTable: {s.table}\nDescription: {s.description}\nColumns:\n{cols}")
        return "\n\n".join(parts)
