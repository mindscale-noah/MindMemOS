"""Shared validation for registered records and query operands."""

from collections.abc import Mapping

from pydantic import JsonValue, TypeAdapter, ValidationError
from sqlalchemy import JSON, and_, true

from .typing import Filter, TableSpec
from .vector import Vector


def validate_record(spec: TableSpec, record: dict, *, partial: bool = False) -> dict:
    """Validate input without dropping unknown fields or filling update defaults.

    Args:
        spec: Registered table definition.
        record: Column names mapped to input values.
        partial: Validate only explicitly supplied update fields.

    Returns:
        Validated Python values suitable for SQLAlchemy.
    """
    model = spec.insert_model
    if not isinstance(record, dict):
        raise ValueError("Records must be dictionaries")
    unknown = record.keys() - (spec.table.c.keys() if partial else model.model_fields.keys())
    if unknown:
        raise ValueError(f"Unknown input fields: {sorted(unknown)}")
    if partial:
        # Validate supplied fields without defaults or whole-record validators.
        values = {
            name: TypeAdapter(model.model_fields[name].rebuild_annotation()).validate_python(value, strict=True)
            if name in model.model_fields
            else value
            for name, value in record.items()
        }
    else:
        parsed = model.model_validate(record, strict=True)
        # Preserve native dates and UUIDs for SQLAlchemy.
        values = parsed.model_dump(mode="python")
    for name, value in list(values.items()):
        column = spec.table.c[name]
        generated = (
            column.default is not None or column.server_default is not None or column is spec.table.autoincrement_column
        )
        if not partial and name not in record and value is None and generated:
            del values[name]
            continue
        if value is None and not column.nullable:
            raise ValueError(f"Column {name!r} cannot be NULL")
        if isinstance(column.type, Vector) and value is not None:
            values[name] = column.type.validate(value)
        elif isinstance(column.type, JSON) and value is not None:
            TypeAdapter(JsonValue).validate_python(value, strict=True)
        elif value is not None:
            TypeAdapter(column.type.python_type).validate_python(value, strict=True)
    if not partial:
        for column in spec.table.c:
            generated = (
                column.default is not None
                or column.server_default is not None
                or column is spec.table.autoincrement_column
            )
            if column.key not in values and not column.nullable and not generated:
                raise ValueError(f"Missing required column: {column.key!r}")
    return values


def validate_batch(spec: TableSpec, records: list[dict], *, partial: bool = False) -> list[dict]:
    """Validate an entire batch before executing SQL.

    Args:
        spec: Registered table definition.
        records: Input records in their original order.
        partial: Whether records contain partial updates.

    Returns:
        Validated records in the same order.
    """
    result = []
    for index, record in enumerate(records):
        try:
            result.append(validate_record(spec, record, partial=partial))
        except (ValueError, TypeError) as exc:
            # Do not echo potentially sensitive input values in error messages.
            if isinstance(exc, ValidationError):
                reason = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            else:
                reason = str(exc)
            raise ValueError(f"Table {spec.name!r}, record {index}: {reason}") from exc
    return result


def column_for(spec: TableSpec, name: str):
    """Resolve a query column without accepting arbitrary SQL identifiers.

    Args:
        spec: Registered table definition.
        name: Column key to resolve.
    """
    if name not in spec.table.c:
        raise ValueError(f"Unknown column {name!r} in table {spec.name!r}")
    return spec.table.c[name]


def filter_clause(spec: TableSpec, filters=None):
    """Compile validated conditions using bound SQL parameters.

    Args:
        spec: Registered table definition.
        filters: Equality mapping or sequence of Filter objects; None means all rows.

    Returns:
        A SQLAlchemy boolean expression.
    """
    if filters is None:
        return true()
    conditions = [Filter(field=k, value=v) for k, v in filters.items()] if isinstance(filters, Mapping) else filters
    expressions = []
    for condition in conditions:
        if not isinstance(condition, Filter):
            raise ValueError("Expected Filter objects or an equality mapping")
        column = column_for(spec, condition.field)
        if isinstance(column.type, Vector):
            raise ValueError("Vector columns cannot be used as scalar filters")
        if condition.op in {"is_null", "not_null"}:
            if condition.value is not None:
                raise ValueError("Null checks do not accept an operand")
            expressions.append(column.is_(None) if condition.op == "is_null" else column.is_not(None))
            continue
        adapter = TypeAdapter(column.type.python_type)
        value = condition.value
        if condition.op == "in":
            if not isinstance(value, (list, tuple)) or any(v is None for v in value):
                raise ValueError("IN requires a sequence of non-null operands")
            expressions.append(column.in_([adapter.validate_python(v, strict=True) for v in value]))
            continue
        if value is None:
            if condition.op not in {"eq", "ne"}:
                raise ValueError("NULL supports only equality or explicit null checks")
        else:
            value = adapter.validate_python(value, strict=True)
        if condition.op == "eq":
            expression = column == value
        elif condition.op == "ne":
            expression = column != value
        elif condition.op == "gt":
            expression = column > value
        elif condition.op == "ge":
            expression = column >= value
        elif condition.op == "lt":
            expression = column < value
        else:
            expression = column <= value
        expressions.append(expression)
    return and_(*expressions) if expressions else true()
