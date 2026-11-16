from typing import Any

import pytest
from test_helpers.tools import addition

from inspect_ai.tool import (
    Content,
    ToolCall,
    ToolCallView,
    ToolDef,
    ToolParam,
    ToolParams,
    tool_with,
)
from inspect_ai.tool._tool_call import ToolCallModelInputHints

NAME = "my_addition"
DESCRIPTION = "my description"
X_DESCRIPTION = "my x description"
Y_DESCRIPTION = "my y description"


def test_tool_with():
    tool = tool_with(
        addition(),
        name=NAME,
        description=DESCRIPTION,
        parameters={"x": X_DESCRIPTION, "y": Y_DESCRIPTION},
    )

    tdef = ToolDef(tool)

    assert tdef.name == NAME
    assert tdef.description == DESCRIPTION
    assert tdef.parameters.properties["x"].description == X_DESCRIPTION
    assert tdef.parameters.properties["y"].description == Y_DESCRIPTION


def test_tool_with_validation():
    try:
        tool_with(
            addition(),
            parameters={"p": X_DESCRIPTION, "q": Y_DESCRIPTION},
        )
        assert False
    except Exception:
        pass


def test_tool_with_name_keeps_tool_def_description() -> None:
    async def execute(**kwargs: Any) -> str:
        return ""

    tool = ToolDef(
        execute,
        name="search",
        description="Search the web.",
        parameters=ToolParams(
            properties={"query": ToolParam(type="string", description="The query.")},
            required=["query"],
        ),
    ).as_tool()

    tdef = ToolDef(tool_with(tool, name=NAME))

    assert tdef.name == NAME
    assert tdef.description == "Search the web."
    assert tdef.parameters.properties["query"].description == "The query."


def test_tool_with_keeps_earlier_tool_with_changes() -> None:
    tool = tool_with(addition(), name=NAME, parameters={"x": X_DESCRIPTION})
    tool = tool_with(tool, description=DESCRIPTION, parameters={"y": Y_DESCRIPTION})

    tdef = ToolDef(tool)

    assert tdef.name == NAME
    assert tdef.description == DESCRIPTION
    assert tdef.parameters.properties["x"].description == X_DESCRIPTION
    assert tdef.parameters.properties["y"].description == Y_DESCRIPTION


def _viewer(call: ToolCall) -> ToolCallView:
    return ToolCallView()


def _model_input(
    message_index: int,
    message_total: int,
    content: str | list[Content],
    hints: ToolCallModelInputHints,
) -> str | list[Content]:
    return content


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parallel": True, "viewer": _viewer},
        {"parallel": True, "model_input": _model_input},
        {"viewer": _viewer, "model_input": _model_input},
        {"parallel": True, "viewer": _viewer, "model_input": _model_input},
    ],
    ids=[
        "parallel+viewer",
        "parallel+model_input",
        "viewer+model_input",
        "all-three",
    ],
)
def test_tool_with_applies_each_attribute(kwargs: dict[str, Any]) -> None:
    tdef = ToolDef(tool_with(addition(), **kwargs))

    assert tdef.parallel == kwargs.get("parallel", False)
    assert tdef.viewer is kwargs.get("viewer")
    assert tdef.model_input is kwargs.get("model_input")


def test_tool_with_applies_parallel_false_with_viewer() -> None:
    tool = tool_with(addition(), parallel=True)
    tdef = ToolDef(tool_with(tool, parallel=False, viewer=_viewer))

    assert tdef.parallel is False
    assert tdef.viewer is _viewer
