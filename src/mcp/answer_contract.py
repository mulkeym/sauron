"""Keep document answering distinct from host-side user clarification tools."""
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware

ANSWER_TOOL = "tool_answer_from_documents"
LEGACY_ANSWER_TOOL = "tool_ask"


class AnswerToolContract(Middleware):
    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        # Existing integrations can still invoke the old name, but fresh tool
        # catalogs advertise only the unambiguous document-answering name.
        return [tool for tool in tools if tool.name != LEGACY_ANSWER_TOOL]

    async def on_call_tool(self, context, call_next):
        if context.message.name in {ANSWER_TOOL, LEGACY_ANSWER_TOOL}:
            arguments = context.message.arguments or {}
            question = arguments.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ToolError(
                    "Sauron answers questions from stored documents; this is not a user-clarification tool. "
                    "Retry tool_answer_from_documents with a nonempty question string containing the user's original request: "
                    '{"question": "<copy the user request here>"}. '
                    "Do not send questions, options, or timeout_ms. Search the documents before deciding whether "
                    "clarification is needed. No search ran because the question was missing or invalid."
                )
        return await call_next(context)
