import logging
import os

from llm_studio.app_utils.sections.chat import show_chat_is_running_dialog
from llm_studio.src.utils.logging_utils import initialize_logging

os.environ["MKL_THREADING_LAYER"] = "GNU"

from h2o_wave import Q, app, copy_expando, main, ui  # noqa: F401

from llm_studio.app_utils.handlers import handle
from llm_studio.app_utils.initializers import initialize_app, initialize_client
from llm_studio.app_utils.sections.common import heap_redact, interface

logger = logging.getLogger(__name__)


def on_startup():
    initialize_logging()
    logger.info("STARTING APP")


@app("/", on_startup=on_startup)
async def serve(q: Q):
    """Serving function."""

    # Chat is still being streamed but user clicks on another button.
    # Wait until streaming has been completed,
    # as currently there is no stop streaming functionality implemented.
    if q.client["experiment/display/chat/finished"] is False:
        await show_chat_is_running_dialog(q)
        return

    if not q.app.initialized:
        await initialize_app(q)

    copy_expando(q.args, q.client)

    await initialize_client(q)
    await handle(q)

    if not q.args["experiment/display/chat/chatbot"]:
        await interface(q)

    await heap_redact(q)
    await q.page.save()
