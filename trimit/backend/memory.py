from griptape.drivers import LocalConversationMemoryDriver
from griptape.memory.structure import ConversationMemory, SummaryConversationMemory
from trimit.app import VOLUME_DIR
import os

# TODO refactor to save memory per user in a database (or maybe just a folder)
MEMORY_FILEPATH = os.path.join(VOLUME_DIR, "memory.json")


def load_memory(summary=True, offset=5, max_runs=5):
    memory_driver = LocalConversationMemoryDriver(file_path=MEMORY_FILEPATH)
    if summary:
        return SummaryConversationMemory(driver=memory_driver, offset=offset)
    return ConversationMemory(driver=memory_driver, max_runs=max_runs)
