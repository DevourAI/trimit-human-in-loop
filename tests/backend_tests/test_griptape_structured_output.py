from trimit.backend.utils import create_structured_agent
from schema import Schema
from griptape.drivers import OpenAiChatPromptDriver
from griptape.engines import JsonExtractionEngine
from griptape.tasks import PromptTask


def extract_using_json_engine(text, schema):
    json_engine = JsonExtractionEngine(
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4o")
    )
    result = json_engine.extract(text, template_schema=schema)
    return [artifact.value for artifact in result.value]


def extract_using_json_mode(text, schema):
    agent = create_structured_agent(schema)
    agent.add_task(PromptTask(text))
    agent.run()
    return agent.task.output.value


if __name__ == "__main__":
    sample_json_text = """
    Alice (Age 28) lives in New York.
    Bob (Age 35) lives in California.
    Jane is Alice's cousin and lives in Vermont.
    Ben is Bob's brother, 10 years older than Bob.
    Jane and Ben have a son named Tom that goes to a college preparatory school in Massachusetts.
    Tom's friend in school is named Jerry.
    Jerry's mom is a lawyer in Rhode Island named Mary.
    Mary has Alice as a client.
    Create a JSON object mapping the named dependencies between all these people, their approximate ages, and where they live.
    """

    relationship_schema = Schema(
        {
            "relationships": [
                {
                    "name": str,
                    "age": int,
                    "location": str,
                    "relationships": [{"name": str, "relationship_type": str}],
                }
            ]
        }
    ).json_schema("RelationshipSchema")
    json_mode_result = extract_using_json_mode(sample_json_text, relationship_schema)
    json_engine_result = extract_using_json_engine(
        sample_json_text, relationship_schema
    )
