from jinja2 import Environment, FileSystemLoader, BaseLoader
import os


def parse_prompt_template(template_name: str, **render_kwargs):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    template_dir = os.path.join(cur_dir, "../prompt_templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(f"{template_name}.jinja2")
    return template.render(**render_kwargs)


def load_prompt_template_as_string(template_name: str):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    template_dir = os.path.join(cur_dir, "../prompt_templates")
    template_path = os.path.join(template_dir, f"{template_name}.jinja2")
    with open(template_path, "r") as file:
        template_string = file.read()
    return template_string


def load_langchain_prompt_template(template_name: str):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    template_dir = os.path.join(cur_dir, "../prompt_templates")
    with open(f"{template_dir}/{template_name}.langchain", "r") as file:
        template = file.read()
    return template


def render_jinja_string(template: str, **render_kwargs):
    rtemplate = Environment(loader=BaseLoader()).from_string(template)
    return rtemplate.render(**render_kwargs)
