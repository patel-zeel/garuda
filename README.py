import io
import ast
import jinja2
import contextlib
from README_codes import *
import logging
from garuda.base import logger, log_format

# Read the README codes
with open("README_codes.py", "r") as f:
    readme_codes = ast.parse(f.read())

function_body_dict = {}
output_dict = {}
for node in ast.walk(readme_codes):
    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
        function_code = ast.unparse(node)
        function_body_code = ast.unparse(node.body)
        function_body_dict[node.name] = function_body_code.strip()
        
        # Get the function
        # exec(compile(ast.parse(function_code), filename="<ast>", mode="exec"))
        function = locals()[node.name]
        
        output_stream = io.StringIO()

        # Set up a custom logging handler
        handler = logging.StreamHandler(output_stream)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
        
        # Execute it and capture the output
        with contextlib.redirect_stdout(output_stream), contextlib.redirect_stderr(output_stream):
            function()
        
        # Save the output
        output = output_stream.getvalue()
        output_dict[f"{node.name}_output"] = output.strip()

# Read the README template
template_path = "README.jinja2"
with open(template_path, "r") as f:
    template = f.read()
template = jinja2.Template(template)

# Render the template
kwargs = {**function_body_dict, **output_dict}
rendered = template.render(**kwargs)

# Write the README
with open("README.md", "w") as f:
    f.write(rendered)
    
print(rendered)