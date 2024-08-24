import re

def test_readme():
    """Test the code snippets in the README."""
    
    # Read the README file
    with open("README.md", "r") as f:
        readme = f.read()
    
    # Get the code snippets
    code_snippets = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    
    # Test each code snippet
    for code in code_snippets:
        try:
            exec(code)
        except Exception as e:
            raise Exception(f"Error in code snippet:\n{code}\n{e}")