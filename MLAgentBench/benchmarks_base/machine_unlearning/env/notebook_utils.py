import inspect

def method_to_notebook_code(method):
    """Extract run() implementation and format for notebook."""
    source = inspect.getsource(method.run)
    body = "\n    ".join(line for line in source.split("\n")[1:])
    min_indent = min(len(line) - len(line.lstrip()) 
                    for line in body.split("\n") if line.strip())
    body = "\n".join(line[min_indent:] for line in body.split("\n"))
    
    return f'''def unlearning(
    net, 
    retain_loader, 
    forget_loader, 
    val_loader):
    """Generated from {method.get_name()} implementation."""
    {body}''' 