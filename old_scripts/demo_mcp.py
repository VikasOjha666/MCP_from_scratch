from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SimpleServer")

@mcp.tool()
def add(a: int, b: int):
    """Add two numbers."""
    return a + b

@mcp.tool()
def subtract(a: int, b: int):
    """Subtract two numbers."""
    return a - b

@mcp.tool()
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float):
    """Divide two numbers. Returns a float. Raises error if dividing by zero."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

@mcp.tool()
def celsius_to_fahrenheit(celsius: float):
    """Convert temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

@mcp.tool()
def fahrenheit_to_celsius(fahrenheit: float):
    """Convert temperature from Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9

if __name__ == "__main__":
    mcp.run("sse")


