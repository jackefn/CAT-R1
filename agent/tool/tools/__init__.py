"""
Specific tool implementations
"""

from agent.tool.tools.calculator_tool import CalculatorTool
from agent.tool.tools.wiki_search_tool import WikiSearchTool
from agent.tool.tools.select_tool import SelectTool

__all__ = [
    'CalculatorTool',
    'WikiSearchTool',
    'SelectTool',
] 

def _default_tools(env):
    if env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'select':
        return [SelectTool()]
    else:
        raise NotImplementedError
