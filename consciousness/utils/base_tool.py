from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseTool(ABC):
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_required_info(self) -> List[str]:
        pass